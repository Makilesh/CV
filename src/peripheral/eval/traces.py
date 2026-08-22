"""Per-clip traces: fast-tier scores and per-frame oracle answers.

**Why traces exist.** Sweeping 5 policies × 9 operating points × 6 clips means ~270 replays. Doing
that against a live VLM would be days of GPU time. But the per-frame oracle already answers *every*
frame, so any policy that decides to call on frame *i* can be given the oracle's answer for frame
*i*. That makes the sweep exact rather than approximate — the same model, the same frame, the same
prompt — and reduces it to a lookup.

The one thing it is not is a latency measurement. **The sweep measures accuracy and call rate
only**; latency comes from Phase 3, where it was measured end to end on the real pipeline. That
separation is stated wherever sweep results appear.

Building a trace is the expensive step: one VLM call per frame per clip. It is cached to disk and
reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .clips import ClipAnnotation

TRACE_SCHEMA = 1


@dataclass
class ClipTrace:
    """Everything needed to replay any policy over one clip without touching the GPU."""

    clip_name: str
    fps: float
    n_frames: int
    motion: np.ndarray
    novelty: np.ndarray
    scene_change: np.ndarray
    answers: list[str]          # per-frame oracle answer
    state_ids: np.ndarray
    encoder: dict[str, Any]
    model: str
    prompt: str
    schema_version: int = TRACE_SCHEMA
    #: Phase 8b patch novelty, when it has been computed. Kept beside the pooled signal so both
    #: exist for identical frames and can be compared without a paired-run confound.
    novelty_patch: np.ndarray | None = None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path.with_suffix(".npz"),
            motion=self.motion,
            novelty=self.novelty,
            scene_change=self.scene_change,
            state_ids=self.state_ids,
        )
        path.with_suffix(".json").write_text(
            json.dumps({
                "clip_name": self.clip_name,
                "fps": self.fps,
                "n_frames": self.n_frames,
                "answers": self.answers,
                "encoder": self.encoder,
                "model": self.model,
                "prompt": self.prompt,
                "schema_version": self.schema_version,
            }, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "ClipTrace":
        path = Path(path)
        arr = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta.get("schema_version") != TRACE_SCHEMA:
            raise ValueError(f"trace schema mismatch for {path}")
        return ClipTrace(
            novelty_patch=arr["novelty_patch"] if "novelty_patch" in arr else None,
            clip_name=meta["clip_name"],
            fps=meta["fps"],
            n_frames=meta["n_frames"],
            motion=arr["motion"],
            novelty=arr["novelty"],
            scene_change=arr["scene_change"],
            answers=meta["answers"],
            state_ids=arr["state_ids"],
            encoder=meta["encoder"],
            model=meta["model"],
            prompt=meta["prompt"],
        )

    def with_signal(self, signal: str) -> "ClipTrace":
        """Return a view of this trace whose `novelty` is the requested signal.

        Swapping the signal rather than teaching every policy about a second field means the whole
        Phase 4 sweep, the simulator and the ablations compare pooled against spatial with
        *identical* code paths — the comparison cannot be contaminated by a branch.
        """
        if signal in ("novelty", "pooled"):
            return self
        if signal in ("patch", "novelty_patch"):
            if self.novelty_patch is None:
                raise ValueError(
                    f"{self.clip_name} has no patch novelty; run peripheral.cli.add_spatial_signals"
                )
            return replace(self, novelty=self.novelty_patch)
        raise ValueError(f"unknown signal {signal!r}")

    @staticmethod
    def exists(path: str | Path) -> bool:
        path = Path(path)
        return path.with_suffix(".npz").exists() and path.with_suffix(".json").exists()


def build_trace(
    annotation: ClipAnnotation,
    fast_tier: Any,
    client: Any,
    prompt: str,
    max_tokens: int = 32,
    progress: Any = None,
) -> ClipTrace:
    """Run the fast tier and the per-frame oracle over one clip.

    The clip is read unpaced: this is offline trace construction, not a streaming run, and pacing
    would only make it take real time for no benefit. Wall-clock replay is enforced where it
    matters — the live pipeline (Phase 1) and the Phase 6 replay harness.
    """
    from ..capture import FileSource

    src = FileSource(annotation.path, realtime=False)
    src.open()
    fast_tier.reset()

    motion, novelty, scene_change, answers = [], [], [], []
    try:
        idx = 0
        while True:
            frame = src.read()
            if frame is None:
                break
            scores = fast_tier.process(frame)
            motion.append(scores.motion)
            novelty.append(scores.novelty)
            scene_change.append(scores.scene_change)

            r = client.describe(frame.image, prompt, max_tokens=max_tokens)
            answers.append(r.text)

            idx += 1
            if progress is not None and idx % 60 == 0:
                progress(f"    {annotation.name}: {idx}/{annotation.n_frames} frames")
    finally:
        src.close()

    n = len(answers)
    return ClipTrace(
        clip_name=annotation.name,
        fps=annotation.fps,
        n_frames=n,
        motion=np.asarray(motion, dtype=np.float32),
        novelty=np.asarray(novelty, dtype=np.float32),
        scene_change=np.asarray(scene_change, dtype=np.float32),
        answers=answers,
        state_ids=np.asarray(annotation.state_ids[:n], dtype=np.int32),
        encoder=fast_tier.encoder.describe(),
        model=getattr(client, "model_name", "unknown"),
        prompt=prompt,
    )
