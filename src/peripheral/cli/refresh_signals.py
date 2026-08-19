"""Recompute the fast-tier signals stored in Phase 4 traces, without touching the oracle answers.

Needed after the `t_presentation` fix: the stored `motion`/`novelty`/`scene_change` arrays were
computed while the rolling reference keyed off wall-clock read time, so they reflected how long the
VLM took per frame rather than the video's own timeline (see `Frame.t_presentation`).

The oracle's per-frame answers are unaffected — they depend only on the frame, not on timing — so
this recomputes the signals in seconds instead of re-running 4,320 VLM calls.

    python -m peripheral.cli.refresh_signals --duration 900 --headless \
        --metrics-out results/phase4_refresh.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..capture import FileSource
from ..eval.clips import ClipAnnotation
from ..fasttier import FastTier, build_encoder
from ..runtime import BoundedRunner
from ._args import parse_and_load


class RefreshSignalsRunner(BoundedRunner):
    name = "phase4_refresh_signals"

    def setup(self) -> None:
        p4 = self.cfg.phase4
        self.clip_dir = Path(p4.clip_dir)
        self.trace_dir = Path(p4.trace_dir)
        self.annotations = [ClipAnnotation.load(p) for p in sorted(self.clip_dir.glob("*.json"))]
        self.encoder = build_encoder(self.cfg)
        self.encoder.warmup(n=20)
        self.fast = FastTier(encoder=self.encoder,
                             motion_size=self.cfg.fasttier.motion_size,
                             half_life_s=self.cfg.fasttier.half_life_s,
                             keep_embedding=True)
        self.changes: list[dict] = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.annotations):
            return False
        ann = self.annotations[self._i]
        self._i += 1

        npz_path = self.trace_dir / f"{ann.name}.npz"
        if not npz_path.exists():
            return True
        existing = dict(np.load(npz_path))
        old_novelty = existing.get("novelty")

        src = FileSource(ann.path, realtime=False)
        src.open()
        self.fast.reset()
        motion, novelty, scene_change, embs = [], [], [], []
        try:
            while True:
                frame = src.read()
                if frame is None:
                    break
                s = self.fast.process(frame)
                motion.append(s.motion)
                novelty.append(s.novelty)
                scene_change.append(s.scene_change)
                embs.append(s.embedding)
        finally:
            src.close()

        existing["motion"] = np.asarray(motion, np.float32)
        existing["novelty"] = np.asarray(novelty, np.float32)
        existing["scene_change"] = np.asarray(scene_change, np.float32)
        existing["embeddings"] = np.asarray(embs, np.float32)
        np.savez_compressed(npz_path, **existing)

        new = np.asarray(novelty, np.float32)
        row = {
            "clip": ann.name,
            "old_novelty_p50": round(float(np.percentile(old_novelty, 50)), 5)
            if old_novelty is not None else None,
            "new_novelty_p50": round(float(np.percentile(new, 50)), 5),
            "old_novelty_max": round(float(old_novelty.max()), 5)
            if old_novelty is not None else None,
            "new_novelty_max": round(float(new.max()), 5),
        }
        self.changes.append(row)
        print(f"  {ann.name:16s} novelty p50 {row['old_novelty_p50']} -> {row['new_novelty_p50']}"
              f"   max {row['old_novelty_max']} -> {row['new_novelty_max']}", file=sys.stderr)
        return True

    def teardown(self) -> None:
        self.recorder.record_extra("phase4_refresh_signals", {
            "changes": self.changes,
            "why": (
                "The stored fast-tier signals were computed with the rolling reference keyed off "
                "wall-clock read time. During trace building the VLM took ~500 ms per frame, so "
                "the reference tracked ~15x faster than the video's own timeline and novelty came "
                "out several times too small. Oracle answers are unaffected (they depend on the "
                "frame, not on timing), so only the signals are recomputed."
            ),
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.refresh_signals",
        description="Recompute Phase 4 trace signals against stream time.",
        argv=argv,
    )
    runner = RefreshSignalsRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                 metrics_out=cfg.run.metrics_out,
                                 headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
