"""Phase 6: StreamingBench Real-Time Visual Understanding, through the replay harness.

Each selected clip is replayed **at wall-clock rate** with its questions arriving at their
timestamps. The scheduler decides which frame is worth holding; when a question comes due it is
answered from the frame the system was holding **at that moment**.

**The ordering in `step()` is the invariant, not a detail.** A question due strictly before the
current frame arrived must be answered from the *previous* held frame. Answering it after processing
the arriving frame hands it evidence from its own future — which is precisely the violation Gate 3
caught here on the first real run (`evidence t=20.020 > query t=20.000`). Our own 24-second clips
masked it because 30 fps puts a frame exactly on every 2-second query; a real benchmark's frame rate
does not cooperate.

    python -m peripheral.cli.streamingbench_eval --duration 3600 --headless \
        --metrics-out results/phase6_streamingbench.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..eval.replay import FutureFrameError, ReplayClock, ReplaySource
from ..eval.streamingbench import (
    SBQuestion,
    extract_letter,
    group_samples,
    load_questions,
    score,
    select_subset,
)
from ..fasttier import FastTier, build_encoder
from ..runtime import BoundedRunner
from ..scheduler.policies import FrameContext, build_policy
from ..telemetry.clock import Stopwatch
from ..vlm import build_client
from ._args import parse_and_load

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi"}


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _find_video(root: Path, sample: int) -> Path | None:
    """Locate `sample_N`'s video.

    The archive is Mac-zipped, so it carries a `__MACOSX` tree of `._video.mp4` resource-fork stubs
    that are a few hundred bytes and not decodable. They sort *before* the real files, so a naive
    glob picks them every time and every clip appears to fail to open.
    """
    exact = root / f"sample_{sample}"
    candidates: list[Path] = []
    if exact.is_dir():
        candidates = [p for p in exact.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
    if not candidates:
        candidates = [p for p in root.rglob(f"sample_{sample}/*")
                      if p.suffix.lower() in VIDEO_SUFFIXES]
    candidates = [
        p for p in candidates
        if "__MACOSX" not in p.parts and not p.name.startswith("._") and p.stat().st_size > 100_000
    ]
    return sorted(candidates, key=lambda p: -p.stat().st_size)[0] if candidates else None


class StreamingBenchRunner(BoundedRunner):
    name = "phase6_streamingbench"

    def setup(self) -> None:
        sb = self.cfg.streamingbench
        csv_path = Path(sb.csv)
        video_root = Path(sb.video_root)
        if not csv_path.exists():
            raise FileNotFoundError(f"StreamingBench annotations not found: {csv_path}")
        if not video_root.exists():
            raise FileNotFoundError(f"StreamingBench video root not found: {video_root}")

        samples = group_samples(load_questions(csv_path))
        available = {s for s in samples if _find_video(video_root, s) is not None}
        if not available:
            raise FileNotFoundError(f"no StreamingBench videos under {video_root}")

        self.subset, self.selection = select_subset(
            samples, available, budget_s=float(sb.wall_clock_budget_s)
        )
        self.video_root = video_root
        self.strict = bool(sb.get("strict_future_check", True))
        self.recorder.note(f"StreamingBench subset: {json.dumps(self.selection)}")
        _progress(f"selected {len(self.subset)} samples, "
                  f"{self.selection['n_questions_selected']} questions, "
                  f"{self.selection['selected_replay_s']:.0f}s of replay")

        self.encoder = build_encoder(self.cfg)
        self.encoder.warmup(n=20)
        self.client = build_client(self.cfg)
        self.client.start()

        self.predictions: list[tuple[SBQuestion, str | None]] = []
        self.records: list[dict[str, Any]] = []
        self.audits: list[dict[str, Any]] = []
        self.violations: list[str] = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.subset):
            return False
        sample = self.subset[self._i]
        self._i += 1

        path = _find_video(self.video_root, sample.sample)
        if path is None:
            return True

        clock = ReplayClock(speed=float(self.cfg.phase6.replay_speed))
        source = ReplaySource(path, clock=clock)
        source.open()

        fast = FastTier(encoder=self.encoder,
                        motion_size=self.cfg.fasttier.motion_size,
                        half_life_s=self.cfg.fasttier.half_life_s)
        policy = build_policy(str(self.cfg.phase6.policy_kind),
                              float(self.cfg.phase6.policy_value),
                              min_gap_s=float(self.cfg.phase4.min_gap_s))
        policy.reset()

        pending = list(sample.questions)
        held_frame = None            # the frame the scheduler last chose to keep
        held_t: float | None = None
        prev_held_frame, prev_held_t = None, None
        t_last_call: float | None = None
        n_calls = 0
        deadline = sample.last_t

        try:
            while pending:
                frame = source.read()
                if frame is None:
                    break
                replay_t = source.due_time(frame.frame_id)
                if replay_t > deadline + 1.0:
                    break

                # 1. Questions due STRICTLY BEFORE this frame arrived are answered from the frame
                #    held at that time — the previous one. This is the ordering Gate 3 enforces.
                for q in [q for q in pending if q.t < replay_t]:
                    pending.remove(q)
                    self._answer(q, prev_held_frame, prev_held_t)

                # 2. Now this frame exists: fast tier, then the scheduler decides to keep it or not.
                scores = fast.process(frame)
                ctx = FrameContext(frame.frame_id, replay_t, scores.motion, scores.novelty,
                                   scores.scene_change, None, t_last_call, n_calls)
                if policy.decide(ctx).fire:
                    held_frame, held_t = frame, replay_t
                    t_last_call = replay_t
                    n_calls += 1
                    policy.observe_call(ctx)

                # 3. A question due exactly at this frame's timestamp may use it — the same instant
                #    is present, not future.
                for q in [q for q in pending if q.t <= replay_t]:
                    pending.remove(q)
                    self._answer(q, held_frame, held_t)

                prev_held_frame, prev_held_t = held_frame, held_t
        finally:
            self.audits.append({"sample": sample.sample, **source.audit()})
            source.close()

        done = [r for r in self.records if r["question_id"].startswith(
            f"Real-Time Visual Understanding_sample_{sample.sample}_")]
        acc = sum(1 for r in done if r.get("correct")) / len(done) if done else 0.0
        _progress(f"  sample {sample.sample:3d}: {len(done)} questions, accuracy {acc:.2f}, "
                  f"{n_calls} scheduler-selected frames")
        return True

    def _answer(self, q: SBQuestion, frame: Any, frame_t: float | None) -> None:
        """Answer one question from the held frame, refusing evidence from its future."""
        if frame is None or frame_t is None:
            self.predictions.append((q, None))
            self.records.append({
                "question_id": q.question_id, "task_type": q.task_type, "t": q.t,
                "predicted": None, "correct_answer": q.answer, "correct": False,
                "reason": "no frame held yet",
            })
            return

        if frame_t > q.t + 1e-9:
            msg = f"{q.question_id}: evidence t={frame_t:.3f} > query t={q.t:.3f}"
            self.violations.append(msg)
            if self.strict:
                raise FutureFrameError(msg)

        with Stopwatch() as sw:
            result = self.client.describe(
                frame.image, prompt=q.prompt(),
                max_tokens=int(self.cfg.streamingbench.max_tokens),
            )
        letter = extract_letter(result.text)
        self.predictions.append((q, letter))
        self.records.append({
            "question_id": q.question_id,
            "task_type": q.task_type,
            "t": q.t,
            "evidence_t": round(frame_t, 3),
            "staleness_s": round(q.t - frame_t, 3),
            "predicted": letter,
            "correct_answer": q.answer,
            "correct": letter == q.answer,
            "raw": result.text[:120],
            "vlm_ms": round(sw.ms, 1),
        })

    def teardown(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            client.stop()
        scored = score(getattr(self, "predictions", []))
        audits = getattr(self, "audits", [])
        stale = [r["staleness_s"] for r in getattr(self, "records", []) if "staleness_s" in r]
        self.recorder.record_extra("phase6_streamingbench", {
            "split": "Real-Time Visual Understanding",
            "selection": getattr(self, "selection", None),
            "score": scored,
            "staleness_s_mean": round(sum(stale) / len(stale), 3) if stale else None,
            "staleness_s_max": round(max(stale), 3) if stale else None,
            "per_question": getattr(self, "records", []),
            "wall_clock_audits": audits,
            "all_within_wall_clock": all(a.get("within_wall_clock", False) for a in audits),
            "future_evidence_violations": len(getattr(self, "violations", [])),
            "caveats": (
                "A SUBSET result, not a StreamingBench score. Selection is biased towards shorter "
                "clips (see selection.bias). The model answers zero-shot from ONE "
                "scheduler-selected frame, whereas published StreamingBench numbers come from "
                "models given the whole clip — so this measures our streaming system, not the "
                "model's ceiling."
            ),
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.streamingbench_eval",
        description="Phase 6: StreamingBench RTVU subset through the wall-clock replay harness.",
        argv=argv,
    )
    runner = StreamingBenchRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                 metrics_out=cfg.run.metrics_out,
                                 headless=cfg.run.headless, seed=cfg.seed)
    code = runner.run()
    s = runner.recorder.finalize()["extra"]["phase6_streamingbench"]["score"]
    _progress(f"\nStreamingBench RTVU subset: {s['correct']}/{s['n']} = {s['accuracy']} "
              f"(random baseline {s['random_baseline']})")
    return code


if __name__ == "__main__":
    sys.exit(main())
