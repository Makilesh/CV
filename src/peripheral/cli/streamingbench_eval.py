"""Phase 6: StreamingBench Real-Time Visual Understanding, through the replay harness.

Each selected clip is replayed **at wall-clock rate** with its questions arriving at their
timestamps. The scheduler decides when to spend a VLM call; when a question comes due it is answered
from whatever frame the system last looked at — and `QueryTimeline` refuses any answer whose
evidence postdates the question.

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


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".avi"}


def _find_video(root: Path, sample: int) -> Path | None:
    """Locate `sample_N`'s video.

    The archive is Mac-zipped, so it carries a `__MACOSX` tree of `._video.mp4` resource-fork stubs
    that are a few hundred bytes and not decodable. They sort *before* the real files, so a naive
    glob picks them every time and every clip appears to fail to open. Excluded explicitly, and a
    size floor guards against any other stub.
    """
    exact = root / f"sample_{sample}"
    candidates: list[Path] = []
    if exact.is_dir():
        candidates = [p for p in exact.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES]
    if not candidates:
        candidates = [
            p for p in root.rglob(f"sample_{sample}/*")
            if p.suffix.lower() in VIDEO_SUFFIXES
        ]
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

        questions = load_questions(csv_path)
        samples = group_samples(questions)
        available = {s for s in samples if _find_video(video_root, s) is not None}
        if not available:
            raise FileNotFoundError(
                f"no StreamingBench videos found under {video_root} — extract a shard first"
            )

        self.subset, self.selection = select_subset(
            samples, available, budget_s=float(sb.wall_clock_budget_s)
        )
        self.video_root = video_root
        self.recorder.note(f"StreamingBench subset: {json.dumps(self.selection)}")
        _progress(f"selected {len(self.subset)} samples, "
                  f"{self.selection['n_questions_selected']} questions, "
                  f"{self.selection['selected_replay_s']:.0f}s of replay")

        self.encoder = build_encoder(self.cfg)
        self.encoder.warmup(n=20)
        self.client = build_client(self.cfg)
        self.client.start()

        self.predictions: list[tuple[Any, str | None]] = []
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
        last_frame = None
        last_frame_t: float | None = None
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

                scores = fast.process(frame)
                ctx = FrameContext(frame.frame_id, replay_t, scores.motion, scores.novelty,
                                   scores.scene_change, None, t_last_call, n_calls)
                if policy.decide(ctx).fire:
                    # Hold the frame; the VLM is not called until a question needs it. The
                    # scheduler's job here is to decide WHICH frame is worth keeping.
                    last_frame = frame
                    last_frame_t = replay_t
                    t_last_call = replay_t
                    n_calls += 1
                    policy.observe_call(ctx)

                due = [q for q in pending if q.t <= replay_t]
                for q in due:
                    pending.remove(q)
                    if last_frame is None or last_frame_t is None:
                        self.predictions.append((q, None))
                        self.records.append({"question_id": q.question_id, "predicted": None,
                                             "correct_answer": q.answer, "reason": "no frame held"})
                        continue
                    if last_frame_t > q.t + 1e-9:
                        msg = (f"{q.question_id}: evidence t={last_frame_t:.3f} > query "
                               f"t={q.t:.3f}")
                        self.violations.append(msg)
                        raise FutureFrameError(msg)

                    with Stopwatch() as sw:
                        result = self.client.describe(
                            last_frame.image, prompt=q.prompt(),
                            max_tokens=int(self.cfg.streamingbench.max_tokens),
                        )
                    letter = extract_letter(result.text)
                    self.predictions.append((q, letter))
                    self.records.append({
                        "question_id": q.question_id,
                        "task_type": q.task_type,
                        "t": q.t,
                        "evidence_t": round(last_frame_t, 3),
                        "staleness_s": round(q.t - last_frame_t, 3),
                        "predicted": letter,
                        "correct_answer": q.answer,
                        "correct": letter == q.answer,
                        "raw": result.text[:120],
                        "vlm_ms": round(sw.ms, 1),
                    })
        finally:
            self.audits.append({"sample": sample.sample, **source.audit()})
            source.close()

        done = [r for r in self.records if r.get("question_id", "").startswith(
            f"Real-Time Visual Understanding_sample_{sample.sample}_")]
        acc = sum(1 for r in done if r.get("correct")) / len(done) if done else 0.0
        _progress(f"  sample {sample.sample:3d}: {len(done)} questions, accuracy {acc:.2f}, "
                  f"{n_calls} scheduler-selected frames")
        return True

    def teardown(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            client.stop()
        scored = score(getattr(self, "predictions", []))
        audits = getattr(self, "audits", [])
        self.recorder.record_extra("phase6_streamingbench", {
            "split": "Real-Time Visual Understanding",
            "selection": getattr(self, "selection", None),
            "score": scored,
            "per_question": getattr(self, "records", []),
            "wall_clock_audits": audits,
            "all_within_wall_clock": all(a.get("within_wall_clock", False) for a in audits),
            "future_evidence_violations": len(getattr(self, "violations", [])),
            "caveats": (
                "A SUBSET result, not a StreamingBench score. Selection is biased towards shorter "
                "clips (see selection.bias). The model answers zero-shot from ONE scheduler-selected "
                "frame, whereas published StreamingBench numbers come from models given the whole "
                "clip context — so this measures our streaming system, not the model's ceiling."
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
