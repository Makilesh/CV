"""Phase 6: replay a clip at wall-clock rate through the real pipeline, answering timed queries.

This is the honest end-to-end run: real capture pacing, real fast tier, real scheduler, real VLM,
real queries arriving at timestamps. Every enforcement gate in `peripheral.eval.replay` is active,
and the audit is written into the metrics file whether it passes or fails.

    python -m peripheral.cli.replay_eval --duration 60 --headless \
        --metrics-out results/phase6_replay_mixed.json \
        -o phase6.clip=data/eval_clips/mixed.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..eval.replay import (
    FutureFrameError,
    QueryTimeline,
    ReplayClock,
    ReplaySource,
    uniform_queries,
)
from ..fasttier import FastTier, build_encoder
from ..runtime import BoundedRunner
from ..scheduler.policies import build_policy
from ..scheduler.policies import FrameContext
from ..telemetry.clock import Stopwatch, now
from ..vlm import build_client
from ._args import parse_and_load


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


class ReplayEvalRunner(BoundedRunner):
    """Single-threaded on purpose.

    The Phase 1 pipeline runs capture on its own thread, which is correct for a live camera. Here the
    point is auditability: one thread means the ordering of "frame became available" and "answer was
    produced" is a program order a reader can verify line by line, which is what `PROMPT.md` asks
    for. The threaded pipeline is exercised by Phase 1 and 2 runs and by the ablations below.
    """

    name = "phase6_replay_eval"

    def setup(self) -> None:
        p6 = self.cfg.phase6
        self.clock = ReplayClock(speed=float(p6.replay_speed))
        self.source = ReplaySource(
            p6.clip, clock=self.clock, late_tolerance_ms=float(p6.late_tolerance_ms)
        )
        self.source.open()
        self.recorder.note(f"replay source: {json.dumps(self.source.describe())}")

        duration = (self.source.n_frames_total / self.source.fps) if self.source.n_frames_total else 24.0
        self.timeline = QueryTimeline(
            queries=uniform_queries(duration, float(p6.query_every_s), str(p6.query_text)),
            strict=True,
        )

        self.encoder = build_encoder(self.cfg)
        self.encoder.warmup(n=20)
        self.fast = FastTier(
            encoder=self.encoder,
            motion_size=self.cfg.fasttier.motion_size,
            half_life_s=self.cfg.fasttier.half_life_s,
        )

        self.use_fast_tier = bool(p6.use_fast_tier)
        self.policy = build_policy(
            str(p6.policy_kind),
            float(p6.policy_value),
            **(dict(min_gap_s=float(self.cfg.phase4.min_gap_s))
               if str(p6.policy_kind) != "fixed_interval" else {}),
        )
        self.policy.reset()

        self.client = build_client(self.cfg)
        self.client.start()
        self.recorder.note(f"vlm: {json.dumps(self.client.describe_config())}")

        self.t_last_call: float | None = None
        self.n_calls = 0
        self.current_answer: str | None = None
        self.evidence_frame: int | None = None
        self.evidence_t: float | None = None
        self._frames = 0

    def step(self) -> bool:
        frame = self.source.read()
        if frame is None:
            return False
        self._frames += 1
        replay_t = self.source.due_time(frame.frame_id)

        self.recorder.record_frame_captured(frame.frame_id, frame.t_capture)

        # --- fast tier -------------------------------------------------------------------
        motion = novelty = scene_change = 0.0
        with Stopwatch() as sw:
            if self.use_fast_tier:
                scores = self.fast.process(frame)
                motion, novelty, scene_change = (
                    scores.motion, scores.novelty, scores.scene_change
                )
        self.recorder.record_stage("fast_tier", sw.ms, frame.frame_id)

        # --- scheduler -------------------------------------------------------------------
        ctx = FrameContext(
            frame_idx=frame.frame_id,
            t=replay_t,
            motion=motion,
            novelty=novelty,
            scene_change=scene_change,
            embedding=None,
            t_last_call=self.t_last_call,
            n_calls=self.n_calls,
        )
        decision = self.policy.decide(ctx)
        self.recorder.record_trigger(
            frame.frame_id, self.policy.name, decision.fire, decision.score, decision.reason
        )

        if decision.fire:
            call_id = f"c{self.n_calls}"
            self.recorder.vlm_call_start(call_id, frame_id=frame.frame_id,
                                         model=self.client.model_name)
            try:
                result = self.client.describe(
                    frame.image,
                    prompt=str(self.cfg.phase6.query_text),
                    max_tokens=int(self.cfg.phase6.max_tokens),
                    on_first_token=lambda: self.recorder.vlm_call_first_token(call_id),
                )
            except Exception as exc:  # noqa: BLE001
                self.recorder.note(f"vlm call {call_id} failed: {type(exc).__name__}: {exc}")
                self.recorder.vlm_call_complete(call_id, gen_tokens=None)
                return True

            self.recorder.vlm_call_complete(call_id, gen_tokens=result.n_tokens)
            self.recorder.record_stage("vlm_total", result.total_ms, frame.frame_id)
            self.recorder.record_answer(call_id, evidence_frame_id=frame.frame_id,
                                        source="vlm", call_id=call_id, text=result.text)
            # Evidence is the frame we actually looked at, timestamped in REPLAY time so the
            # query gate can compare it against query times on the same axis.
            self.current_answer = result.text
            self.evidence_frame = frame.frame_id
            self.evidence_t = replay_t
            self.t_last_call = replay_t
            self.n_calls += 1
            self.policy.observe_call(ctx)

        # --- answer any queries whose time has come --------------------------------------
        for query in self.timeline.due(replay_t):
            self.timeline.answer(
                query,
                answer=self.current_answer,
                evidence_frame=self.evidence_frame,
                evidence_t=self.evidence_t,
                answered_at=replay_t,
            )
        return True

    def teardown(self) -> None:
        source = getattr(self, "source", None)
        audit = source.audit() if source is not None else {}
        if source is not None:
            source.close()
        client = getattr(self, "client", None)
        if client is not None:
            client.stop()

        timeline = getattr(self, "timeline", None)
        self.recorder.record_extra("phase6_replay", {
            "clip": str(self.cfg.phase6.clip),
            "policy": self.policy.describe() if hasattr(self, "policy") else None,
            "use_fast_tier": getattr(self, "use_fast_tier", None),
            "wall_clock_audit": audit,
            "queries": timeline.summary() if timeline else None,
            "answers": [
                {
                    "query_id": a.query.query_id, "t": a.query.t,
                    "evidence_frame": a.evidence_frame, "staleness_s": a.staleness_s,
                    "answer": (a.answer or "")[:160],
                }
                for a in (timeline.answered if timeline else [])
            ],
            "enforcement": (
                "Three gates: (1) the decoder never runs ahead of the replay clock, so future "
                "frames are absent rather than withheld; (2) frame_at() raises FutureFrameError on "
                "explicit forward access; (3) QueryTimeline.answer() refuses evidence timestamped "
                "after the query. wall_clock_audit.within_wall_clock must be true."
            ),
        })
        if not audit.get("within_wall_clock", True):
            self.recorder.note("WARNING: source delivered more frames than wall clock allowed")


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.replay_eval",
        description="Phase 6: wall-clock replay with hard no-future-frames enforcement.",
        argv=argv,
    )
    runner = ReplayEvalRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                             metrics_out=cfg.run.metrics_out,
                             headless=cfg.run.headless, seed=cfg.seed)
    code = runner.run()

    extra = runner.recorder.finalize()["extra"]["phase6_replay"]
    audit, q = extra["wall_clock_audit"], extra["queries"]
    _progress(
        f"replay audit: {audit['frames_delivered']} frames in {audit['elapsed_s']}s "
        f"(clock allowed {audit['max_frames_allowed_by_clock']}) "
        f"within_wall_clock={audit['within_wall_clock']} | "
        f"queries {q['n_answered']}/{q['n_queries']} answered, "
        f"{q['future_evidence_violations']} future-evidence violations, "
        f"mean staleness {q['staleness_s_mean']}s"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
