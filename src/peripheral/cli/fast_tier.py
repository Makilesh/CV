"""Phase 2 fast tier: motion, embedding, novelty and scene-change on every frame, VLM disabled.

`PROMPT.md`: *must sustain 30 FPS with headroom, VLM disabled. Profile per-stage and report the
millisecond budget.*

Two runs are needed to answer that honestly, because a paced source cannot show headroom — it can
only show that we keep up with it:

    # sustained: does the fast tier hold the camera's rate?
    python -m peripheral.cli.fast_tier --duration 30 --headless \
        --metrics-out results/phase2_fasttier_webcam.json

    # headroom: how fast could it go if the source did not limit it?
    python -m peripheral.cli.fast_tier --duration 30 --headless \
        --metrics-out results/phase2_fasttier_unpaced.json \
        -o capture=file -o capture.path=data/clips/desk_60s.mp4 -o capture.realtime=false
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ..capture import build_source
from ..fasttier import FastTier, build_encoder
from ..pipeline import Pipeline
from ..runtime import BoundedRunner
from ._args import parse_and_load


class FastTierRunner(BoundedRunner):
    name = "phase2_fast_tier"

    def setup(self) -> None:
        self.source = build_source(self.cfg)
        self.source.open()
        described = self.source.describe()
        self.recorder.note(f"frame source: {json.dumps(described)}")
        if described.get("too_dark"):
            print(
                f"WARNING: frames are near-black (mean "
                f"{described.get('warmup_frame_brightness')}/255) — timings are valid, "
                "novelty and scene-change scores are not.",
                file=sys.stderr,
            )
            self.recorder.note("WARNING: near-black frames; vision-derived scores invalid")

        self.encoder = build_encoder(self.cfg)
        # Warmup outside the measured window: lazy CUDA kernels and allocator paths would
        # otherwise land in the p99 and read as pipeline jitter.
        self.encoder.warmup(n=20)
        self.recorder.note(f"encoder: {json.dumps(self.encoder.describe())}")

        ft = self.cfg.fasttier
        self.fast = FastTier(
            encoder=self.encoder,
            motion_size=ft.motion_size,
            half_life_s=ft.half_life_s,
            keep_embedding=False,
        )

        p = self.cfg.pipeline
        self.pipeline = Pipeline(
            source=self.source,
            client=None,
            recorder=self.recorder,
            capture_q_size=p.capture_q_size,
            vlm_q_size=p.vlm_q_size,
            answer_q_size=p.answer_q_size,
            policy=p.policy,
            queue_sample_hz=p.queue_sample_hz,
            headless=self.headless,
            fast_tier=self.fast,
            enable_vlm=False,
        )
        self.pipeline.start()

    def step(self) -> bool:
        failed = self.pipeline.failed_stage
        if failed is not None:
            raise RuntimeError(f"stage {failed.stage_name!r} died: {failed.error!r}")
        if self.pipeline.stop_event.wait(0.05):
            return False
        return True

    def teardown(self) -> None:
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            pipeline.stop()
            self.recorder.note(f"pipeline stats: {json.dumps(pipeline.stats())}")

            scores = pipeline.fast_tier.scores
            if scores:
                arr = np.asarray([(s[2], s[3], s[4]) for s in scores], dtype=float)
                self.recorder.record_extra(
                    "fast_tier",
                    {
                        "encoder": self.encoder.describe(),
                        "n_scored": len(scores),
                        "motion": _summary(arr[:, 0]),
                        "novelty": _summary(arr[:, 1]),
                        "scene_change": _summary(arr[:, 2]),
                        "series_columns": ["frame_id", "t_capture", "motion", "novelty",
                                           "scene_change"],
                        "series": [[s[0], round(s[1], 6), round(s[2], 6), round(s[3], 6),
                                    round(s[4], 6)] for s in scores],
                    },
                )
        source = getattr(self, "source", None)
        if source is not None:
            source.close()


def _summary(a: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(a.mean()), 6),
        "p50": round(float(np.percentile(a, 50)), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
        "p99": round(float(np.percentile(a, 99)), 6),
        "max": round(float(a.max()), 6),
    }


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.fast_tier",
        description="Phase 2 fast tier: per-frame scoring with the VLM disabled.",
        argv=argv,
    )
    runner = FastTierRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    code = runner.run()

    m = runner.recorder.finalize()["metrics"]
    stages = m["stages_ms"] or {}
    ft = stages.get("fast_tier", {})
    print(
        f"capture {m['achieved_fps']} FPS | {m['frames_captured']} frames, "
        f"{m['frames_dropped']} dropped | fast_tier p50 {ft.get('p50')} ms, "
        f"p95 {ft.get('p95')} ms, p99 {ft.get('p99')} ms",
        file=sys.stderr,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
