"""Phase 1 naive baseline: invoke the VLM on every frame it can get.

This is the run that motivates the entire project. It is *supposed* to fail, and the exit criterion
is that it fails **quantitatively**: capture holds >=25 FPS while the VLM stage is saturated, and
the resulting drop rate, staleness and latency distribution are recorded rather than described.

    python -m peripheral.cli.naive_baseline --duration 60 --headless \
        --metrics-out results/phase1_naive_webcam.json

    python -m peripheral.cli.naive_baseline --duration 60 --headless \
        --metrics-out results/phase1_naive_clip.json \
        -o capture=file -o capture.path=data/clips/desk_60s.mp4
"""

from __future__ import annotations

import json
import sys

from ..capture import build_source
from ..pipeline import Pipeline
from ..runtime import BoundedRunner
from ..vlm import build_client
from ._args import parse_and_load


class NaiveBaselineRunner(BoundedRunner):
    name = "phase1_naive_baseline"

    def setup(self) -> None:
        self.source = build_source(self.cfg)
        self.source.open()
        described = self.source.describe()
        self.recorder.note(f"frame source: {json.dumps(described)}")
        if described.get("too_dark"):
            msg = (
                f"frames are near-black (mean {described.get('warmup_frame_brightness')}/255). "
                "Timings are valid; anything vision-derived is not."
            )
            self.recorder.note(f"WARNING: {msg}")
            print(f"WARNING: {msg}", file=sys.stderr)

        # Model load happens in setup, outside the measured window — it is a startup cost, not a
        # per-frame cost, and folding it in would understate steady-state throughput.
        self.client = build_client(self.cfg)
        self.client.start()
        self.recorder.note(f"vlm: {json.dumps(self.client.describe_config())}")

        p = self.cfg.pipeline
        self.pipeline = Pipeline(
            source=self.source,
            client=self.client,
            recorder=self.recorder,
            capture_q_size=p.capture_q_size,
            vlm_q_size=p.vlm_q_size,
            answer_q_size=p.answer_q_size,
            policy=p.policy,
            prompt=p.prompt,
            max_tokens=p.max_tokens,
            queue_sample_hz=p.queue_sample_hz,
            headless=self.headless,
        )
        self.pipeline.start()

    def step(self) -> bool:
        """The pipeline runs on its own threads; this only watches for a stage dying.

        A crashed stage must end the run loudly. A pipeline that quietly loses its VLM thread would
        keep producing beautiful capture numbers and no answers at all.
        """
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
        client = getattr(self, "client", None)
        if client is not None:
            client.stop()
        source = getattr(self, "source", None)
        if source is not None:
            source.close()


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.naive_baseline",
        description="Phase 1 naive baseline: VLM on every frame, threaded pipeline.",
        argv=argv,
    )
    runner = NaiveBaselineRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    code = runner.run()

    m = runner.recorder.finalize()["metrics"]
    ttft = m["photon_to_first_token_ms"]
    print(
        f"capture {m['achieved_fps']} FPS | {m['frames_captured']} frames, "
        f"{m['frames_dropped']} dropped ({(m['drop_rate'] or 0) * 100:.1f}%) | "
        f"VLM {m['vlm_calls']} calls ({m['vlm_calls_per_min']}/min) | "
        f"photon->first-token p50 {ttft['p50'] if ttft else None} ms, "
        f"p95 {ttft['p95'] if ttft else None} ms",
        file=sys.stderr,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
