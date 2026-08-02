"""Phase 0 smoke runner — proves the telemetry path end to end.

Captures frames from the configured source and records what actually happened: capture timestamps,
inter-frame intervals, a per-stage timing, queue depth, GPU power and VRAM.

**It does not invoke a VLM, and it does not pretend to.** VLM latency, staleness, accuracy and
false-trigger metrics serialize as `null` — meaning *not measured*, not zero. Fabricating those
events to make the JSON look complete would be exactly the kind of number this project must never
produce. The recorder's ability to handle them is proven by unit tests with synthetic events.

    python -m peripheral.cli.smoke --duration 5 --headless --metrics-out results/phase0_smoke.json
    python -m peripheral.cli.smoke --duration 5 --headless -o capture=synthetic
"""

from __future__ import annotations

import json
import sys

from ..capture import build_source
from ..runtime import BoundedRunner
from ..telemetry.clock import Stopwatch
from ._args import parse_and_load


class SmokeRunner(BoundedRunner):
    name = "phase0_smoke"

    def setup(self) -> None:
        self.source = build_source(self.cfg)
        self.source.open()
        described = self.source.describe()
        self.recorder.note(f"frame source: {json.dumps(described)}")

        # Pinned exposure buys deterministic 30 FPS but goes near-black in a dim room. Timings
        # would still look perfect, so the run has to say so loudly or it will be mistaken for a
        # valid result. Every runner that opens a camera must carry this check.
        if described.get("too_dark"):
            msg = (
                f"frames are near-black (mean {described.get('warmup_frame_brightness')}/255) — "
                "pinned exposure in a dim room. Timings are valid; anything vision-derived is not. "
                "Light the room, or use -o capture=webcam_demo for a non-benchmark run."
            )
            self.recorder.note(f"WARNING: {msg}")
            print(f"WARNING: {msg}", file=sys.stderr)
        self._frames = 0

    def step(self) -> bool:
        with Stopwatch() as sw:
            frame = self.source.read()
        if frame is None:
            return False  # source exhausted; the runner records this and stops cleanly

        self.recorder.record_frame_captured(frame.frame_id, frame.t_capture)
        self.recorder.record_stage("capture_read", sw.ms, frame.frame_id)
        # Phase 0 has no queues yet. Recording a constant 0 keeps the field exercised and honest:
        # zero here means "measured, and nothing was queued", which is true.
        self.recorder.record_queue_depth("capture_out", 0, t=frame.t_capture)
        self._frames += 1
        return True

    def teardown(self) -> None:
        source = getattr(self, "source", None)
        if source is not None:
            source.close()


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.smoke",
        description="Phase 0 bounded smoke run: capture + telemetry, no VLM.",
        argv=argv,
    )
    runner = SmokeRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    code = runner.run()

    if not cfg.run.headless:
        m = runner.recorder.finalize()["metrics"]
        print(
            f"{m['frames_captured']} frames in {runner.recorder.duration_s:.2f}s "
            f"-> {m['achieved_fps']} FPS"
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
