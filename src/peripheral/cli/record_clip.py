"""Record a clip from the webcam for offline replay.

Phase 1 needs one recorded clip to run the baseline against; Phase 4 needs a set of them,
including the lighting-drift and rapid-motion-without-semantic-event cases that separate a real
scheduler from a motion detector.

**Records with the benchmark capture profile by default**, meaning exposure is pinned. On
auto-exposure the camera trades frame rate for exposure time as the room darkens (measured 30 to
10 FPS across one evening), which would bake frame-rate drift into the very clips meant to isolate
lighting drift as a false-trigger source.

    python -m peripheral.cli.record_clip --duration 60 --headless \
        --metrics-out results/record_desk.json -o record.path=data/clips/desk_60s.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from ..capture import build_source
from ..runtime import BoundedRunner
from ..telemetry.clock import Stopwatch
from ._args import parse_and_load


class RecordRunner(BoundedRunner):
    name = "record_clip"

    def setup(self) -> None:
        self.source = build_source(self.cfg)
        self.source.open()
        described = self.source.describe()
        self.recorder.note(f"frame source: {json.dumps(described)}")
        if described.get("too_dark"):
            print(
                f"WARNING: frames are near-black (mean "
                f"{described.get('warmup_frame_brightness')}/255) — light the room before "
                f"recording a clip you intend to run vision on.",
                file=sys.stderr,
            )
            self.recorder.note("WARNING: clip recorded with near-black frames")

        self.out_path = Path(self.cfg.record.path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(self.cfg.capture.target_fps)
        first = self.source.read()
        if first is None:
            raise RuntimeError("source produced no frames")
        h, w = first.image.shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.out_path), cv2.VideoWriter_fourcc(*self.cfg.record.fourcc), fps, (w, h)
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self.out_path}")
        self.writer.write(first.image)
        self.n_written = 1

    def step(self) -> bool:
        frame = self.source.read()
        if frame is None:
            return False
        self.recorder.record_frame_captured(frame.frame_id, frame.t_capture)
        with Stopwatch() as sw:
            self.writer.write(frame.image)
        self.recorder.record_stage("video_write", sw.ms, frame.frame_id)
        self.n_written += 1
        return True

    def teardown(self) -> None:
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.release()
            self.recorder.note(
                f"wrote {self.n_written} frames to {self.out_path} "
                f"({self.out_path.stat().st_size / 1024**2:.1f} MB)"
            )
        source = getattr(self, "source", None)
        if source is not None:
            source.close()


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.record_clip",
        description="Record a webcam clip with the benchmark capture profile.",
        argv=argv,
    )
    runner = RecordRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    code = runner.run()
    print(
        f"wrote {getattr(runner, 'n_written', 0)} frames to {cfg.record.path}", file=sys.stderr
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
