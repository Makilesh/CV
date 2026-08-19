"""Frame sources.

**Deliberately synchronous (STATUS.md decision D6).** The threaded pipeline with bounded queues and
backpressure is Phase 1's deliverable; Phase 0 only needs *a* source to prove the telemetry path
end to end. Phase 1 wraps these in a capture thread — it does not rewrite them.

Three sources:

* `webcam`    — the real thing. `CAP_DSHOW` by default (measured: same 30 FPS as MSMF, p99
                inter-frame 51 ms vs 65 ms; jitter is what hurts a capture thread).
* `file`      — a recorded clip. **Paced to wall clock**, because reading a file as fast as it
                decodes is not streaming evaluation and silently inflates every number.
* `synthetic` — deterministic generated frames at a target rate. No camera, no clip, no GPU:
                the CI runner and the deterministic tests use this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from ..telemetry.clock import now

BACKENDS = {
    "CAP_DSHOW": cv2.CAP_DSHOW,
    "CAP_MSMF": cv2.CAP_MSMF,
    "CAP_ANY": cv2.CAP_ANY,
}


@dataclass
class Frame:
    """One captured frame.

    `t_capture` is the instant the frame became available to us — the t0 of every latency in the
    system. It is *not* photon arrival: sensor exposure and USB transport add an unmeasured
    constant offset. See `telemetry.schema.DEFINITIONS['t0']`.
    """

    frame_id: int
    t_capture: float
    image: np.ndarray
    #: Fast-tier scores, attached by FastTierStage. None until Phase 2's fast tier runs.
    scores: Any | None = None
    #: The frame's position in the STREAM's timeline, in seconds. Defaults to `t_capture`.
    #:
    #: These differ, and conflating them is a real bug we shipped and had to fix. `t_capture` is
    #: when *we* got the frame; `t_presentation` is when it happened. For a live camera they are
    #: the same. For a file read offline they are not: during Phase 4 trace building the VLM took
    #: ~500 ms per frame, so consecutive frames were "captured" 500 ms apart while actually being
    #: 33 ms apart in the video. Any time-constant signal — the fast tier's rolling reference has a
    #: half-life in seconds — then behaves as though the video were 15x slower, and novelty came
    #: out ~4x too small. Latency metrics must use `t_capture`; signal processing must use
    #: `t_presentation`.
    t_presentation: float | None = None

    @property
    def t_stream(self) -> float:
        """Presentation time, falling back to capture time for live sources."""
        return self.t_capture if self.t_presentation is None else self.t_presentation

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape


class FrameSource:
    """Common interface. `read()` returns the next Frame, or None when exhausted."""

    name = "source"

    def open(self) -> None:  # pragma: no cover - trivial
        pass

    def read(self) -> Frame | None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def describe(self) -> dict[str, Any]:
        return {"source": self.name}

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame


class WebcamSource(FrameSource):
    name = "webcam"

    #: Mean pixel value below which a frame is too dark to be worth sending to a VLM.
    #: Pinned exposure in a dim room produces a near-black image (measured: 1.6/255) that would
    #: silently destroy accuracy while every timing number still looks perfect.
    DARK_FRAME_THRESHOLD = 25.0

    def __init__(
        self,
        device_index: int = 0,
        backend: str = "CAP_DSHOW",
        width: int = 640,
        height: int = 480,
        fourcc: str | None = "MJPG",
        warmup_frames: int = 5,
        auto_exposure: float | None = 0.25,
        exposure: float | None = -5.0,
    ) -> None:
        self.device_index = int(device_index)
        self.backend = backend
        self.width = int(width)
        self.height = int(height)
        self.fourcc = fourcc
        self.warmup_frames = int(warmup_frames)
        self.auto_exposure = auto_exposure
        self.exposure = exposure
        self._cap: cv2.VideoCapture | None = None
        self._next_id = 0
        self._actual: dict[str, Any] = {}

    def open(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; expected one of {list(BACKENDS)}")
        cap = cv2.VideoCapture(self.device_index, BACKENDS[self.backend])
        if not cap.isOpened():
            raise RuntimeError(
                f"could not open camera {self.device_index} via {self.backend}. "
                "Check the device is present and not held by another application."
            )
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Pin exposure. MEASURED 2026-08-01: left on auto, this camera trades frame rate for
        # exposure time as the room darkens — 30 FPS in good light, 19.9 FPS dimmer, 10 FPS
        # dimmer still, with no error and no warning. Every FPS-threshold exit criterion in this
        # project would silently depend on the time of day. Manual exposure at -5 (1/32 s = 31 ms,
        # the longest exposure that fits a 33 ms frame budget) gives a repeatable 30.1 FPS.
        if self.auto_exposure is not None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)
        if self.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

        # Warm up: the first frames after open carry driver negotiation latency that would
        # otherwise land in the p99 and look like pipeline jitter.
        sample = None
        for _ in range(self.warmup_frames):
            ok, img = cap.read()
            if ok:
                sample = img
        self._cap = cap
        brightness = float(sample.mean()) if sample is not None else None
        self._actual = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            # DSHOW reports -1 here; FPS comes from config, never from the driver.
            "driver_reported_fps": cap.get(cv2.CAP_PROP_FPS),
            "exposure_readback": cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_exposure_readback": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "warmup_frame_brightness": round(brightness, 2) if brightness is not None else None,
            # Load-bearing: a run whose frames are black produces perfect timings and worthless
            # answers. This flag is what stops that from passing as a result.
            "too_dark": (
                None if brightness is None else bool(brightness < self.DARK_FRAME_THRESHOLD)
            ),
        }

    def read(self) -> Frame | None:
        if self._cap is None:
            raise RuntimeError("read() before open()")
        ok, image = self._cap.read()
        t = now()  # timestamp immediately on return, before any processing
        if not ok:
            return None
        frame = Frame(self._next_id, t, image)
        self._next_id += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "device_index": self.device_index,
            "backend": self.backend,
            "requested": f"{self.width}x{self.height}",
            "fourcc": self.fourcc,
            **self._actual,
        }


class FileSource(FrameSource):
    """A recorded clip, paced to wall clock.

    Invariant 1: batch-processing a video file is not streaming evaluation. This source refuses to
    hand out frames faster than their presentation timestamps allow. Phase 6 builds the full replay
    harness (with hard no-future-frames enforcement); this is the honest minimum until then.
    """

    name = "file"

    def __init__(self, path: str | Path, realtime: bool = True, loop: bool = False) -> None:
        self.path = Path(path)
        self.realtime = bool(realtime)
        self.loop = bool(loop)
        self._cap: cv2.VideoCapture | None = None
        self._next_id = 0
        self._t_first: float | None = None
        self._fps: float = 30.0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"clip not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open clip: {self.path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        self._cap = cap

    def read(self) -> Frame | None:
        if self._cap is None:
            raise RuntimeError("read() before open()")
        ok, image = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, image = self._cap.read()
            if not ok:
                return None

        if self.realtime:
            # Hold the frame until its presentation time. Sleeping here is the point: it is what
            # makes this a stream rather than a batch job.
            if self._t_first is None:
                self._t_first = now()
            due = self._t_first + self._next_id / self._fps
            _sleep_until(due)

        # t_presentation comes from the clip's own timeline, so time-constant signals behave the
        # same whether the clip is replayed live or processed offline. See Frame.t_presentation.
        frame = Frame(self._next_id, now(), image, t_presentation=self._next_id / self._fps)
        self._next_id += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "path": str(self.path),
            "clip_fps": self._fps,
            "realtime_paced": self.realtime,
        }


class SyntheticSource(FrameSource):
    """Deterministic frames at a target rate — no camera, no clip, no GPU.

    Used by the tests and by CI (CPU-only, gates correctness not performance). The pattern moves so
    that frame differencing and novelty scoring have something real to chew on in Phase 2.
    """

    name = "synthetic"

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        target_fps: float = 30.0,
        seed: int = 1337,
        realtime: bool = True,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.target_fps = float(target_fps)
        self.seed = int(seed)
        self.realtime = bool(realtime)
        self._next_id = 0
        self._t_first: float | None = None
        self._rng = np.random.default_rng(seed)
        self._noise = None

    def open(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # One fixed noise field, so successive frames differ only by the moving object —
        # a scene with a controllable amount of semantic change.
        self._noise = self._rng.integers(
            0, 40, size=(self.height, self.width, 3), dtype=np.uint8
        )

    def read(self) -> Frame | None:
        if self.realtime:
            if self._t_first is None:
                self._t_first = now()
            _sleep_until(self._t_first + self._next_id / self.target_fps)

        image = self._noise.copy() if self._noise is not None else np.zeros(
            (self.height, self.width, 3), np.uint8
        )
        # A box tracking a circular path: smooth motion, no semantic event.
        phase = self._next_id / max(1.0, self.target_fps)
        cx = int(self.width / 2 + self.width / 4 * np.cos(phase))
        cy = int(self.height / 2 + self.height / 4 * np.sin(phase))
        cv2.rectangle(image, (cx - 30, cy - 30), (cx + 30, cy + 30), (200, 180, 60), -1)

        frame = Frame(self._next_id, now(), image,
                      t_presentation=self._next_id / self.target_fps)
        self._next_id += 1
        return frame

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "size": f"{self.width}x{self.height}",
            "target_fps": self.target_fps,
            "seed": self.seed,
            "realtime_paced": self.realtime,
        }


def _sleep_until(deadline: float) -> None:
    """Sleep to `deadline` on the monotonic clock.

    `time.sleep` on Windows has coarse granularity (~1–15 ms depending on timer resolution), so the
    last millisecond is spun. Overshoot here becomes capture jitter, which is exactly the number
    Phase 1 reports — we cannot afford to be sloppy about it.
    """
    import time

    remaining = deadline - now()
    if remaining <= 0:
        return
    if remaining > 0.002:
        time.sleep(remaining - 0.002)
    while now() < deadline:
        pass


def build_source(cfg: Any) -> FrameSource:
    """Construct the source named by `cfg.capture.source`. No hardcoded model IDs or paths."""
    cap = cfg["capture"] if not hasattr(cfg, "capture") else cfg.capture
    kind = str(cap["source"])

    if kind == "webcam":
        return WebcamSource(
            device_index=cap.get("device_index", 0),
            backend=cap.get("backend", "CAP_DSHOW"),
            width=cap.get("width", 640),
            height=cap.get("height", 480),
            fourcc=cap.get("fourcc", "MJPG"),
            warmup_frames=cap.get("warmup_frames", 5),
            auto_exposure=cap.get("auto_exposure", 0.25),
            exposure=cap.get("exposure", -5.0),
        )
    if kind == "file":
        path = cap.get("path")
        if not path:
            raise ValueError("capture.source=file requires capture.path")
        return FileSource(path, realtime=cap.get("realtime", True), loop=cap.get("loop", False))
    if kind == "synthetic":
        return SyntheticSource(
            width=cap.get("width", 640),
            height=cap.get("height", 480),
            target_fps=cap.get("target_fps", 30.0),
            seed=cap.get("seed", 1337),
            realtime=cap.get("realtime", True),
        )
    raise ValueError(f"unknown capture.source {kind!r}; expected webcam, file or synthetic")
