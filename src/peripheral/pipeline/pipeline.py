"""Wires the stages into a running pipeline and owns their lifecycle."""

from __future__ import annotations

import threading
from typing import Any

from ..telemetry.metrics import MetricsRecorder
from .queues import BackpressurePolicy, BoundedQueue
from .stages import (
    CaptureStage,
    FastTierStage,
    QueueDepthSampler,
    RenderStage,
    Stage,
    VlmStage,
)


class Pipeline:
    """Capture -> fast tier -> VLM -> render, joined by bounded queues.

    The queues are where the design decisions live: sizes and backpressure policy determine what
    gets dropped when the VLM cannot keep up, which in the naive baseline is always.
    """

    def __init__(
        self,
        source: Any,
        client: Any,
        recorder: MetricsRecorder,
        capture_q_size: int = 4,
        vlm_q_size: int = 2,
        answer_q_size: int = 8,
        policy: str = BackpressurePolicy.DROP_OLDEST.value,
        prompt: str = "Describe what you see in one short sentence.",
        max_tokens: int = 32,
        queue_sample_hz: float = 20.0,
        headless: bool = True,
    ) -> None:
        self.recorder = recorder
        self.stop_event = threading.Event()

        def on_drop(frame_id: int | None, stage: str, reason: str) -> None:
            recorder.record_frame_dropped(frame_id if frame_id is not None else -1, stage, reason)

        self.capture_q = BoundedQueue("capture_q", capture_q_size, policy, on_drop=on_drop)
        self.vlm_q = BoundedQueue("vlm_q", vlm_q_size, policy, on_drop=on_drop)
        # The answer queue never applies backpressure to capture, so it drops the newest rather
        # than evicting an answer the render stage may be about to show.
        self.answer_q = BoundedQueue(
            "answer_q", answer_q_size, BackpressurePolicy.DROP_NEWEST, on_drop=on_drop
        )

        self.capture = CaptureStage(source, self.capture_q, recorder, self.stop_event)
        self.fast_tier = FastTierStage(self.capture_q, self.vlm_q, recorder, self.stop_event)
        self.vlm = VlmStage(
            client, self.vlm_q, self.answer_q, recorder, self.stop_event, prompt, max_tokens
        )
        self.render = RenderStage(self.answer_q, recorder, self.stop_event, headless=headless)

        self.stages: list[Stage] = [self.capture, self.fast_tier, self.vlm, self.render]
        self.sampler = QueueDepthSampler(
            [self.capture_q, self.vlm_q, self.answer_q], recorder, self.stop_event, queue_sample_hz
        )

    def start(self) -> None:
        for stage in self.stages:
            stage.start()
        self.sampler.start()

    def stop(self, join_timeout: float = 10.0) -> None:
        self.stop_event.set()
        for q in (self.capture_q, self.vlm_q, self.answer_q):
            q.close()
        for stage in self.stages:
            stage.join(timeout=join_timeout)
        self.sampler.join(timeout=2.0)

    @property
    def failed_stage(self) -> Stage | None:
        """A stage that died. The runner surfaces this rather than reporting a clean run."""
        return next((s for s in self.stages if s.error is not None), None)

    def stats(self) -> dict[str, Any]:
        return {
            "queues": [q.stats() for q in (self.capture_q, self.vlm_q, self.answer_q)],
            "stages": {s.stage_name: s.n_processed for s in self.stages},
        }
