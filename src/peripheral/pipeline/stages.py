"""Pipeline stages, one thread each.

Threads, not `multiprocessing`: Windows spawns rather than forks, so a process worker reloads the
CUDA context and costs VRAM we do not have (12 GB is the binding constraint). The GIL is not a
problem here because every stage that matters spends its time in OpenCV, in socket waits, or in
llama.cpp — all of which release it.

Stage graph (Phase 1):

    CaptureStage ──capture_q──> FastTierStage ──vlm_q──> VlmStage ──answer_q──> RenderStage

`FastTierStage` is a pass-through in Phase 1 and is replaced with real scoring in Phase 2. It
exists now because retrofitting a stage into a running pipeline means re-measuring everything.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

from ..telemetry.clock import Stopwatch, now
from ..telemetry.metrics import MetricsRecorder
from .queues import BoundedQueue


class Stage(threading.Thread, ABC):
    """One pipeline stage on its own thread.

    Subclasses implement `tick()`. The loop, stop signalling and exception capture live here, so a
    crashed stage takes down the run loudly instead of hanging it quietly.
    """

    def __init__(self, name: str, recorder: MetricsRecorder, stop_event: threading.Event) -> None:
        super().__init__(name=name, daemon=True)
        self.stage_name = name
        self.recorder = recorder
        self.stop_event = stop_event
        self.error: BaseException | None = None
        self.n_processed = 0

    @abstractmethod
    def tick(self) -> None:
        """One unit of work. Must return promptly so the stop signal is honoured."""

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.tick()
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-surfaced by the runner
            self.error = exc
            self.stop_event.set()


class CaptureStage(Stage):
    """Reads frames and hands them downstream. **Never blocks on inference.**

    This thread's achieved FPS is the Phase 1 exit criterion: it must hold >=25 FPS while the VLM
    stage is saturated. If it does not, the pipeline is coupled and every downstream number is a
    measurement of the wrong system.
    """

    def __init__(
        self,
        source: Any,
        out_q: BoundedQueue,
        recorder: MetricsRecorder,
        stop_event: threading.Event,
    ) -> None:
        super().__init__("capture", recorder, stop_event)
        self.source = source
        self.out_q = out_q

    def tick(self) -> None:
        with Stopwatch() as sw:
            frame = self.source.read()
        if frame is None:
            self.stop_event.set()
            return
        self.recorder.record_frame_captured(frame.frame_id, frame.t_capture)
        self.recorder.record_stage("capture_read", sw.ms, frame.frame_id)
        self.out_q.put(frame, frame_id=frame.frame_id)
        self.n_processed += 1


class FastTierStage(Stage):
    """Phase 1: pass-through. Phase 2: frame differencing, embedding, novelty and scene-change.

    It still records its own timing so the Phase 2 budget has a Phase 1 baseline to beat, and so
    the per-stage millisecond breakdown exists from the first measured run.
    """

    def __init__(
        self,
        in_q: BoundedQueue,
        out_q: BoundedQueue,
        recorder: MetricsRecorder,
        stop_event: threading.Event,
    ) -> None:
        super().__init__("fast_tier", recorder, stop_event)
        self.in_q = in_q
        self.out_q = out_q

    def tick(self) -> None:
        frame = self.in_q.get(timeout=0.1)
        if frame is None:
            return
        with Stopwatch() as sw:
            pass  # Phase 2 fills this in
        self.recorder.record_stage("fast_tier", sw.ms, frame.frame_id)
        self.out_q.put(frame, frame_id=frame.frame_id)
        self.n_processed += 1


class VlmStage(Stage):
    """Invokes the VLM. The slow stage, by a factor of ten.

    In the naive baseline this fires on **every frame it can get**, which is the whole point: it
    saturates, the queue overflows, and the drop rate and staleness that result are the motivating
    measurement.

    It takes the *newest* queued frame and discards the backlog. Answering about a stale frame
    would be strictly worse than answering about the current one, and it would flatter the latency
    numbers by hiding queue wait inside "processing".
    """

    def __init__(
        self,
        client: Any,
        in_q: BoundedQueue,
        out_q: BoundedQueue,
        recorder: MetricsRecorder,
        stop_event: threading.Event,
        prompt: str,
        max_tokens: int = 32,
    ) -> None:
        super().__init__("vlm", recorder, stop_event)
        self.client = client
        self.in_q = in_q
        self.out_q = out_q
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._call_seq = 0

    def tick(self) -> None:
        frame = self.in_q.drain_newest()
        if frame is None:
            self.stop_event.wait(0.002)
            return

        call_id = f"c{self._call_seq}"
        self._call_seq += 1

        self.recorder.vlm_call_start(
            call_id, frame_id=frame.frame_id, model=self.client.model_name
        )
        try:
            result = self.client.describe(
                frame.image,
                prompt=self.prompt,
                max_tokens=self.max_tokens,
                on_first_token=lambda: self.recorder.vlm_call_first_token(call_id),
            )
        except Exception as exc:  # noqa: BLE001 - one bad call must not kill the run
            self.recorder.note(f"vlm call {call_id} failed: {type(exc).__name__}: {exc}")
            self.recorder.vlm_call_complete(call_id, gen_tokens=None)
            return

        self.recorder.vlm_call_complete(call_id, gen_tokens=result.n_tokens)
        self.recorder.record_stage("vlm_total", result.total_ms, frame.frame_id)
        self.recorder.record_stage("vlm_encode_jpeg", result.encode_ms, frame.frame_id)
        self.recorder.record_answer(
            answer_id=call_id,
            evidence_frame_id=frame.frame_id,
            source="vlm",
            call_id=call_id,
            text=result.text,
        )
        self.out_q.put(result, frame_id=frame.frame_id)
        self.n_processed += 1


class RenderStage(Stage):
    """Consumes answers. Headless: records timing only; Phase 7 adds the HUD.

    Present from Phase 1 so display cost is inside the measured pipeline from the start rather
    than appearing as a surprise regression when the GUI lands.
    """

    def __init__(
        self,
        in_q: BoundedQueue,
        recorder: MetricsRecorder,
        stop_event: threading.Event,
        headless: bool = True,
    ) -> None:
        super().__init__("render", recorder, stop_event)
        self.in_q = in_q
        self.headless = headless
        self.latest: Any | None = None

    def tick(self) -> None:
        item = self.in_q.get(timeout=0.1)
        if item is None:
            return
        with Stopwatch() as sw:
            self.latest = item
        self.recorder.record_stage("render", sw.ms)
        self.n_processed += 1


class QueueDepthSampler(threading.Thread):
    """Samples every queue on a fixed cadence.

    Recording depth only on `put` would sample exactly when the queue is longest and produce a
    biased time series. Queue depth over time is a Phase 1 deliverable, so it gets an unbiased
    sampler.
    """

    def __init__(
        self,
        queues: list[BoundedQueue],
        recorder: MetricsRecorder,
        stop_event: threading.Event,
        hz: float = 20.0,
    ) -> None:
        super().__init__(name="queue-sampler", daemon=True)
        self.queues = queues
        self.recorder = recorder
        self.stop_event = stop_event
        self.period = 1.0 / hz

    def run(self) -> None:
        while not self.stop_event.is_set():
            t = now()
            for q in self.queues:
                self.recorder.record_queue_depth(q.name, q.depth, t=t)
            self.stop_event.wait(self.period)
