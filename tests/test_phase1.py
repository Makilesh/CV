"""Phase 1 exit criteria.

`PROMPT.md`: *test asserting capture sustains >=25 FPS with the VLM stage saturated, plus
`results/phase1_naive_baseline.png`. Make the failure quantitative.*

Phase 1 must show:
  1. bounded queues implement each backpressure policy exactly, and record every drop;
  2. the capture thread never blocks on inference (the invariant-7 test);
  3. a stage that dies takes the run down loudly instead of hanging it quietly;
  4. capture holds >=25 FPS with the VLM stage saturated, on real hardware;
  5. the motivating chart exists and is derived from real metrics files.

The heavyweight tests are marked `vlm` and skip when the llama.cpp binary or the model weights are
absent, so a fresh clone without ~500 MB of weights still runs the rest of the suite.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from peripheral.capture import Frame
from peripheral.cli._args import load_config
from peripheral.pipeline import BackpressurePolicy, BoundedQueue, Pipeline
from peripheral.runtime import EXIT_FAILED, EXIT_OK
from peripheral.telemetry import MetricsRecorder, schema
from peripheral.telemetry.schema import validate_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _frame(i: int) -> Frame:
    return Frame(frame_id=i, t_capture=float(i), image=np.zeros((4, 4, 3), np.uint8))


def _vlm_available(cfg) -> bool:
    return Path(cfg.vlm.binary).exists() and Path(cfg.vlm.model).exists()


# --------------------------------------------------------------------------------------------
# 1. Backpressure policies
# --------------------------------------------------------------------------------------------
def test_drop_oldest_keeps_the_freshest_frames():
    """The correct policy for video: an answer about a stale frame is worth less."""
    drops = []
    q = BoundedQueue("q", 2, BackpressurePolicy.DROP_OLDEST, on_drop=lambda f, s, r: drops.append((f, r)))
    for i in range(5):
        assert q.put(_frame(i), frame_id=i) is True

    survivors = [q.get(timeout=0.1).frame_id for _ in range(2)]
    assert survivors == [3, 4], "the two newest frames must survive"
    assert [f for f, _ in drops] == [0, 1, 2]
    assert all(r == "queue_full_drop_oldest" for _, r in drops)


def test_drop_newest_preserves_the_backlog():
    drops = []
    q = BoundedQueue("q", 2, BackpressurePolicy.DROP_NEWEST, on_drop=lambda f, s, r: drops.append((f, r)))
    for i in range(5):
        q.put(_frame(i), frame_id=i)

    survivors = [q.get(timeout=0.1).frame_id for _ in range(2)]
    assert survivors == [0, 1], "the oldest frames must survive"
    assert [f for f, _ in drops] == [2, 3, 4]


def test_block_policy_makes_the_producer_wait():
    """Retained only so its cost can be measured. It violates invariant 7 by design."""
    q = BoundedQueue("q", 1, BackpressurePolicy.BLOCK)
    assert q.put(_frame(0), frame_id=0) is True

    t0 = time.perf_counter()
    assert q.put(_frame(1), frame_id=1, timeout=0.15) is False  # times out, producer stalled
    assert time.perf_counter() - t0 >= 0.1, "block policy must actually block"


def test_drain_newest_discards_the_backlog_and_records_it():
    """The VLM only ever wants the current state of the world."""
    drops = []
    q = BoundedQueue("q", 8, on_drop=lambda f, s, r: drops.append((f, r)))
    for i in range(5):
        q.put(_frame(i), frame_id=i)

    newest = q.drain_newest()
    assert newest.frame_id == 4
    assert q.depth == 0
    assert [f for f, _ in drops] == [0, 1, 2, 3]
    assert all(r == "superseded_by_newer" for _, r in drops)


def test_every_drop_reaches_the_metrics_file():
    """A dropped frame is a policy decision. If it is not recorded, it is an accident."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    q = BoundedQueue(
        "vlm_q", 1, BackpressurePolicy.DROP_OLDEST,
        on_drop=lambda f, s, r: rec.record_frame_dropped(f if f is not None else -1, s, r),
    )
    for i in range(4):
        rec.record_frame_captured(i, t=float(i))
        q.put(_frame(i), frame_id=i)
    rec.finish(schema.STATUS_COMPLETED)

    doc = rec.finalize()
    assert doc["metrics"]["frames_dropped"] == 3
    assert doc["metrics"]["drop_rate"] == pytest.approx(3 / 7, abs=1e-4)
    assert {d["reason"] for d in doc["events"]["drops"]} == {"queue_full_drop_oldest"}


def test_put_never_raises_when_full():
    """Overflow is expected operation for a real-time source, not an error condition."""
    q = BoundedQueue("q", 1, BackpressurePolicy.DROP_OLDEST)
    for i in range(1000):
        assert q.put(_frame(i), frame_id=i) is True


def test_queue_rejects_zero_size():
    with pytest.raises(ValueError):
        BoundedQueue("q", 0)


# --------------------------------------------------------------------------------------------
# 2. Invariant 7 — the capture thread never blocks on inference
# --------------------------------------------------------------------------------------------
def test_producer_is_not_slowed_by_a_stalled_consumer():
    """The headline invariant, tested without needing a GPU.

    A consumer that never consumes must not slow the producer down. With drop_oldest the producer
    should run at full speed and the drops should be recorded.
    """
    q = BoundedQueue("q", 2, BackpressurePolicy.DROP_OLDEST)

    n = 2000
    t0 = time.perf_counter()
    for i in range(n):
        q.put(_frame(i), frame_id=i)
    elapsed = time.perf_counter() - t0

    assert q.n_dropped == n - 2
    # 2000 puts into a never-drained queue must be fast; a blocking implementation would take
    # 2000 x timeout instead.
    assert elapsed < 1.0, f"producer was throttled by the stalled consumer: {elapsed:.2f}s"


# --------------------------------------------------------------------------------------------
# 3. A dead stage is loud
# --------------------------------------------------------------------------------------------
class _ExplodingSource:
    def read(self):
        raise RuntimeError("camera unplugged")

    def close(self):
        pass


def test_a_dying_stage_is_surfaced_not_swallowed():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    pipe = Pipeline(
        source=_ExplodingSource(), client=None, recorder=rec,
        capture_q_size=2, vlm_q_size=2, answer_q_size=2,
    )
    pipe.start()
    deadline = time.perf_counter() + 5.0
    while pipe.failed_stage is None and time.perf_counter() < deadline:
        time.sleep(0.02)
    failed = pipe.failed_stage
    pipe.stop()

    assert failed is not None, "a stage that raised must be reported"
    assert failed.stage_name == "capture"
    assert isinstance(failed.error, RuntimeError)


def test_pipeline_stop_is_idempotent_and_joins_threads():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()

    class _Idle:
        def read(self):
            time.sleep(0.005)
            return None

        def close(self):
            pass

    pipe = Pipeline(source=_Idle(), client=None, recorder=rec)
    pipe.start()
    time.sleep(0.1)
    pipe.stop()
    pipe.stop()  # must not raise
    assert all(not s.is_alive() for s in pipe.stages)


# --------------------------------------------------------------------------------------------
# 4. Config
# --------------------------------------------------------------------------------------------
def test_pipeline_defaults_do_not_block_capture():
    cfg = load_config("config", [])
    assert cfg.pipeline.policy == "drop_oldest", "the default must never stall the capture thread"
    assert cfg.pipeline.vlm_q_size <= 4, (
        "a deep VLM queue converts dropped frames into stale answers, which is worse — "
        "the staleness hides inside latency instead of showing up as a drop"
    )


def test_vlm_config_points_at_the_cuda_13_3_build():
    """sm_120 (Blackwell) needs CUDA 12.8+; the 12.4 asset predates it."""
    cfg = load_config("config", [])
    assert "llama-server" in str(cfg.vlm.binary)
    assert cfg.vlm.n_gpu_layers >= 1, "a CPU-only baseline would not be a GPU latency measurement"


# --------------------------------------------------------------------------------------------
# 5. THE EXIT CRITERION — capture sustains >=25 FPS with the VLM saturated
# --------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.vlm
def test_capture_sustains_25fps_with_vlm_saturated(tmp_path):
    """Phase 1 exit criterion, on real hardware with the real model.

    Runs the naive baseline against the recorded clip (deterministic input, no dependence on
    room lighting) and asserts the capture thread was not dragged down by the saturated VLM stage.
    """
    from peripheral.cli import naive_baseline

    cfg = load_config("config", [])
    if not _vlm_available(cfg):
        pytest.skip("llama-server binary or model weights not present")
    clip = REPO_ROOT / "data" / "clips" / "desk_60s.mp4"
    if not clip.exists():
        pytest.skip("recorded clip not present; run peripheral.cli.record_clip")

    out = tmp_path / "phase1.json"
    code = naive_baseline.main([
        "--duration", "20", "--headless", "--metrics-out", str(out),
        "-o", "capture=file", "-o", f"capture.path={clip.as_posix()}",
    ])
    assert code == EXIT_OK

    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_metrics(doc)
    m = doc["metrics"]

    # The criterion.
    assert m["achieved_fps"] >= 25.0, (
        f"capture was dragged down to {m['achieved_fps']} FPS by the VLM stage — "
        "the pipeline is coupled and every downstream number measures the wrong system"
    )
    # The VLM stage must actually have been saturated, or the test proves nothing.
    assert m["vlm_calls"] > 0, "no VLM calls: the slow stage was not exercised"
    vlm_rate = m["vlm_calls_per_min"] / 60.0
    assert vlm_rate < m["achieved_fps"], (
        f"VLM kept up ({vlm_rate:.1f} calls/s vs {m['achieved_fps']} FPS) — "
        "this is not a saturated baseline and the motivating chart would be wrong"
    )
    assert m["frames_dropped"] > 0, "a saturated pipeline must be dropping frames"
    # Latency must include everything, so it cannot plausibly be model-only.
    assert m["photon_to_answer_ms"]["p50"] > m["photon_to_first_token_ms"]["p50"]


@pytest.mark.vlm
def test_the_motivating_chart_exists():
    """`results/phase1_naive_baseline.png` is half the Phase 1 exit criterion."""
    png = RESULTS / "phase1_naive_baseline.png"
    if not (RESULTS / "phase1_naive_webcam.json").exists():
        pytest.skip("baseline metrics not generated in this checkout")
    assert png.exists(), "run peripheral.viz.phase1_chart to build it"
    assert png.stat().st_size > 50_000, "chart looks truncated"


@pytest.mark.vlm
def test_baseline_metrics_show_a_quantitative_failure():
    """The chart must be backed by numbers, and the numbers must show the failure."""
    path = RESULTS / "phase1_naive_webcam.json"
    if not path.exists():
        pytest.skip("baseline metrics not generated in this checkout")
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_metrics(doc)
    m = doc["metrics"]

    assert m["achieved_fps"] >= 25.0
    assert m["photon_to_answer_ms"]["p50"] > 33.3, (
        "if photon-to-answer fits inside a frame budget there is nothing to schedule around"
    )
    assert m["power"]["energy_per_query_j"] is not None, "energy per query is a Phase 1 deliverable"
    assert m["queue_depth"]["vlm_q"]["series"], "queue depth over time is a Phase 1 deliverable"
