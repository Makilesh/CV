"""Phase 0 exit criteria.

`PROMPT.md`: *phase exit criteria are pytest tests, not claims in a report.*

Phase 0 must show:
  1. the metrics schema is enforced, not merely documented;
  2. every metric family named in PROMPT.md is recorded and derived correctly;
  3. unmeasured metrics serialize as null, never as a flattering zero;
  4. the bounded-headless-runner contract holds, including when the run raises;
  5. a 5-second bounded run emits a valid metrics JSON.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from peripheral.capture import SyntheticSource, build_source
from peripheral.cli import smoke
from peripheral.cli._args import build_parser, load_config
from peripheral.runtime import EXIT_FAILED, EXIT_OK, BoundedRunner
from peripheral.telemetry import MetricsRecorder, schema
from peripheral.telemetry.metrics import _summary
from peripheral.telemetry.power import _trapezoid
from peripheral.telemetry.schema import MetricsSchemaError, validate_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------
# 1. Schema is enforced
# --------------------------------------------------------------------------------------------
def test_valid_document_passes_validation():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.finish(schema.STATUS_COMPLETED)
    validate_metrics(rec.finalize())


@pytest.mark.parametrize("missing", schema.REQUIRED_TOP_LEVEL)
def test_missing_top_level_key_rejected(missing):
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.finish(schema.STATUS_COMPLETED)
    doc = rec.finalize()
    doc.pop(missing)
    with pytest.raises(MetricsSchemaError, match=missing):
        validate_metrics(doc)


@pytest.mark.parametrize("missing", schema.REQUIRED_METRIC_KEYS)
def test_missing_metric_family_rejected(missing):
    """Every metric family in PROMPT.md must be present, so none can be quietly dropped later."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.finish(schema.STATUS_COMPLETED)
    doc = rec.finalize()
    doc["metrics"].pop(missing)
    with pytest.raises(MetricsSchemaError, match=missing):
        validate_metrics(doc)


def test_empty_summary_must_be_null_not_zero_filled():
    """An n=0 percentile block is the exact shape of a lie; the schema refuses it."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.finish(schema.STATUS_COMPLETED)
    doc = rec.finalize()
    doc["metrics"]["photon_to_answer_ms"] = {
        "n": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0
    }
    with pytest.raises(MetricsSchemaError, match="n=0"):
        validate_metrics(doc)


def test_unknown_status_rejected():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    with pytest.raises(ValueError):
        rec.finish("mostly_fine")


# --------------------------------------------------------------------------------------------
# 2. Derived metrics are correct
# --------------------------------------------------------------------------------------------
def test_photon_latency_measured_from_frame_capture():
    """photon->first-token and photon->answer are measured from t0 of the *triggering* frame."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.record_frame_captured(frame_id=7, t=100.0)
    rec.vlm_call_start("c0", frame_id=7, t=100.05)     # 50 ms of scheduling before the call
    rec.vlm_call_first_token("c0", t=100.40)           # -> 400 ms photon-to-first-token
    rec.vlm_call_complete("c0", gen_tokens=12, t=100.90)  # -> 900 ms photon-to-answer
    rec.finish(schema.STATUS_COMPLETED)

    m = rec.finalize()["metrics"]
    assert m["photon_to_first_token_ms"]["p50"] == pytest.approx(400.0, abs=1e-3)
    assert m["photon_to_answer_ms"]["p50"] == pytest.approx(900.0, abs=1e-3)
    assert m["vlm_calls"] == 1


def test_latency_excludes_calls_with_unknown_source_frame():
    """A call we cannot attribute to a captured frame must not contribute a bogus latency."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.vlm_call_start("orphan", frame_id=999, t=10.0)
    rec.vlm_call_first_token("orphan", t=10.2)
    rec.finish(schema.STATUS_COMPLETED)

    m = rec.finalize()["metrics"]
    assert m["photon_to_first_token_ms"] is None
    assert m["vlm_calls"] == 1


def test_staleness_is_age_of_evidence_not_age_of_answer():
    """A cache hit's staleness is the age of the evidence frame, not of the cache entry."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.record_frame_captured(frame_id=1, t=50.0)   # evidence observed here
    rec.record_frame_captured(frame_id=2, t=52.0)   # current frame at answer time
    rec.record_answer("a0", evidence_frame_id=1, source="cache", t=52.5)
    rec.finish(schema.STATUS_COMPLETED)

    m = rec.finalize()["metrics"]
    assert m["answer_staleness_ms"]["p50"] == pytest.approx(2500.0, abs=1e-3)


def test_false_trigger_rate_counts_only_labelled_fires():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.record_trigger(1, "novelty", fired=True, semantic_change=True)    # good fire
    rec.record_trigger(2, "novelty", fired=True, semantic_change=False)   # false trigger
    rec.record_trigger(3, "novelty", fired=True, semantic_change=False)   # false trigger
    rec.record_trigger(4, "novelty", fired=True, semantic_change=None)    # unlabelled
    rec.record_trigger(5, "novelty", fired=False, semantic_change=False)  # correctly suppressed
    rec.finish(schema.STATUS_COMPLETED)

    ft = rec.finalize()["metrics"]["false_trigger_rate"]
    assert ft["rate"] == pytest.approx(2 / 3, abs=1e-4)  # rate is rounded to 4dp on serialize
    assert ft["n_false_triggers"] == 2
    assert ft["n_labelled_fires"] == 3
    assert ft["n_unlabelled_fires"] == 1


def test_false_trigger_rate_is_null_without_labels():
    """An unlabelled run reports null — never a flattering zero."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.record_trigger(1, "novelty", fired=True)
    rec.finish(schema.STATUS_COMPLETED)
    assert rec.finalize()["metrics"]["false_trigger_rate"] is None


def test_drop_rate_and_fps():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    for i in range(9):
        rec.record_frame_captured(i, t=100.0 + i * 0.01)
    rec.record_frame_dropped(9, stage="vlm_queue", reason="backpressure")
    rec.finish(schema.STATUS_COMPLETED)

    m = rec.finalize()["metrics"]
    assert m["frames_captured"] == 9
    assert m["frames_dropped"] == 1
    assert m["drop_rate"] == pytest.approx(0.1)
    assert m["capture_interval_ms"]["p50"] == pytest.approx(10.0, abs=1e-6)


def test_vlm_calls_per_minute():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    for i in range(3):
        rec.vlm_call_start(f"c{i}")
    rec.finish(schema.STATUS_COMPLETED)
    doc = rec.finalize()
    assert doc["metrics"]["vlm_calls"] == 3
    assert doc["metrics"]["vlm_calls_per_min"] == pytest.approx(
        3.0 / (rec.duration_s / 60.0), rel=1e-3
    )


def test_summary_percentiles_on_known_population():
    s = _summary([float(x) for x in range(1, 101)])
    assert s["n"] == 100
    assert s["min"] == 1.0 and s["max"] == 100.0
    assert s["p50"] == pytest.approx(50.5)
    assert s["p95"] == pytest.approx(95.05)
    assert s["p99"] == pytest.approx(99.01)


def test_summary_of_empty_population_is_none():
    assert _summary([]) is None


def test_energy_integration_trapezoid():
    """Constant 100 W over 2 s is 200 J; the ramp is integrated, not averaged."""
    assert _trapezoid([0.0, 1.0, 2.0], [100.0, 100.0, 100.0]) == pytest.approx(200.0)
    assert _trapezoid([0.0, 2.0], [0.0, 100.0]) == pytest.approx(100.0)


def test_stage_timings_recorded_per_stage():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    for ms in (1.0, 2.0, 3.0):
        rec.record_stage("fast_tier", ms)
    rec.record_stage("vlm", 250.0)
    rec.finish(schema.STATUS_COMPLETED)

    stages = rec.finalize()["metrics"]["stages_ms"]
    assert stages["fast_tier"]["n"] == 3
    assert stages["fast_tier"]["p50"] == pytest.approx(2.0)
    assert stages["vlm"]["max"] == pytest.approx(250.0)


# --------------------------------------------------------------------------------------------
# 3. null means not-measured
# --------------------------------------------------------------------------------------------
def test_unmeasured_families_are_null_not_zero():
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.record_frame_captured(0, t=1.0)
    rec.finish(schema.STATUS_COMPLETED)

    m = rec.finalize()["metrics"]
    for key in (
        "photon_to_first_token_ms",
        "photon_to_answer_ms",
        "answer_staleness_ms",
        "accuracy_vs_oracle",
        "false_trigger_rate",
        "trigger_count",
    ):
        assert m[key] is None, f"{key} must be null when not measured, got {m[key]!r}"
    # Counts of things that genuinely did not happen are real zeros, and stay zeros.
    assert m["vlm_calls"] == 0
    assert m["frames_dropped"] == 0


def test_definitions_block_is_shipped_with_every_file():
    """A number without its definition is not a result."""
    rec = MetricsRecorder(run_name="t", seed=1)
    rec.start()
    rec.finish(schema.STATUS_COMPLETED)
    defs = rec.finalize()["definitions"]
    assert "lower bound" in defs["t0"].lower()
    assert "null" in defs["null_vs_zero"].lower()


def test_environment_block_supports_reproducibility():
    rec = MetricsRecorder(run_name="t", seed=99)
    rec.start()
    rec.finish(schema.STATUS_COMPLETED)
    env = rec.finalize()["environment"]
    assert env["seed"] == 99
    assert env["python"].startswith("3.1")
    for key in ("platform", "gpu_name", "git_commit", "peripheral_version"):
        assert key in env


# --------------------------------------------------------------------------------------------
# 4. The bounded-runner contract
# --------------------------------------------------------------------------------------------
class _CountingRunner(BoundedRunner):
    name = "counting"

    def setup(self):
        self.n = 0

    def step(self) -> bool:
        self.n += 1
        self.recorder.record_frame_captured(self.n)
        return True


class _ExplodingRunner(BoundedRunner):
    name = "exploding"

    def step(self) -> bool:
        raise RuntimeError("boom")


class _ExhaustingRunner(BoundedRunner):
    name = "exhausting"

    def step(self) -> bool:
        return False


def test_runner_is_bounded_and_exits_zero(tmp_path, synthetic_cfg):
    out = tmp_path / "m.json"
    runner = _CountingRunner(cfg=synthetic_cfg, duration_s=0.5, metrics_out=out)
    assert runner.run() == EXIT_OK
    assert out.exists()

    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_metrics(doc)
    assert doc["run"]["status"] == schema.STATUS_COMPLETED
    assert doc["run"]["duration_requested_s"] == 0.5
    assert 0.4 <= doc["run"]["duration_actual_s"] <= 1.2  # bounded, with slack for Windows timers


def test_runner_writes_metrics_when_the_run_raises(tmp_path, synthetic_cfg):
    """A crash must still leave numbers behind — and must be unmistakably marked as a crash."""
    out = tmp_path / "failed.json"
    runner = _ExplodingRunner(cfg=synthetic_cfg, duration_s=0.5, metrics_out=out)
    assert runner.run() == EXIT_FAILED

    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_metrics(doc)
    assert doc["run"]["status"] == schema.STATUS_FAILED
    assert "boom" in doc["run"]["failure"]


def test_runner_stops_cleanly_when_source_is_exhausted(tmp_path, synthetic_cfg):
    out = tmp_path / "short.json"
    runner = _ExhaustingRunner(cfg=synthetic_cfg, duration_s=5.0, metrics_out=out)
    assert runner.run() == EXIT_OK
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["run"]["duration_actual_s"] < 1.0
    assert any("ended early" in n for n in doc["run"]["notes"])


class _SlowTeardownRunner(BoundedRunner):
    """Teardown costs 400 ms — like releasing a DSHOW camera (measured ~250 ms)."""

    name = "slow_teardown"

    def setup(self):
        self.n = 0

    def step(self) -> bool:
        self.n += 1
        self.recorder.record_frame_captured(self.n)
        return True

    def teardown(self):
        import time

        time.sleep(0.4)


def test_measured_window_excludes_teardown(tmp_path, synthetic_cfg):
    """Rate metrics must not be deflated by cleanup time.

    Regression: a real 5 s webcam run read 152 frames at a true 30.2 FPS but reported 28.7 FPS,
    purely because `cap.release()` was inside the measured window.
    """
    out = tmp_path / "m.json"
    runner = _SlowTeardownRunner(cfg=synthetic_cfg, duration_s=1.0, metrics_out=out)
    assert runner.run() == EXIT_OK

    doc = json.loads(out.read_text(encoding="utf-8"))
    # Window is the 1 s loop, not 1.4 s including teardown.
    assert doc["run"]["duration_actual_s"] < 1.2, (
        f"teardown leaked into the measured window: {doc['run']['duration_actual_s']}s"
    )


def test_metrics_out_parent_directories_are_created(tmp_path, synthetic_cfg):
    out = tmp_path / "deep" / "nested" / "m.json"
    assert _CountingRunner(cfg=synthetic_cfg, duration_s=0.2, metrics_out=out).run() == EXIT_OK
    assert out.exists()


def test_power_block_present_whether_or_not_nvml_exists(tmp_path, synthetic_cfg):
    """CI is CPU-only. The power block must exist and say why it is empty, not vanish."""
    out = tmp_path / "m.json"
    _CountingRunner(cfg=synthetic_cfg, duration_s=0.5, metrics_out=out).run()
    power = json.loads(out.read_text(encoding="utf-8"))["metrics"]["power"]
    assert "available" in power
    if not power["available"]:
        assert power["error"], "unavailable NVML must record why"


# --------------------------------------------------------------------------------------------
# 5. CLI contract and config
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["--duration", "--headless", "--metrics-out"])
def test_mandated_flags_exist(flag):
    """Invariant 4. If this fails, the project can no longer evaluate its own work."""
    opts = {s for a in build_parser("p", "d")._actions for s in a.option_strings}
    assert flag in opts


def test_duration_is_required():
    with pytest.raises(SystemExit):
        build_parser("p", "d").parse_args(["--headless"])


def test_config_composes_and_overrides_apply():
    cfg = load_config("config", ["capture=synthetic", "seed=7", "capture.target_fps=15"])
    assert cfg.capture.source == "synthetic"
    assert cfg.capture.target_fps == 15
    assert cfg.seed == 7


def test_default_capture_config_matches_measured_backend_choice():
    """DSHOW was chosen on measured p99 jitter (51 ms vs 65 ms). Guard the decision."""
    cfg = load_config("config", [])
    assert cfg.capture.source == "webcam"
    assert cfg.capture.backend == "CAP_DSHOW"


def test_benchmark_webcam_profile_pins_exposure():
    """Measured: on auto-exposure this camera runs 30/20/10 FPS as the room darkens, silently.

    Every FPS-threshold exit criterion in this project depends on exposure being pinned, so the
    default benchmark profile must never drift back to auto.
    """
    cfg = load_config("config", [])
    assert cfg.capture.auto_exposure == 0.25, "0.25 is manual mode under DirectShow"
    assert cfg.capture.exposure == -5.0, "-5 = 31 ms, the longest exposure fitting a 33 ms budget"


def test_demo_webcam_profile_prefers_image_quality():
    cfg = load_config("config", ["capture=webcam_demo"])
    assert cfg.capture.auto_exposure == 0.75
    assert cfg.capture.exposure is None


def test_build_source_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown capture.source"):
        build_source({"capture": {"source": "telepathy"}})


def test_build_source_rejects_file_without_path():
    with pytest.raises(ValueError, match="capture.path"):
        build_source({"capture": {"source": "file"}})


# --------------------------------------------------------------------------------------------
# 6. Sources
# --------------------------------------------------------------------------------------------
def test_synthetic_source_frames_are_ordered_and_paced():
    src = SyntheticSource(width=160, height=120, target_fps=50.0)
    src.open()
    frames = [src.read() for _ in range(10)]
    src.close()

    assert [f.frame_id for f in frames] == list(range(10))
    ts = [f.t_capture for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:])), "capture timestamps must be monotonic"
    # ~9 intervals at 20 ms; generous upper bound for Windows timer granularity.
    assert 0.15 <= (ts[-1] - ts[0]) <= 0.45
    assert frames[0].image.shape == (120, 160, 3)


def test_synthetic_source_content_changes_between_frames():
    """Phase 2's frame differencing needs a source that actually moves."""
    src = SyntheticSource(width=160, height=120, target_fps=200.0)
    src.open()
    a, b = src.read(), src.read()
    src.close()
    assert not (a.image == b.image).all()


@pytest.mark.webcam
def test_webcam_source_opens_and_reads(has_webcam):
    if not has_webcam:
        pytest.skip("no webcam present")
    cfg = load_config("config", [])
    src = build_source(cfg)
    src.open()
    try:
        frame = src.read()
        assert frame is not None
        assert frame.image.shape[2] == 3
        d = src.describe()
        assert d["backend"] == "CAP_DSHOW"
        assert d["exposure_readback"] == -5.0, "exposure must actually be pinned, not just requested"
        # A dark run must be flagged, not silently accepted — but it is not a test failure,
        # because the room lighting is not the code's responsibility.
        assert d["too_dark"] in (True, False)
        assert d["warmup_frame_brightness"] is not None
    finally:
        src.close()


# --------------------------------------------------------------------------------------------
# 7. THE EXIT CRITERION — a 5-second bounded run emits a valid metrics JSON
# --------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_five_second_bounded_run_emits_valid_metrics(tmp_path):
    """Phase 0 exit criterion, run through the real CLI as a subprocess.

    Uses the synthetic source so this passes on the CPU-only CI runner. The equivalent webcam run
    is exercised by `test_webcam_smoke_run` below and reported in RESULTS.md.
    """
    out = tmp_path / "phase0_smoke.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "peripheral.cli.smoke",
            "--duration", "5", "--headless", "--metrics-out", str(out),
            "-o", "capture=synthetic",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert proc.returncode == EXIT_OK, f"stderr:\n{proc.stderr}"
    assert out.exists(), "bounded run must write its metrics file"

    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_metrics(doc)

    assert doc["run"]["status"] == schema.STATUS_COMPLETED
    assert 4.5 <= doc["run"]["duration_actual_s"] <= 6.5

    m = doc["metrics"]
    assert m["frames_captured"] > 100          # ~150 at 30 FPS over 5 s
    assert m["achieved_fps"] > 25
    assert m["capture_interval_ms"]["p50"] > 0
    assert m["stages_ms"]["capture_read"]["n"] == m["frames_captured"]
    # No VLM ran, so every VLM-derived family must be null rather than zero.
    assert m["photon_to_first_token_ms"] is None
    assert m["vlm_calls"] == 0


@pytest.mark.slow
@pytest.mark.webcam
def test_webcam_smoke_run(tmp_path, has_webcam):
    """The same bounded run against the real camera. Skipped where there is no device."""
    if not has_webcam:
        pytest.skip("no webcam present")
    out = tmp_path / "webcam_smoke.json"
    assert smoke.main(
        ["--duration", "3", "--headless", "--metrics-out", str(out)]
    ) == EXIT_OK

    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_metrics(doc)
    # The camera is capped at 30 FPS (measured); a synchronous read loop should track it closely.
    assert doc["metrics"]["achieved_fps"] > 25
