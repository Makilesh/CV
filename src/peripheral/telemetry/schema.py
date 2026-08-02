"""The metrics JSON contract.

This module is the single source of truth for what a metrics file contains. Both the writer
(`MetricsRecorder.finalize`) and the tests validate against `validate_metrics` here, so the schema
cannot drift away from the thing that produces it.

Bump ``SCHEMA_VERSION`` whenever a field changes meaning. Old result files stay readable and
comparisons across versions stay honest.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1

# --- run status ------------------------------------------------------------------------------
STATUS_COMPLETED = "completed"      # ran to its full duration
STATUS_INTERRUPTED = "interrupted"  # Ctrl+C or an explicit stop; metrics are valid but truncated
STATUS_FAILED = "failed"            # raised; metrics are partial and must not be used for claims

ALL_STATUSES = (STATUS_COMPLETED, STATUS_INTERRUPTED, STATUS_FAILED)

# --- event kinds -----------------------------------------------------------------------------
# The recorder stores raw events and derives every aggregate at finalize time. If a definition
# changes later we can recompute from the events rather than re-run the benchmark.
EV_FRAME_CAPTURED = "frame_captured"
EV_FRAME_DROPPED = "frame_dropped"
EV_STAGE = "stage"
EV_QUEUE_DEPTH = "queue_depth"
EV_TRIGGER = "trigger"
EV_VLM_CALL = "vlm_call"
EV_ANSWER = "answer"
EV_ACCURACY = "accuracy"
EV_POWER = "power"

# Top-level blocks every metrics file must carry.
REQUIRED_TOP_LEVEL = (
    "schema_version",
    "run",
    "environment",
    "config",
    "definitions",
    "metrics",
    "events",
)

# Every metric family named in PROMPT.md. Present in every file, `null` when not measured.
REQUIRED_METRIC_KEYS = (
    "frames_captured",
    "frames_dropped",
    "drop_rate",
    "achieved_fps",
    "capture_interval_ms",
    "stages_ms",
    "queue_depth",
    "photon_to_first_token_ms",
    "photon_to_answer_ms",
    "vlm_calls",
    "vlm_calls_per_min",
    "answer_staleness_ms",
    "accuracy_vs_oracle",
    "false_trigger_rate",
    "trigger_count",
    "power",
    "vram",
)

# Written verbatim into every metrics file. A number without its definition is not a result.
DEFINITIONS: dict[str, str] = {
    "t0": (
        "Frame timestamps are taken with time.perf_counter() at the instant OpenCV returns the "
        "frame — NOT true photon arrival. Sensor exposure and driver/USB transport add an "
        "unmeasured constant offset, so photon-to-answer latency reported here is a LOWER BOUND."
    ),
    "photon_to_first_token_ms": (
        "t(first generated token of the answer) - t0(frame that triggered the VLM call). "
        "Includes preprocessing, vision encoding, prefill and any IPC — never model-only time."
    ),
    "photon_to_answer_ms": (
        "t(final token of the answer) - t0(frame that triggered the VLM call). Same inclusions."
    ),
    "answer_staleness_ms": (
        "t(answer emitted) - t0(frame whose evidence backs that answer). For a cache hit this is "
        "the age of the cached scene evidence, not the age of the cache entry."
    ),
    "false_trigger_rate": (
        "Fraction of VLM invocations fired on frames externally labelled as carrying no semantic "
        "change. Requires ground-truth labels; null when the run has none."
    ),
    "accuracy_vs_oracle": (
        "Agreement with the per-frame oracle (VLM invoked on every frame) on the same query "
        "timeline. Null unless an oracle run was supplied."
    ),
    "percentiles": (
        "numpy.percentile with linear interpolation over the full population of the run "
        "(no warmup discarded unless the run config says so)."
    ),
    "energy_j": (
        "Trapezoidal integration of NVML power samples over the measured window. "
        "'marginal' subtracts the idle baseline measured before the workload started."
    ),
    "null_vs_zero": (
        "null means NOT MEASURED. 0 means measured and the value was zero. They are never "
        "interchangeable."
    ),
}


class MetricsSchemaError(ValueError):
    """A metrics object does not satisfy the contract."""


def validate_metrics(obj: Any) -> None:
    """Raise `MetricsSchemaError` unless `obj` is a valid metrics document.

    Deliberately strict: a malformed metrics file is worse than a missing one, because it looks
    like a result.
    """
    if not isinstance(obj, dict):
        raise MetricsSchemaError(f"metrics must be a dict, got {type(obj).__name__}")

    for key in REQUIRED_TOP_LEVEL:
        if key not in obj:
            raise MetricsSchemaError(f"missing top-level key: {key!r}")

    if obj["schema_version"] != SCHEMA_VERSION:
        raise MetricsSchemaError(
            f"schema_version {obj['schema_version']!r} != expected {SCHEMA_VERSION}"
        )

    run = obj["run"]
    if not isinstance(run, dict):
        raise MetricsSchemaError("'run' must be a dict")
    for key in ("name", "status", "duration_requested_s", "duration_actual_s", "started_utc"):
        if key not in run:
            raise MetricsSchemaError(f"missing run key: {key!r}")
    if run["status"] not in ALL_STATUSES:
        raise MetricsSchemaError(f"unknown run status: {run['status']!r}")
    if not isinstance(run["duration_actual_s"], (int, float)):
        raise MetricsSchemaError("run.duration_actual_s must be numeric")

    metrics = obj["metrics"]
    if not isinstance(metrics, dict):
        raise MetricsSchemaError("'metrics' must be a dict")
    for key in REQUIRED_METRIC_KEYS:
        if key not in metrics:
            raise MetricsSchemaError(f"missing metric key: {key!r} (use null if not measured)")

    for name in ("photon_to_first_token_ms", "photon_to_answer_ms", "answer_staleness_ms",
                 "capture_interval_ms"):
        _validate_summary(name, metrics[name])

    if not isinstance(obj["definitions"], dict) or "t0" not in obj["definitions"]:
        raise MetricsSchemaError("'definitions' must be a dict carrying at least 't0'")

    env = obj["environment"]
    if not isinstance(env, dict):
        raise MetricsSchemaError("'environment' must be a dict")
    for key in ("python", "platform", "gpu_name", "seed"):
        if key not in env:
            raise MetricsSchemaError(f"missing environment key: {key!r}")


def _validate_summary(name: str, value: Any) -> None:
    """A latency summary is either null (not measured) or a full percentile block."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise MetricsSchemaError(f"{name} must be null or a dict, got {type(value).__name__}")
    for key in ("n", "mean", "p50", "p95", "p99", "min", "max"):
        if key not in value:
            raise MetricsSchemaError(f"{name} missing {key!r}")
    if value["n"] == 0:
        raise MetricsSchemaError(f"{name} has n=0; an empty population must serialize as null")
