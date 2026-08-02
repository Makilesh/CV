"""Telemetry — load-bearing. Every metric in PROMPT.md is recorded from Phase 0 onward."""

from .clock import Clock, FakeClock, Stopwatch, now, wall_anchor
from .metrics import MetricsRecorder
from .power import PowerSampler
from .schema import SCHEMA_VERSION, MetricsSchemaError, validate_metrics

__all__ = [
    "SCHEMA_VERSION",
    "Clock",
    "FakeClock",
    "MetricsRecorder",
    "MetricsSchemaError",
    "PowerSampler",
    "Stopwatch",
    "now",
    "validate_metrics",
    "wall_anchor",
]
