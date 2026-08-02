"""One monotonic clock for the whole system.

Every timestamp in Peripheral comes from here. Mixing `time.time()` and `time.perf_counter()` in a
latency pipeline produces numbers that look plausible and are wrong — `time.time()` can step
backwards on Windows when NTP corrects the system clock, which silently manufactures negative
latencies or, worse, small positive ones.

`perf_counter()` is monotonic but its origin is arbitrary, so we also record a single wall-clock
anchor per run for correlating with external logs (clip timestamps, NVML traces).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now() -> float:
    """Monotonic seconds. The only timestamp source in the codebase."""
    return time.perf_counter()


@dataclass
class Clock:
    """Injectable clock, so tests can drive time deterministically instead of sleeping."""

    _source = staticmethod(time.perf_counter)

    def now(self) -> float:
        return self._source()


@dataclass
class FakeClock(Clock):
    """Test double. Time only advances when you say so."""

    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


@dataclass
class Stopwatch:
    """Times a block and reports milliseconds.

    Used for per-stage budgets (Phase 2 needs a millisecond breakdown per stage).

        with Stopwatch() as sw:
            ...
        recorder.record_stage("fast_tier", frame_id, sw.ms)
    """

    clock: Clock = field(default_factory=Clock)
    _t0: float = 0.0
    ms: float = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = self.clock.now()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (self.clock.now() - self._t0) * 1000.0


def wall_anchor() -> dict[str, float | str]:
    """A single correlation point between the monotonic clock and UTC wall time."""
    return {
        "monotonic": now(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "unix": time.time(),
    }
