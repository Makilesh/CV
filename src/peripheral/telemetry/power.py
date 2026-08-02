"""NVML power and VRAM sampling.

Energy per query is one of the metrics this project exists to report, and it cannot be
reconstructed after the fact — it has to be integrated from samples taken *during* the run. Hence a
background sampler started by every bounded run, whether or not the run uses the GPU.

Threads, not processes: a process worker would reload the CUDA context and cost us VRAM we do not
have (12 GB is the binding constraint).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .clock import now


@dataclass
class PowerSample:
    t: float
    power_w: float
    mem_used_b: int
    util_gpu_pct: int
    temp_c: int


class PowerSampler:
    """Background NVML poller.

    Degrades to a no-op when NVML is unavailable, recording *why* rather than pretending the run
    had no power data by choice. `available` is False and every derived metric serializes as null.
    """

    def __init__(self, sample_hz: float = 10.0, device_index: int = 0) -> None:
        self.sample_hz = float(sample_hz)
        self.device_index = int(device_index)
        self.samples: list[PowerSample] = []
        self.available = False
        self.error: str | None = None
        self.baseline_power_w: float | None = None
        self.power_limit_w: float | None = None
        self.driver: str | None = None

        self._nvml: Any = None
        self._handle: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self.driver = _as_str(pynvml.nvmlSystemGetDriverVersion())
            try:
                self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
            except Exception:  # noqa: BLE001 - optional field, absence is not an error
                self.power_limit_w = None
            self.available = True
        except Exception as exc:  # noqa: BLE001 - NVML absence must not kill a run
            self.error = f"{type(exc).__name__}: {exc}"

    # -- lifecycle ------------------------------------------------------------------------
    def measure_baseline(self, seconds: float = 0.5) -> float | None:
        """Idle draw before the workload starts, so we can report *marginal* energy.

        Total board power on a laptop includes whatever else is on the GPU; attributing all of it
        to our queries would overstate energy per query.
        """
        if not self.available:
            return None
        readings: list[float] = []
        deadline = now() + seconds
        while now() < deadline:
            s = self._read()
            if s is not None:
                readings.append(s.power_w)
            self._stop.wait(1.0 / self.sample_hz)
        self.baseline_power_w = sum(readings) / len(readings) if readings else None
        return self.baseline_power_w

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="power-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def close(self) -> None:
        self.stop()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    # -- sampling -------------------------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self.sample_hz
        while not self._stop.is_set():
            s = self._read()
            if s is not None:
                with self._lock:
                    self.samples.append(s)
            self._stop.wait(period)

    def _read(self) -> PowerSample | None:
        try:
            p = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            temp = self._nvml.nvmlDeviceGetTemperature(
                self._handle, self._nvml.NVML_TEMPERATURE_GPU
            )
            return PowerSample(now(), p, int(mem.used), int(util.gpu), int(temp))
        except Exception as exc:  # noqa: BLE001 - a dropped sample must not kill the run
            if self.error is None:
                self.error = f"sampling: {type(exc).__name__}: {exc}"
            return None

    # -- derived --------------------------------------------------------------------------
    def summary(self, n_queries: int | None = None) -> dict[str, Any]:
        """Power block for the metrics file. Every field is null when NVML was unavailable."""
        with self._lock:
            samples = list(self.samples)

        if not self.available or len(samples) < 2:
            return {
                "available": self.available,
                "error": self.error if not self.available else "fewer than 2 samples",
                "driver": self.driver,
                "power_limit_w": self.power_limit_w,
                "baseline_power_w": self.baseline_power_w,
                "n_samples": len(samples),
                "mean_w": None,
                "peak_w": None,
                "energy_j": None,
                "energy_marginal_j": None,
                "energy_per_query_j": None,
                "energy_marginal_per_query_j": None,
            }

        powers = [s.power_w for s in samples]
        energy_j = _trapezoid([s.t for s in samples], powers)

        marginal_j: float | None = None
        if self.baseline_power_w is not None:
            adjusted = [max(0.0, p - self.baseline_power_w) for p in powers]
            marginal_j = _trapezoid([s.t for s in samples], adjusted)

        per_query = energy_j / n_queries if n_queries else None
        marginal_per_query = (
            marginal_j / n_queries if (n_queries and marginal_j is not None) else None
        )

        return {
            "available": True,
            "error": self.error,
            "driver": self.driver,
            "power_limit_w": self.power_limit_w,
            "baseline_power_w": _r(self.baseline_power_w),
            "n_samples": len(samples),
            "sample_hz": self.sample_hz,
            "mean_w": _r(sum(powers) / len(powers)),
            "peak_w": _r(max(powers)),
            "energy_j": _r(energy_j),
            "energy_marginal_j": _r(marginal_j),
            "energy_per_query_j": _r(per_query),
            "energy_marginal_per_query_j": _r(marginal_per_query),
            "mean_util_pct": _r(sum(s.util_gpu_pct for s in samples) / len(samples), 1),
            "peak_temp_c": max(s.temp_c for s in samples),
        }

    def vram_summary(self) -> dict[str, Any]:
        """Peak VRAM as NVML saw it — board-wide, which is what the 12 GB limit actually is."""
        with self._lock:
            samples = list(self.samples)
        if not self.available or not samples:
            return {"available": self.available, "peak_used_gb": None, "mean_used_gb": None}
        used = [s.mem_used_b for s in samples]
        return {
            "available": True,
            "peak_used_gb": _r(max(used) / 1024**3),
            "mean_used_gb": _r(sum(used) / len(used) / 1024**3),
            "note": "NVML board-wide usage, includes other processes — not our allocation alone",
        }


def _trapezoid(ts: list[float], ys: list[float]) -> float:
    """Integrate an irregularly sampled series. Sampling jitter is real; do not assume dt."""
    total = 0.0
    for i in range(1, len(ts)):
        total += (ts[i] - ts[i - 1]) * (ys[i] + ys[i - 1]) / 2.0
    return total


def _r(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(float(x), nd)


def _as_str(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
