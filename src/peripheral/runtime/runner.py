"""The bounded-headless-runner contract.

`PROMPT.md` invariant 4: *every runnable supports `--duration N --headless --metrics-out
path.json`* — bounded run, writes metrics, exits 0. This is how the project evaluates its own work.
Nobody can watch a GUI or wait on an infinite loop.

Subclass `BoundedRunner`, implement `setup`/`step`/`teardown`, and the contract is satisfied:
deadline enforcement, seeding, power sampling, signal handling and metrics serialisation all come
from here.

**Metrics are written even when the run raises.** A crashed run that produces a partial file marked
`status: failed` is diagnosable; one that produces nothing is not. The status field is what stops a
partial file from being mistaken for a result.
"""

from __future__ import annotations

import random
import signal
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..telemetry import schema
from ..telemetry.clock import now
from ..telemetry.metrics import MetricsRecorder
from ..telemetry.power import PowerSampler

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INTERRUPTED = 130


def seed_everything(seed: int) -> None:
    """Seed every RNG we might touch. Invariant 6: runs are seeded."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class BoundedRunner(ABC):
    """A run that always terminates and always leaves numbers behind."""

    name: str = "runner"

    def __init__(
        self,
        cfg: Any,
        duration_s: float,
        metrics_out: str | Path | None = None,
        headless: bool = True,
        seed: int = 1337,
    ) -> None:
        self.cfg = cfg
        self.duration_s = float(duration_s)
        self.headless = bool(headless)
        self.metrics_out = Path(metrics_out) if metrics_out else None
        self.seed = int(seed)

        self.power = PowerSampler(
            sample_hz=float(_cfg_get(cfg, "telemetry.power_sample_hz", 10.0)),
        )
        self.recorder = MetricsRecorder(
            run_name=self.name,
            config=_cfg_to_dict(cfg),
            seed=self.seed,
            duration_requested_s=self.duration_s,
            power_sampler=self.power,
            keep_events=bool(_cfg_get(cfg, "telemetry.keep_events", True)),
        )
        self._stop = False
        self._deadline: float | None = None

    # -- subclass hooks ------------------------------------------------------------------
    def setup(self) -> None:
        """Open devices, load models. Time spent here is outside the measured window."""

    @abstractmethod
    def step(self) -> bool:
        """Do one unit of work. Return False to end the run early (e.g. source exhausted)."""

    def teardown(self) -> None:
        """Release devices. Always called, even after a failure."""

    # -- control -------------------------------------------------------------------------
    def request_stop(self) -> None:
        self._stop = True

    @property
    def time_left(self) -> float:
        if self._deadline is None:
            return self.duration_s
        return max(0.0, self._deadline - now())

    # -- the contract --------------------------------------------------------------------
    def run(self) -> int:
        seed_everything(self.seed)

        previous_sigint = None
        try:
            previous_sigint = signal.signal(signal.SIGINT, self._on_sigint)
        except ValueError:
            # Not on the main thread (pytest sometimes isn't). Ctrl+C handling is a nicety;
            # the deadline is what actually bounds the run.
            pass

        status = schema.STATUS_COMPLETED
        failure: str | None = None
        exit_code = EXIT_OK
        # The measured window closes when the work stops, not when cleanup finishes. Teardown
        # (releasing a DSHOW camera, unloading a model) would otherwise be counted as time in
        # which we captured no frames, deflating every rate metric.
        loop_end: float | None = None

        try:
            self.setup()
            # Baseline *after* setup so model-loading transients don't inflate idle draw,
            # and before the workload so it is genuinely a baseline.
            self.power.measure_baseline(seconds=min(0.5, self.duration_s / 10.0))
            self.power.start()

            self.recorder.start()
            self._deadline = now() + self.duration_s
            while now() < self._deadline and not self._stop:
                if not self.step():
                    self.recorder.note("run ended early: step() returned False")
                    break
            loop_end = now()
            if self._stop:
                status = schema.STATUS_INTERRUPTED
                exit_code = EXIT_INTERRUPTED
        except KeyboardInterrupt:
            loop_end = now()
            status = schema.STATUS_INTERRUPTED
            exit_code = EXIT_INTERRUPTED
        except Exception as exc:  # noqa: BLE001 - we must still emit metrics
            loop_end = now()
            status = schema.STATUS_FAILED
            failure = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            exit_code = EXIT_FAILED
        finally:
            try:
                self.teardown()
            except Exception as exc:  # noqa: BLE001
                self.recorder.note(f"teardown raised: {type(exc).__name__}: {exc}")
            self.power.stop()
            self.recorder.finish(status=status, failure=failure, t_end=loop_end)
            if self.metrics_out is not None:
                self.recorder.write(self.metrics_out)
            self.power.close()
            if previous_sigint is not None:
                signal.signal(signal.SIGINT, previous_sigint)

        return exit_code

    def _on_sigint(self, *_: Any) -> None:
        self.request_stop()


def _cfg_get(cfg: Any, dotted: str, default: Any) -> Any:
    """Read `a.b.c` from an OmegaConf node or a plain dict, tolerating absence."""
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        try:
            node = node[part]
        except (KeyError, TypeError, AttributeError):
            return default
    return default if node is None else node


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    """Config snapshot for the metrics file — two runs are comparable only if this matches."""
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass
    return dict(cfg) if isinstance(cfg, dict) else {"repr": repr(cfg)}
