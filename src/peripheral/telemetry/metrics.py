"""MetricsRecorder — the load-bearing piece of Phase 0.

Every metric named in `PROMPT.md` is recorded here from Phase 0 onward. Adding one later means
re-running every prior benchmark, which is why this is built before the pipeline it measures.

Design (see STATUS.md decisions D4/D5):

* **Event-based.** Raw events go in; every aggregate is derived at `finalize()`. If a definition
  changes we recompute from events instead of re-running the benchmark.
* **`null` is not `0`.** A metric that was not measured serializes as `null`. Zero means measured
  and it was zero. Silent zero-filling is how a results table lies.
* **Thread-safe.** Capture, fast tier, VLM and render all record concurrently from Phase 1.
"""

from __future__ import annotations

import json
import platform
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import schema
from .clock import now, wall_anchor
from .power import PowerSampler


@dataclass
class _VlmCall:
    call_id: str
    frame_id: int | None
    t_start: float
    t_first_token: float | None = None
    t_complete: float | None = None
    prompt_tokens: int | None = None
    gen_tokens: int | None = None
    model: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class MetricsRecorder:
    """Collects timings and events for one bounded run, then serializes them to JSON.

    Usage:
        rec = MetricsRecorder(run_name="phase0_smoke", config=cfg, seed=1337)
        rec.start()
        ...
        rec.finish(status=schema.STATUS_COMPLETED)
        rec.write(Path("results/phase0_smoke.json"))
    """

    def __init__(
        self,
        run_name: str,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        duration_requested_s: float | None = None,
        power_sampler: PowerSampler | None = None,
        keep_events: bool = True,
        max_events: int = 500_000,
    ) -> None:
        self.run_name = run_name
        self.config = config or {}
        self.seed = seed
        self.duration_requested_s = duration_requested_s
        self.keep_events = keep_events
        self.max_events = max_events
        self.power = power_sampler

        self._lock = threading.Lock()
        self._t_start: float | None = None
        self._t_end: float | None = None
        self._status = schema.STATUS_FAILED  # pessimistic until finish() says otherwise
        self._failure: str | None = None
        self._wall = wall_anchor()

        # raw series
        self._frame_t: dict[int, float] = {}
        self._capture_order: list[float] = []
        self._drops: list[dict[str, Any]] = []
        self._stages: dict[str, list[float]] = {}
        self._queues: dict[str, list[tuple[float, int]]] = {}
        self._triggers: list[dict[str, Any]] = []
        self._calls: dict[str, _VlmCall] = {}
        self._answers: list[dict[str, Any]] = []
        self._accuracy: list[dict[str, Any]] = []
        self._events_truncated = False
        self._notes: list[str] = []

    # -- lifecycle -----------------------------------------------------------------------
    def start(self) -> float:
        with self._lock:
            self._t_start = now()
            return self._t_start

    def finish(self, status: str, failure: str | None = None, t_end: float | None = None) -> None:
        """Close the measured window.

        `t_end` must be the instant the *work* stopped, not the instant cleanup finished. Teardown
        is not free — releasing a DSHOW camera takes ~250 ms — and folding it into the window
        deflates every rate metric derived from it. Measured: 152 frames read at a true 30.2 FPS
        reported as 28.7 FPS purely because `cap.release()` was inside the window.
        """
        if status not in schema.ALL_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._lock:
            self._t_end = now() if t_end is None else t_end
            self._status = status
            self._failure = failure

    def note(self, text: str) -> None:
        """Free-text caveat carried into the metrics file. Use for anything that qualifies a number."""
        with self._lock:
            self._notes.append(text)

    # -- capture -------------------------------------------------------------------------
    def record_frame_captured(self, frame_id: int, t: float | None = None) -> float:
        """`t` is the instant OpenCV returned the frame — the t0 of every latency in this run."""
        t = now() if t is None else t
        with self._lock:
            self._frame_t[frame_id] = t
            self._capture_order.append(t)
        return t

    def record_frame_dropped(self, frame_id: int, stage: str, reason: str) -> None:
        """A dropped frame is a policy decision. Recording it is what makes it not an accident."""
        with self._lock:
            self._drops.append(
                {"t": now(), "frame_id": frame_id, "stage": stage, "reason": reason}
            )

    def record_stage(self, stage: str, ms: float, frame_id: int | None = None) -> None:
        """Per-stage millisecond budget. Phase 2 reports these; Phase 0 just has to support them."""
        with self._lock:
            self._stages.setdefault(stage, []).append(float(ms))

    def record_queue_depth(self, queue: str, depth: int, t: float | None = None) -> None:
        t = now() if t is None else t
        with self._lock:
            series = self._queues.setdefault(queue, [])
            if len(series) < self.max_events:
                series.append((t, int(depth)))
            else:
                self._events_truncated = True

    # -- scheduler -----------------------------------------------------------------------
    def record_trigger(
        self,
        frame_id: int,
        policy: str,
        fired: bool,
        score: float | None = None,
        reason: str | None = None,
        semantic_change: bool | None = None,
    ) -> None:
        """One scheduler decision.

        `semantic_change` is external ground truth: True if this frame really did carry a semantic
        event, False if it did not, None if unlabelled. False-trigger rate is derived only from
        labelled frames — an unlabelled run reports null rather than a flattering zero.
        """
        with self._lock:
            self._triggers.append(
                {
                    "t": now(),
                    "frame_id": frame_id,
                    "policy": policy,
                    "fired": bool(fired),
                    "score": score,
                    "reason": reason,
                    "semantic_change": semantic_change,
                }
            )

    # -- VLM -----------------------------------------------------------------------------
    def vlm_call_start(
        self,
        call_id: str,
        frame_id: int | None = None,
        model: str | None = None,
        prompt_tokens: int | None = None,
        t: float | None = None,
    ) -> None:
        with self._lock:
            self._calls[call_id] = _VlmCall(
                call_id=call_id,
                frame_id=frame_id,
                t_start=now() if t is None else t,
                model=model,
                prompt_tokens=prompt_tokens,
            )

    def vlm_call_first_token(self, call_id: str, t: float | None = None) -> None:
        with self._lock:
            call = self._calls.get(call_id)
            if call is not None and call.t_first_token is None:
                call.t_first_token = now() if t is None else t

    def vlm_call_complete(
        self, call_id: str, gen_tokens: int | None = None, t: float | None = None
    ) -> None:
        with self._lock:
            call = self._calls.get(call_id)
            if call is not None:
                call.t_complete = now() if t is None else t
                call.gen_tokens = gen_tokens

    def record_answer(
        self,
        answer_id: str,
        evidence_frame_id: int | None,
        source: str = "vlm",
        call_id: str | None = None,
        t: float | None = None,
        text: str | None = None,
    ) -> None:
        """`evidence_frame_id` is the frame whose content actually backs this answer.

        For a cache hit that is the frame the cached observation came from, *not* the current
        frame — otherwise staleness reads as zero for exactly the answers we most need to audit.
        """
        with self._lock:
            self._answers.append(
                {
                    "t": now() if t is None else t,
                    "answer_id": answer_id,
                    "evidence_frame_id": evidence_frame_id,
                    "source": source,
                    "call_id": call_id,
                    "text": text,
                }
            )

    def record_accuracy(
        self, query_id: str, correct: bool, oracle: str | None = None, got: str | None = None
    ) -> None:
        with self._lock:
            self._accuracy.append(
                {"query_id": query_id, "correct": bool(correct), "oracle": oracle, "got": got}
            )

    # -- derivation ----------------------------------------------------------------------
    @property
    def duration_s(self) -> float:
        if self._t_start is None:
            return 0.0
        end = self._t_end if self._t_end is not None else now()
        return max(0.0, end - self._t_start)

    def _photon_latencies(self) -> tuple[list[float], list[float]]:
        """(photon→first token, photon→complete answer) in ms, over calls with a known source frame."""
        first: list[float] = []
        full: list[float] = []
        for call in self._calls.values():
            t0 = self._frame_t.get(call.frame_id) if call.frame_id is not None else None
            if t0 is None:
                continue
            if call.t_first_token is not None:
                first.append((call.t_first_token - t0) * 1000.0)
            if call.t_complete is not None:
                full.append((call.t_complete - t0) * 1000.0)
        return first, full

    def _staleness_ms(self) -> list[float]:
        out: list[float] = []
        for ans in self._answers:
            t0 = (
                self._frame_t.get(ans["evidence_frame_id"])
                if ans["evidence_frame_id"] is not None
                else None
            )
            if t0 is not None:
                out.append((ans["t"] - t0) * 1000.0)
        return out

    def _false_trigger_rate(self) -> dict[str, Any] | None:
        fired = [t for t in self._triggers if t["fired"]]
        labelled = [t for t in fired if t["semantic_change"] is not None]
        if not labelled:
            return None
        false_fires = [t for t in labelled if t["semantic_change"] is False]
        return {
            "rate": round(len(false_fires) / len(labelled), 4),
            "n_false_triggers": len(false_fires),
            "n_labelled_fires": len(labelled),
            "n_unlabelled_fires": len(fired) - len(labelled),
        }

    def _metrics_block(self) -> dict[str, Any]:
        dur = self.duration_s
        n_frames = len(self._capture_order)

        intervals = [
            (self._capture_order[i] - self._capture_order[i - 1]) * 1000.0
            for i in range(1, n_frames)
        ]
        first_tok, full_ans = self._photon_latencies()
        n_calls = len(self._calls)
        acc = [a["correct"] for a in self._accuracy]

        return {
            "frames_captured": n_frames,
            "frames_dropped": len(self._drops),
            "drop_rate": round(len(self._drops) / (n_frames + len(self._drops)), 6)
            if (n_frames + len(self._drops)) > 0
            else None,
            "achieved_fps": round(n_frames / dur, 3) if dur > 0 else None,
            "capture_interval_ms": _summary(intervals),
            "stages_ms": {k: _summary(v) for k, v in sorted(self._stages.items())} or None,
            "queue_depth": {
                name: {
                    "max": max(d for _, d in series),
                    "mean": round(sum(d for _, d in series) / len(series), 3),
                    "n_samples": len(series),
                    "series": [[round(t, 6), d] for t, d in series] if self.keep_events else None,
                }
                for name, series in sorted(self._queues.items())
                if series
            }
            or None,
            "photon_to_first_token_ms": _summary(first_tok),
            "photon_to_answer_ms": _summary(full_ans),
            "vlm_calls": n_calls,
            "vlm_calls_per_min": round(n_calls / (dur / 60.0), 3) if dur > 0 else None,
            "answer_staleness_ms": _summary(self._staleness_ms()),
            "accuracy_vs_oracle": round(sum(acc) / len(acc), 4) if acc else None,
            "false_trigger_rate": self._false_trigger_rate(),
            "trigger_count": {
                "decisions": len(self._triggers),
                "fired": sum(1 for t in self._triggers if t["fired"]),
            }
            if self._triggers
            else None,
            "power": self.power.summary(n_queries=n_calls or None)
            if self.power is not None
            else None,
            "vram": self.power.vram_summary() if self.power is not None else None,
        }

    def finalize(self) -> dict[str, Any]:
        """Build the full metrics document. Safe to call more than once."""
        with self._lock:
            doc = {
                "schema_version": schema.SCHEMA_VERSION,
                "run": {
                    "name": self.run_name,
                    "status": self._status,
                    "failure": self._failure,
                    "duration_requested_s": self.duration_requested_s,
                    "duration_actual_s": round(self.duration_s, 4),
                    "started_utc": self._wall["utc"],
                    "wall_anchor": self._wall,
                    "notes": list(self._notes),
                },
                "environment": _environment(self.seed),
                "config": self.config,
                "definitions": dict(schema.DEFINITIONS),
                "metrics": self._metrics_block(),
                "events": self._events_block(),
            }
        schema.validate_metrics(doc)
        return doc

    def _events_block(self) -> dict[str, Any]:
        counts = {
            "frames": len(self._capture_order),
            "drops": len(self._drops),
            "triggers": len(self._triggers),
            "vlm_calls": len(self._calls),
            "answers": len(self._answers),
            "accuracy_records": len(self._accuracy),
            "truncated": self._events_truncated,
        }
        if not self.keep_events:
            return {"counts": counts, "kept": False}
        return {
            "counts": counts,
            "kept": True,
            "drops": self._drops,
            "triggers": self._triggers,
            "answers": self._answers,
            "vlm_calls": [
                {
                    "call_id": c.call_id,
                    "frame_id": c.frame_id,
                    "model": c.model,
                    "t_start": round(c.t_start, 6),
                    "t_first_token": round(c.t_first_token, 6)
                    if c.t_first_token is not None
                    else None,
                    "t_complete": round(c.t_complete, 6) if c.t_complete is not None else None,
                    "prompt_tokens": c.prompt_tokens,
                    "gen_tokens": c.gen_tokens,
                }
                for c in self._calls.values()
            ],
        }

    def write(self, path: str | Path) -> Path:
        """Serialize to JSON. Parent directories are created; the file is validated before writing."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.finalize()
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path


# --- helpers ---------------------------------------------------------------------------------
def _summary(values: list[float]) -> dict[str, Any] | None:
    """Percentile block, or None for an empty population (never a zero-filled dict)."""
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "min": round(float(a.min()), 3),
        "max": round(float(a.max()), 3),
        "std": round(float(a.std()), 3),
    }


def _environment(seed: int | None) -> dict[str, Any]:
    """Everything needed to know whether two result files are comparable."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "seed": seed,
        "peripheral_version": _package_version(),
        "git_commit": _git_commit(),
        "gpu_name": None,
        "gpu_vram_gb": None,
        "torch": None,
        "cuda": None,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            env["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # noqa: BLE001 - CI is CPU-only; absence of CUDA is not a failure
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    return env


def _package_version() -> str:
    from peripheral import __version__

    return __version__


def _git_commit() -> str | None:
    """Result files must be traceable to code. Returns None outside a git checkout."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[3],
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None
