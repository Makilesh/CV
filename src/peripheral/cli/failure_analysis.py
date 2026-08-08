"""Phase 6: honest failure analysis.

`PROMPT.md`: *an honest failure analysis: which events are missed, which false-trigger, and where the
oracle gap actually comes from.*

This attributes the gap rather than describing it. For the chosen policy on every clip it reports:

* **which specific events were missed**, with their kind and timestamp — not a count, the actual list
* **which calls were false triggers**, and what the fast tier saw at that moment
* **a decomposition of the oracle gap**: every frame whose held answer is invalid is attributed to
  exactly one cause — a missed event, detection lag, or the warmup period.

    python -m peripheral.cli.failure_analysis --duration 300 --headless \
        --metrics-out results/phase6_failures.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.clips import ClipAnnotation
from ..eval.traces import ClipTrace
from ..runtime import BoundedRunner
from ..scheduler.policies import FrameContext, build_policy
from ._args import parse_and_load


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


class FailureAnalysisRunner(BoundedRunner):
    name = "phase6_failure_analysis"

    def setup(self) -> None:
        p4 = self.cfg.phase4
        trace_dir = Path(p4.trace_dir)
        self.traces = {p.stem: ClipTrace.load(trace_dir / p.stem)
                       for p in sorted(trace_dir.glob("*.npz"))}
        self.annotations = {
            a.name: a for a in
            (ClipAnnotation.load(p) for p in sorted(Path(p4.clip_dir).glob("*.json")))
        }
        self.names = sorted(self.traces)
        self.reports: list[dict[str, Any]] = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.names):
            return False
        name = self.names[self._i]
        self._i += 1
        trace = self.traces[name]
        ann = self.annotations.get(name)

        policy = build_policy(str(self.cfg.phase6.policy_kind),
                             float(self.cfg.phase6.policy_value),
                             min_gap_s=float(self.cfg.phase4.min_gap_s))
        policy.reset()
        warmup = int(self.cfg.phase4.warmup_frames)
        fps = trace.fps
        st = trace.state_ids

        transitions = [i for i in range(1, trace.n_frames) if st[i] != st[i - 1]]
        calls: list[dict[str, Any]] = []
        t_last: float | None = None
        state_at_last_call: int | None = None
        evidence_frame: int | None = None
        covered: dict[int, int] = {}

        # Per-frame attribution of every invalid frame to exactly one cause.
        causes = {"warmup": 0, "detection_lag": 0, "missed_event": 0, "valid": 0}

        for i in range(trace.n_frames):
            t = i / fps
            ctx = FrameContext(i, t, float(trace.motion[i]), float(trace.novelty[i]),
                               float(trace.scene_change[i]), None, t_last, len(calls))
            decision = policy.decide(ctx)
            if decision.fire:
                state_now = int(st[i])
                justified = state_at_last_call is None or state_now != state_at_last_call
                calls.append({
                    "frame": i, "t": round(t, 3), "state": state_now,
                    "justified": bool(justified),
                    "novelty": round(float(trace.novelty[i]), 5),
                    "motion": round(float(trace.motion[i]), 5),
                    "scene_change": round(float(trace.scene_change[i]), 5),
                })
                evidence_frame = i
                state_at_last_call = state_now
                t_last = t
                policy.observe_call(ctx)
                for tr in transitions:
                    if tr <= i and tr not in covered and int(st[tr]) == state_now:
                        covered[tr] = i

            if i < warmup:
                causes["warmup"] += 1
                continue
            if evidence_frame is not None and int(st[evidence_frame]) == int(st[i]):
                causes["valid"] += 1
                continue
            # Invalid. Was the event that put us in this state ever picked up at all?
            last_tr = max((tr for tr in transitions if tr <= i), default=None)
            if last_tr is None or last_tr not in covered:
                causes["missed_event"] += 1
            else:
                causes["detection_lag"] += 1

        missed = [tr for tr in transitions if tr not in covered]
        report = {
            "clip": name,
            "purpose": ann.purpose if ann else "",
            "n_frames": trace.n_frames,
            "n_calls": len(calls),
            "calls_per_min": round(len(calls) / (trace.n_frames / fps / 60.0), 2),
            "n_events": len(transitions),
            "events_missed": [
                {
                    "frame": tr, "t": round(tr / fps, 3),
                    "to_state": int(st[tr]),
                    "kind": _event_kind(ann, tr / fps),
                    "novelty_at_event": round(float(trace.novelty[tr]), 5),
                    "peak_novelty_after": round(
                        float(trace.novelty[tr:min(tr + 60, trace.n_frames)].max()), 5
                    ),
                }
                for tr in missed
            ],
            "false_triggers": [c for c in calls if not c["justified"]],
            "detection_delays_ms": [
                round((covered[tr] - tr) / fps * 1000.0, 1) for tr in sorted(covered)
            ],
            "oracle_gap_attribution": {
                **causes,
                "scored_frames": trace.n_frames - warmup,
                "validity": round(causes["valid"] / max(1, trace.n_frames - warmup), 5),
            },
        }
        self.reports.append(report)
        _progress(
            f"  {name:16s} calls {len(calls):3d}  missed {len(missed)}/{len(transitions)} events  "
            f"false triggers {len(report['false_triggers']):3d}  "
            f"validity {report['oracle_gap_attribution']['validity']:.3f}"
        )
        return True

    def teardown(self) -> None:
        totals = {"warmup": 0, "detection_lag": 0, "missed_event": 0, "valid": 0}
        for r in self.reports:
            for k in totals:
                totals[k] += r["oracle_gap_attribution"][k]
        scored = sum(v for v in totals.values())
        invalid = scored - totals["valid"] - totals["warmup"]

        self.recorder.record_extra("phase6_failures", {
            "policy": f"{self.cfg.phase6.policy_kind} @ {self.cfg.phase6.policy_value}",
            "per_clip": self.reports,
            "totals": totals,
            "oracle_gap_decomposition": {
                "invalid_frames": invalid,
                "from_missed_events": totals["missed_event"],
                "from_detection_lag": totals["detection_lag"],
                "pct_from_missed_events": round(100 * totals["missed_event"] / invalid, 1)
                if invalid else None,
                "pct_from_detection_lag": round(100 * totals["detection_lag"] / invalid, 1)
                if invalid else None,
                "note": (
                    "Every frame whose held answer describes the wrong scene state is attributed to "
                    "exactly one cause. 'missed_event' means the transition into that state was "
                    "never queried at all; 'detection_lag' means it was queried, but later. The two "
                    "call for different fixes: a lower threshold for the first, a shorter minimum "
                    "gap for the second."
                ),
            },
        })


def _event_kind(ann: ClipAnnotation | None, t: float) -> str:
    if ann is None:
        return "unknown"
    best, best_dt = "unknown", 1e9
    for e in ann.events:
        dt = abs(e.t_start - t)
        if dt < best_dt:
            best, best_dt = f"{e.kind}: {e.description}", dt
    return best


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.failure_analysis",
        description="Phase 6: which events are missed, which false-trigger, where the gap comes from.",
        argv=argv,
    )
    runner = FailureAnalysisRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                  metrics_out=cfg.run.metrics_out,
                                  headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
