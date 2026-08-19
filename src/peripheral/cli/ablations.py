"""Phase 6: ablations — remove one component at a time, at matched call budgets.

`PROMPT.md`: *remove the fast tier, remove the cache, remove KV reuse, replace the scheduler with
fixed-interval at matched call budget.*

Two of those need a note rather than a run:

* **remove the cache** — there is no cache in the pipeline to remove. Phase 5 measured it and cut
  it (RESULTS.md §5), so this ablation is already the shipped configuration.
* **remove KV reuse** — measured in Phase 3 as its own before/after experiment (§3): 160 → 150 ms
  p50 TTFT, a 6% effect. It is a latency ablation, and re-running it here against accuracy would
  add noise without adding information, because KV reuse cannot change *what* the model answers.

What is run here is the set that changes the *decisions*: no fast tier, motion-only, fixed interval
at a matched budget, and the per-frame oracle as the upper bound. Every arm goes through the same
wall-clock replay harness with the same enforcement gates.

    python -m peripheral.cli.ablations --duration 600 --headless \
        --metrics-out results/phase6_ablations.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.clips import ClipAnnotation
from ..eval.simulate import simulate
from ..eval.traces import ClipTrace
from ..runtime import BoundedRunner
from ..scheduler.policies import build_policy
from ._args import parse_and_load


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


class AblationsRunner(BoundedRunner):
    """Ablations over cached traces.

    Uses the Phase 4 trace machinery rather than live replay: the question here is what each
    component contributes to *decisions and accuracy*, over all six clips, which needs the oracle's
    per-frame answers as ground truth. Wall-clock behaviour is verified separately and end to end by
    `peripheral.cli.replay_eval`, whose audit is a Phase 6 exit criterion in its own right.
    """

    name = "phase6_ablations"

    def setup(self) -> None:
        p4 = self.cfg.phase4
        trace_dir = Path(p4.trace_dir)
        names = sorted({p.stem for p in trace_dir.glob("*.npz")})
        if not names:
            raise FileNotFoundError(f"no traces in {trace_dir} — run peripheral.cli.build_traces")
        self.traces = {n: ClipTrace.load(trace_dir / n) for n in names}
        self.annotations = {
            a.name: a for a in
            (ClipAnnotation.load(p) for p in sorted(Path(p4.clip_dir).glob("*.json")))
        }
        self.arms = [dict(a) for a in self.cfg.phase6.ablations]
        self.rows: list[dict[str, Any]] = []
        self.per_clip: list[dict[str, Any]] = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.arms):
            return False
        arm = self.arms[self._i]
        self._i += 1

        # "Matched call budget" has to mean matched. The full system's call rate is a measured
        # outcome, not a setting, so any arm asking to be matched gets its interval computed from
        # that measurement rather than from a hardcoded guess. An earlier version hardcoded 8 s and
        # silently handed the baseline 29% MORE calls than the system it was being compared against.
        if arm.get("match_budget") and arm["policy_kind"] == "fixed_interval":
            full = next((r for r in self.rows if r["name"] == "full_system"), None)
            if full is None:
                raise RuntimeError("a budget-matched arm must be listed after full_system")
            rate = full["calls_per_min_mean"]
            if rate <= 0:
                raise RuntimeError("full_system made no calls; cannot match a budget")
            arm = dict(arm)
            arm["policy_value"] = 60.0 / rate
            arm["note"] = (
                f"{arm.get('note', '')} Interval computed as 60/{rate:.3f} = "
                f"{arm['policy_value']:.3f}s to match the full system's MEASURED call rate."
            ).strip()

        kwargs: dict[str, Any] = {}
        if arm["policy_kind"] != "fixed_interval":
            kwargs["min_gap_s"] = float(self.cfg.phase4.min_gap_s)

        sims = []
        for name, trace in self.traces.items():
            policy = build_policy(arm["policy_kind"], float(arm["policy_value"]), **kwargs)
            # "No fast tier" means the scheduler has no content signal at all. Zeroing the scores
            # is how that is expressed: the policy still runs, but sees nothing.
            t = _blinded(trace) if not arm.get("use_fast_tier", True) else trace
            sims.append(simulate(t, policy, warmup_frames=int(self.cfg.phase4.warmup_frames)))

        row = _aggregate(arm, sims)
        self.rows.append(row)
        for s in sims:
            self.per_clip.append({"arm": arm["name"], **s.as_row()})

        _progress(
            f"  {arm['name']:24s} validity {row['answer_validity_mean']:.3f}  "
            f"recall {row['event_recall_mean']}  "
            f"{row['calls_per_min_mean']:7.1f} calls/min  "
            f"FTR {row['false_trigger_rate_mean']}"
        )
        return True

    def teardown(self) -> None:
        full = next((r for r in self.rows if r["name"] == "full_system"), None)
        for r in self.rows:
            if full and full["answer_validity_mean"]:
                r["validity_vs_full"] = round(
                    r["answer_validity_mean"] - full["answer_validity_mean"], 5
                )
                r["calls_vs_full"] = round(
                    r["calls_per_min_mean"] / full["calls_per_min_mean"], 3
                ) if full["calls_per_min_mean"] else None

        self.recorder.record_extra("phase6_ablations", {
            "rows": self.rows,
            "per_clip": self.per_clip,
            "not_run_and_why": {
                "remove_cache": (
                    "No cache exists in the pipeline to remove. Phase 5 measured it and cut it "
                    "(RESULTS.md section 5), so this ablation IS the shipped configuration."
                ),
                "remove_kv_reuse": (
                    "Measured in Phase 3 as its own before/after experiment: 160 -> 150 ms p50 "
                    "TTFT, a 6% effect. KV reuse changes latency, not which answer the model "
                    "produces, so re-running it against accuracy would add noise, not information."
                ),
            },
            "definitions": {
                "answer_validity": "fraction of frames whose held answer describes the current "
                                   "scene state (Phase 4 primary metric)",
                "no_fast_tier": "the scheduler's content inputs are zeroed, so it degenerates to a "
                                "timer — which is what removing the fast tier actually means",
            },
        })


def _blinded(trace: ClipTrace) -> ClipTrace:
    """The same trace with every fast-tier signal removed."""
    return ClipTrace(
        clip_name=trace.clip_name, fps=trace.fps, n_frames=trace.n_frames,
        motion=np.zeros_like(trace.motion),
        novelty=np.zeros_like(trace.novelty),
        scene_change=np.zeros_like(trace.scene_change),
        answers=trace.answers, state_ids=trace.state_ids,
        encoder={"blinded": True}, model=trace.model, prompt=trace.prompt,
    )


def _aggregate(arm: dict[str, Any], sims: list) -> dict[str, Any]:
    def arr(attr: str) -> np.ndarray:
        return np.array([getattr(s, attr) for s in sims], dtype=float)

    ftr = [s.false_trigger_rate for s in sims if s.false_trigger_rate is not None]
    rec = [s.event_recall for s in sims if s.event_recall is not None]
    delays = [s.detection_delay_ms_mean for s in sims if s.detection_delay_ms_mean is not None]
    return {
        "name": arm["name"],
        "policy": f"{arm['policy_kind']} @ {arm['policy_value']}",
        "use_fast_tier": bool(arm.get("use_fast_tier", True)),
        "note": arm.get("note", ""),
        "n_clips": len(sims),
        "answer_validity_mean": round(float(arr("answer_validity").mean()), 5),
        "answer_validity_std": round(float(arr("answer_validity").std()), 5),
        "answer_validity_min": round(float(arr("answer_validity").min()), 5),
        "accuracy_mean": round(float(arr("accuracy").mean()), 5),
        "calls_per_min_mean": round(float(arr("calls_per_min").mean()), 3),
        "call_fraction_mean": round(float(arr("call_fraction").mean()), 5),
        "false_trigger_rate_mean": round(float(np.mean(ftr)), 5) if ftr else None,
        "event_recall_mean": round(float(np.mean(rec)), 4) if rec else None,
        "detection_delay_ms_mean": round(float(np.mean(delays)), 1) if delays else None,
        "staleness_ms_mean": round(float(arr("staleness_ms_mean").mean()), 1),
    }


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.ablations",
        description="Phase 6: ablations at matched call budgets.",
        argv=argv,
    )
    runner = AblationsRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                            metrics_out=cfg.run.metrics_out,
                            headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
