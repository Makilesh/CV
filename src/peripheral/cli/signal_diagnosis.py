"""Phase 8a: is the scheduler's problem the threshold, or the signal?

Phase 6 left the project unable to support its own headline: at a matched budget a plain timer
equals or beats the embedding-novelty scheduler. There are two competing explanations, and they call
for completely different fixes, so guessing is expensive:

  **THRESHOLD PROBLEM** — novelty separates events from non-events *within* each clip, but the
  useful cut-point differs per scene. A global constant then works on the clip it was tuned on and
  fails elsewhere. Fix: make the threshold adapt to the scene.

  **SIGNAL PROBLEM** — event novelty sits inside the no-event distribution, so no cut-point works
  on any clip. Fix: a different fast-tier signal; adaptation would be building on sand.

Two measurements settle it:

1. **Separability (ROC AUC) per clip** — can this signal tell "just after a state change" from
   "nothing happening", *within one scene*, where scale is not a confound? Near 0.5 means the signal
   carries no event information and the problem is upstream of any threshold.

2. **Adaptation headroom** — the cost of the best *single global* threshold versus the best
   *per-clip* threshold, both required to hit the same validity bar. This is the oracle upper bound
   on what per-scene adaptation could ever buy. If it is ~1.0x there is nothing to win.

    python -m peripheral.cli.signal_diagnosis --duration 600 --headless \\
        --metrics-out results/phase8_signal_diagnosis.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.simulate import simulate
from ..eval.traces import ClipTrace
from ..runtime import BoundedRunner
from ..scheduler.policies import EmbeddingNoveltyPolicy, MotionThresholdPolicy
from ._args import parse_and_load

# A fine grid, much denser than the Phase 4 sweep: we are locating optima, not drawing a Pareto.
NOVELTY_GRID = [round(x, 4) for x in np.concatenate([
    np.linspace(0.005, 0.20, 40), np.linspace(0.21, 0.60, 20)
])]
MOTION_GRID = [round(x, 5) for x in np.concatenate([
    np.linspace(0.0005, 0.05, 40), np.linspace(0.055, 0.20, 15)
])]


def _progress(line: str) -> None:
    try:
        sys.stderr.write(line.encode("ascii", "replace").decode("ascii") + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based ROC AUC with tie handling. 0.5 = the signal says nothing."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class SignalDiagnosisRunner(BoundedRunner):
    name = "phase8_signal_diagnosis"

    def setup(self) -> None:
        trace_dir = Path(self.cfg.phase4.trace_dir)
        names = sorted({p.stem for p in trace_dir.glob("*.npz")})
        if not names:
            raise FileNotFoundError(f"no traces in {trace_dir}")
        raw = {n: ClipTrace.load(trace_dir / n) for n in names}
        signal = str(self.cfg.phase8.get("signal", "pooled"))
        self.signal = signal
        self.traces = {n: t.with_signal(signal) for n, t in raw.items()}

        p8 = self.cfg.phase8
        self.response_frames = int(p8.event_response_frames)
        self.quiet_margin = int(p8.quiet_margin_frames)
        self.validity_bar = float(p8.validity_bar)
        self.warmup = int(self.cfg.phase4.warmup_frames)
        self.min_gap_s = float(self.cfg.phase4.min_gap_s)

        self.separability: list[dict[str, Any]] = []
        self.curves: dict[str, list[dict[str, Any]]] = {}
        self._stage = 0

    # -- stage 1: can the signal separate events at all, within a clip? -------------------
    def _separability(self) -> None:
        for name, tr in self.traces.items():
            st = tr.state_ids
            transitions = [i for i in range(1, tr.n_frames) if st[i] != st[i - 1]]
            if not transitions:
                continue  # a single-state clip has no events to separate

            # "event" = the window just after a state change, where the signal should respond.
            event_mask = np.zeros(tr.n_frames, bool)
            for t in transitions:
                event_mask[t: min(t + self.response_frames, tr.n_frames)] = True

            # "quiet" = far enough from any transition that a response would have decayed.
            quiet_mask = np.ones(tr.n_frames, bool)
            for t in transitions:
                lo = max(0, t - self.quiet_margin)
                hi = min(tr.n_frames, t + self.response_frames + self.quiet_margin)
                quiet_mask[lo:hi] = False
            quiet_mask[: self.warmup] = False

            row: dict[str, Any] = {"clip": name, "n_events": len(transitions),
                                   "n_event_frames": int(event_mask.sum()),
                                   "n_quiet_frames": int(quiet_mask.sum())}
            for sig in ("novelty", "motion", "scene_change"):
                arr = getattr(tr, sig)
                row[f"auc_{sig}"] = round(_auc(arr[event_mask], arr[quiet_mask]), 4)
                row[f"{sig}_event_median"] = round(float(np.median(arr[event_mask])), 5)
                row[f"{sig}_quiet_p95"] = round(float(np.percentile(arr[quiet_mask], 95)), 5)
            # The decisive per-clip fact: does the event signal clear the quiet noise ceiling?
            row["event_clears_quiet_p95"] = bool(
                row["novelty_event_median"] > row["novelty_quiet_p95"]
            )
            self.separability.append(row)
            _progress(
                f"  {name:16s} AUC novelty {row['auc_novelty']:.3f}  motion {row['auc_motion']:.3f}"
                f"  scene {row['auc_scene_change']:.3f}   "
                f"event median {row['novelty_event_median']:.4f} vs quiet p95 "
                f"{row['novelty_quiet_p95']:.4f}"
            )

    # -- stage 2: what could a per-scene threshold buy? ----------------------------------
    def _curves(self) -> None:
        for kind, grid in (("embedding_novelty", NOVELTY_GRID), ("motion_threshold", MOTION_GRID)):
            rows: list[dict[str, Any]] = []
            for thr in grid:
                for name, tr in self.traces.items():
                    policy = (EmbeddingNoveltyPolicy(thr, min_gap_s=self.min_gap_s)
                              if kind == "embedding_novelty"
                              else MotionThresholdPolicy(thr, min_gap_s=self.min_gap_s))
                    r = simulate(tr, policy, warmup_frames=self.warmup)
                    rows.append({
                        "clip": name, "threshold": thr,
                        "validity": r.answer_validity,
                        "calls_per_min": r.calls_per_min,
                        "event_recall": r.event_recall,
                    })
            self.curves[kind] = rows
            _progress(f"  swept {kind}: {len(grid)} thresholds x {len(self.traces)} clips")

    def step(self) -> bool:
        if self._stage == 0:
            _progress("stage 1 - separability within each clip")
            self._separability()
            self._stage = 1
            return True
        if self._stage == 1:
            _progress("stage 2 - global vs per-clip optimal thresholds")
            self._curves()
            self._stage = 2
            return True
        return False

    # -- verdict -------------------------------------------------------------------------
    def _headroom(self, kind: str) -> dict[str, Any]:
        rows = self.curves[kind]
        clips = sorted({r["clip"] for r in rows})
        thresholds = sorted({r["threshold"] for r in rows})
        by = {(r["clip"], r["threshold"]): r for r in rows}
        bar = self.validity_bar

        # Best single global threshold: cheapest mean cost whose WORST clip still clears the bar.
        global_best = None
        for thr in thresholds:
            rs = [by[(c, thr)] for c in clips]
            if min(r["validity"] for r in rs) < bar:
                continue
            cost = float(np.mean([r["calls_per_min"] for r in rs]))
            if global_best is None or cost < global_best["mean_calls_per_min"]:
                global_best = {
                    "threshold": thr,
                    "mean_calls_per_min": round(cost, 3),
                    "min_validity": round(min(r["validity"] for r in rs), 4),
                    "mean_recall": round(float(np.mean(
                        [r["event_recall"] for r in rs if r["event_recall"] is not None])), 4),
                }

        # Per-clip oracle: each clip gets its own cheapest threshold clearing the bar.
        per_clip: dict[str, Any] = {}
        for c in clips:
            best = None
            for thr in thresholds:
                r = by[(c, thr)]
                if r["validity"] < bar:
                    continue
                if best is None or r["calls_per_min"] < best["calls_per_min"]:
                    best = {"threshold": thr, "calls_per_min": r["calls_per_min"],
                            "validity": r["validity"], "event_recall": r["event_recall"]}
            per_clip[c] = best

        achievable = [v for v in per_clip.values() if v]
        unreachable = sorted(c for c, v in per_clip.items() if v is None)

        # Headroom is computed over the clips that ANY threshold can serve. A clip no threshold can
        # serve is not a calibration problem to be adapted away — it is the signal failing, and it
        # is reported separately rather than being allowed to silently drop out of the mean.
        per_clip_cost = (round(float(np.mean([v["calls_per_min"] for v in achievable])), 3)
                         if achievable else None)
        headroom = (round(global_best["mean_calls_per_min"] / per_clip_cost, 3)
                    if global_best and per_clip_cost else None)
        spread = (round(max(v["threshold"] for v in achievable)
                        / max(1e-9, min(v["threshold"] for v in achievable)), 2)
                  if len(achievable) > 1 else None)

        return {
            "validity_bar": bar,
            "global_best": global_best,
            "per_clip_best": per_clip,
            "clips_no_threshold_can_serve": unreachable,
            "n_achievable": len(achievable),
            "n_clips": len(clips),
            "per_clip_mean_calls_per_min": per_clip_cost,
            "adaptation_headroom_x": headroom,
            "optimal_threshold_spread": spread,
        }

    def teardown(self) -> None:
        if not self.separability or not self.curves:
            return

        aucs = [r["auc_novelty"] for r in self.separability]
        mean_auc = float(np.mean(aucs))
        clears = sum(r["event_clears_quiet_p95"] for r in self.separability)

        headroom = {k: self._headroom(k) for k in self.curves}
        bar_txt = f"{self.validity_bar:.2f}"
        nov_head = headroom["embedding_novelty"]["adaptation_headroom_x"]
        spread = headroom["embedding_novelty"]["optimal_threshold_spread"]

        # The verdict is computed, not written. Thresholds chosen up front:
        #   AUC < 0.65 anywhere important  -> the signal cannot separate events
        #   headroom < 1.2x                -> per-scene adaptation cannot pay for itself
        nov = headroom["embedding_novelty"]
        unreachable = nov["clips_no_threshold_can_serve"]

        # A clip that NO threshold can serve is the strongest possible evidence of a signal problem:
        # it is not that the cut-point is miscalibrated, it is that no cut-point exists.
        if unreachable:
            verdict, reason = "signal_problem", (
                f"no novelty threshold anywhere in the swept range reaches validity "
                f"{bar_txt} on {unreachable} — the events on those clips do not clear the "
                f"scene's own background novelty (mean AUC {mean_auc:.3f}; the event median sits "
                f"below the quiet p95 on {len(self.separability) - clears} of "
                f"{len(self.separability)} clips with events). Adaptive thresholding cannot fix a "
                "cut-point that does not exist; the fast-tier signal is what needs changing."
            )
        elif mean_auc < 0.65:
            verdict, reason = "signal_problem", (
                f"novelty separates events from quiet frames at only AUC {mean_auc:.3f} averaged "
                f"over clips with events, and clears the quiet p95 on {clears}/"
                f"{len(self.separability)} of them. No cut-point works on any single clip, so an "
                "adaptive threshold would be built on a signal that does not carry the event."
            )
        elif nov_head is None or nov_head < 1.2:
            verdict, reason = "neither", (
                f"the signal separates (AUC {mean_auc:.3f}) but per-scene adaptation is worth only "
                f"{nov_head if nov_head is None else format(nov_head, '.2f')}x — a single global "
                "threshold is already close to the per-clip oracle, so the ceiling is the signal's "
                "quality, not its calibration."
            )
        else:
            verdict, reason = "threshold_problem", (
                f"novelty separates events within each clip (AUC {mean_auc:.3f}), but the optimal "
                f"cut-point varies {spread}x across scenes and per-scene adaptation would cut cost "
                f"{format(nov_head, '.2f')}x. Adaptive thresholding is the right fix."
            )

        self.recorder.record_extra(f"phase8_signal_diagnosis_{self.signal}", {
            "signal": self.signal,
            "separability": self.separability,
            "mean_auc_novelty": round(mean_auc, 4),
            "headroom": headroom,
            "verdict": verdict,
            "reason": reason,
            "definitions": {
                "auc": "ROC AUC of the signal separating frames just after a state change from "
                       "frames far from any change, WITHIN one clip — so per-scene scale is not a "
                       "confound. 0.5 means no information.",
                "adaptation_headroom_x": "cost of the best single global threshold divided by the "
                                         "cost of the best per-clip threshold, both required to "
                                         "clear the same validity bar. The oracle upper bound on "
                                         "what per-scene adaptation could ever buy.",
            },
        })
        _progress(f"\nVERDICT: {verdict.upper()}\n  {reason}")


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.signal_diagnosis",
        description="Phase 8a: threshold problem or signal problem?",
        argv=argv,
    )
    runner = SignalDiagnosisRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                  metrics_out=cfg.run.metrics_out,
                                  headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
