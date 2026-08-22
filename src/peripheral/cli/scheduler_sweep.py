"""Phase 4 step 3: sweep every policy across its operating range. **The core experiment.**

Produces the data behind the headline figure: accuracy vs VLM-calls-per-minute, every policy on one
plot, with spread across clips as error bars.

Clips are split into **train** and **held-out**. The learned policy is fit on train only, and the
exit criterion is asserted on held-out clips — otherwise the learned policy would be graded on the
clips it memorised, which is the one way this experiment could flatter itself.

    python -m peripheral.cli.scheduler_sweep --duration 900 --headless \
        --metrics-out results/phase4_sweep.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..eval.simulate import aggregate, metric_ceiling, simulate
from ..eval.traces import ClipTrace
from ..runtime import BoundedRunner
from ..scheduler.policies import (
    AdaptiveNoveltyPolicy,
    EmbeddingNoveltyPolicy,
    FixedIntervalPolicy,
    InformationGainPolicy,
    LearnedPolicy,
    MotionThresholdPolicy,
)
from ..scheduler.train import fit
from ._args import parse_and_load

POLICY_CLASSES = {
    "adaptive_novelty": AdaptiveNoveltyPolicy,
    "fixed_interval": FixedIntervalPolicy,
    "motion_threshold": MotionThresholdPolicy,
    "embedding_novelty": EmbeddingNoveltyPolicy,
    "information_gain": InformationGainPolicy,
    "learned": LearnedPolicy,
}


class SchedulerSweepRunner(BoundedRunner):
    name = "phase4_scheduler_sweep"

    def setup(self) -> None:
        c = self.cfg.phase4
        trace_dir = Path(c.trace_dir)
        names = sorted({p.stem for p in trace_dir.glob("*.npz")})
        if not names:
            raise FileNotFoundError(
                f"no traces in {trace_dir} — run peripheral.cli.build_traces first"
            )
        signal = str(self.cfg.phase8.get("signal", "pooled"))
        self.signal = signal
        self.traces = {n: ClipTrace.load(trace_dir / n).with_signal(signal) for n in names}

        held = list(c.held_out_clips)
        self.held_out = [n for n in names if n in held]
        self.train = [n for n in names if n not in held]
        if not self.held_out:
            raise ValueError(f"none of the held-out clips {held} are present in {names}")

        # Fit the learned policy on training clips only.
        self.learned = fit(
            [self.traces[n] for n in self.train],
            min_gap_s=float(c.min_gap_s),
            seed=self.seed,
        )
        self.recorder.note(f"learned policy trained on {self.train}, held out {self.held_out}")

        self.jobs = self._build_jobs()
        self.results: list[dict[str, Any]] = []
        self.raw: list[Any] = []
        self._i = 0

    def _build_jobs(self) -> list[tuple[str, float]]:
        jobs: list[tuple[str, float]] = []
        for kind, cls in POLICY_CLASSES.items():
            for point in cls.operating_points():
                jobs.append((kind, point))
        return jobs

    def _make_policy(self, kind: str, point: float):
        gap = float(self.cfg.phase4.min_gap_s)
        if kind == "fixed_interval":
            return FixedIntervalPolicy(point)
        if kind == "motion_threshold":
            return MotionThresholdPolicy(point, min_gap_s=gap)
        if kind == "embedding_novelty":
            return EmbeddingNoveltyPolicy(point, min_gap_s=gap)
        if kind == "information_gain":
            return InformationGainPolicy(
                point, staleness_half_life_s=float(self.cfg.phase4.staleness_half_life_s),
                min_gap_s=gap,
            )
        if kind == "adaptive_novelty":
            return AdaptiveNoveltyPolicy(
                point, window_s=float(self.cfg.phase8.adaptive_window_s), min_gap_s=gap
            )
        if kind == "learned":
            return LearnedPolicy(self.learned["weights"], self.learned["bias"],
                                 threshold=point, min_gap_s=gap)
        raise ValueError(kind)

    def step(self) -> bool:
        if self._i >= len(self.jobs):
            return False
        kind, point = self.jobs[self._i]
        self._i += 1

        for split_name, clip_names in (("all", list(self.traces)),
                                       ("held_out", self.held_out),
                                       ("train", self.train)):
            sims = []
            for n in clip_names:
                policy = self._make_policy(kind, point)
                sims.append(simulate(self.traces[n], policy,
                                     warmup_frames=int(self.cfg.phase4.warmup_frames)))
            agg = aggregate(sims)
            agg["kind"] = kind
            agg["split"] = split_name
            self.results.append(agg)
            if split_name == "held_out":
                self.raw.extend(s.as_row() for s in sims)
                print(
                    f"  {kind:18s} @ {point:<7.4g} "
                    f"valid {agg['answer_validity_mean']:.3f}+-{agg['answer_validity_std']:.3f}  "
                    f"F1 {agg['accuracy_mean']:.3f}  "
                    f"{agg['calls_per_min_mean']:7.1f} calls/min  "
                    f"({agg['call_fraction_mean'] * 100:5.1f}% of oracle)  "
                    f"FTR {agg['false_trigger_rate_mean']}",
                    file=sys.stderr,
                )
        return True

    def teardown(self) -> None:
        ceilings = {n: round(metric_ceiling(t), 5) for n, t in self.traces.items()}
        self.recorder.record_extra("phase4_sweep", {
            "signal": self.signal,
            "results": self.results,
            "per_clip": self.raw,
            "learned_policy": self.learned,
            "train_clips": self.train,
            "held_out_clips": self.held_out,
            "text_agreement_ceiling": {
                "per_clip": ceilings,
                "mean": round(sum(ceilings.values()) / len(ceilings), 5),
                "what_it_is": (
                    "Agreement between the oracle's answers on two ADJACENT frames in the same "
                    "scene state — nothing semantic changed, so every difference is serving noise. "
                    "This is the highest `accuracy` any non-oracle policy can reach. The oracle "
                    "scores 1.0 only because it is compared against its own cached string."
                ),
            },
            "definitions": {
                "answer_validity": (
                    "PRIMARY METRIC. Fraction of frames on which the answer being held describes "
                    "the scene state the camera is actually in. Immune to paraphrase, 1.0 for the "
                    "oracle by construction."
                ),
                "accuracy": (
                    "SECONDARY. Mean content-F1 between the answer the system is CURRENTLY HOLDING "
                    "and the per-frame oracle's answer for that frame. Read against "
                    "text_agreement_ceiling, NOT against 1.0: it decays with staleness even when "
                    "no event was missed, because the model paraphrases itself, so it largely "
                    "measures staleness rather than correctness. That is why it is not the "
                    "headline metric."
                ),
                "call_fraction": "calls / frames — the fraction of the per-frame oracle's budget",
                "false_trigger_rate": (
                    "Fraction of calls fired while the scene state had not changed since the "
                    "previous call. Ground truth is exact because events were composited in."
                ),
                "error_bars": (
                    "Spread ACROSS CLIPS. Policies are deterministic given a trace, so repeating a "
                    "run changes nothing; what changes is the scene."
                ),
                "latency": "NOT measured here. See Phase 3 for end-to-end latency.",
            },
        })


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.scheduler_sweep",
        description="Phase 4: sweep trigger policies against the per-frame oracle.",
        argv=argv,
    )
    runner = SchedulerSweepRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                                 metrics_out=cfg.run.metrics_out,
                                 headless=cfg.run.headless, seed=cfg.seed)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
