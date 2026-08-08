"""Phase 5 step 2: does the semantic cache earn its place?

Runs the Phase 4 winning policy with and without a cache, across cache thresholds, and reports the
three numbers that decide it: calls avoided, false-hit rate, and the resulting answer validity.

`PROMPT.md`: *if the accuracy cost exceeds the latency benefit, recommend cutting the feature —
four solid components beat five with one that doesn't earn its place.* This runner is built to be
able to return that verdict, and it computes the recommendation rather than leaving it to prose.

    python -m peripheral.cli.cache_sweep --duration 900 --headless \
        --metrics-out results/phase5_cache.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..cache.semantic_cache import SemanticCache
from ..eval.simulate_cache import simulate_with_cache
from ..eval.traces import ClipTrace
from ..runtime import BoundedRunner
from ..scheduler.policies import EmbeddingNoveltyPolicy
from ._args import parse_and_load


class CacheSweepRunner(BoundedRunner):
    name = "phase5_cache_sweep"

    def setup(self) -> None:
        c5 = self.cfg.phase5
        trace_dir = Path(self.cfg.phase4.trace_dir)
        names = sorted({p.stem for p in trace_dir.glob("*.npz")})
        if not names:
            raise FileNotFoundError(f"no traces in {trace_dir}")

        self.traces: dict[str, ClipTrace] = {}
        self.embeddings: dict[str, np.ndarray] = {}
        for n in names:
            self.traces[n] = ClipTrace.load(trace_dir / n)
            arr = np.load(trace_dir / f"{n}.npz")
            if "embeddings" not in arr:
                raise RuntimeError(
                    f"trace {n} has no embeddings — run peripheral.cli.extract_embeddings first"
                )
            self.embeddings[n] = arr["embeddings"]

        self.policy_threshold = float(c5.policy_threshold)
        self.min_gap_s = float(self.cfg.phase4.min_gap_s)
        self.thresholds = [None] + [float(t) for t in c5.cache_thresholds]
        self.max_staleness_s = float(c5.max_staleness_s)
        self.max_entries = int(c5.max_entries)
        self.held_out = list(self.cfg.phase4.held_out_clips)

        self.rows: list[dict[str, Any]] = []
        self._i = 0

    def step(self) -> bool:
        if self._i >= len(self.thresholds):
            return False
        thr = self.thresholds[self._i]
        self._i += 1

        for split, clip_names in (("all", list(self.traces)), ("held_out", self.held_out)):
            sims = []
            for name in clip_names:
                cache = None if thr is None else SemanticCache(
                    threshold=thr, max_entries=self.max_entries,
                    max_staleness_s=self.max_staleness_s,
                )
                policy = EmbeddingNoveltyPolicy(self.policy_threshold, min_gap_s=self.min_gap_s)
                sims.append(simulate_with_cache(
                    self.traces[name], policy, cache, self.embeddings[name],
                    warmup_frames=int(self.cfg.phase4.warmup_frames),
                ))

            agg = _aggregate(sims, thr, split)
            self.rows.append(agg)
            if split == "held_out":
                label = "no cache" if thr is None else f"cache @ {thr:g}"
                print(
                    f"  {label:16s} calls/min {agg['calls_per_min_mean']:6.2f}  "
                    f"avoided {agg['calls_avoided_pct_mean']:5.1f}%  "
                    f"hit rate {agg['cache_hit_rate_mean']}  "
                    f"false hits {agg['false_hit_rate_mean']}  "
                    f"validity {agg['answer_validity_mean']:.3f}",
                    file=sys.stderr,
                )
        return True

    def teardown(self) -> None:
        held = [r for r in self.rows if r["split"] == "held_out"]
        baseline = next((r for r in held if r["cache_threshold"] is None), None)
        verdict = _verdict(held, baseline)
        self.recorder.record_extra("phase5_cache", {
            "rows": self.rows,
            "policy": f"embedding_novelty @ {getattr(self, 'policy_threshold', None)}",
            "verdict": verdict,
            "definitions": {
                "calls_avoided_pct": "share of the policy's triggers served from cache instead of "
                                     "the VLM — the ONLY benefit",
                "false_hit_rate": "share of cache hits that served an answer from a DIFFERENT "
                                  "scene state — a confidently wrong answer at zero cost, which "
                                  "the system cannot detect",
                "answer_validity": "same primary metric as Phase 4, so the accuracy cost of "
                                   "caching is directly comparable",
                "latency_benefit": "A cache hit skips a VLM call whose measured cost is 159 ms "
                                   "p50 TTFT / 487 ms to a complete answer (Phase 3), and 43.8 J.",
            },
        })


def _aggregate(sims: list, thr: float | None, split: str) -> dict[str, Any]:
    def mean(attr: str) -> float:
        return float(np.mean([getattr(s, attr) for s in sims]))

    hits = [s.cache_hit_rate for s in sims if s.cache_hit_rate is not None]
    fhr = [s.false_hit_rate for s in sims if s.false_hit_rate is not None]
    return {
        "cache_threshold": thr,
        "split": split,
        "n_clips": len(sims),
        "calls_per_min_mean": round(mean("calls_per_min"), 3),
        "call_fraction_mean": round(mean("call_fraction"), 5),
        "calls_avoided_pct_mean": round(mean("calls_avoided_pct"), 3),
        "calls_avoided_total": int(sum(s.calls_avoided for s in sims)),
        "n_triggers_total": int(sum(s.n_triggers for s in sims)),
        "cache_hit_rate_mean": round(float(np.mean(hits)), 5) if hits else None,
        "false_hits_total": int(sum(s.false_hits for s in sims)),
        "false_hit_rate_mean": round(float(np.mean(fhr)), 5) if fhr else None,
        "answer_validity_mean": round(mean("answer_validity"), 5),
        "accuracy_mean": round(mean("accuracy"), 5),
        "staleness_ms_mean": round(mean("staleness_ms_mean"), 2),
        "per_clip": {s.clip: {"triggers": s.n_triggers, "vlm_calls": s.n_vlm_calls,
                              "hits": s.n_cache_hits, "false_hits": s.false_hits,
                              "validity": s.answer_validity} for s in sims},
    }


def _verdict(held: list[dict], baseline: dict | None) -> dict[str, Any]:
    """Compute the keep-or-cut recommendation instead of arguing it in prose."""
    if baseline is None:
        return {"recommendation": "inconclusive", "reason": "no no-cache baseline"}

    candidates = [r for r in held if r["cache_threshold"] is not None]
    if not candidates:
        return {"recommendation": "inconclusive", "reason": "no cache configurations"}

    # Best = most calls avoided among configurations that cost no validity.
    safe = [r for r in candidates
            if r["answer_validity_mean"] >= baseline["answer_validity_mean"] - 1e-9]
    best_safe = max(safe, key=lambda r: r["calls_avoided_pct_mean"]) if safe else None
    best_any = max(candidates, key=lambda r: r["calls_avoided_pct_mean"])

    if best_safe and best_safe["calls_avoided_pct_mean"] >= 10.0:
        rec = "keep"
        reason = (
            f"at threshold {best_safe['cache_threshold']:g} the cache avoids "
            f"{best_safe['calls_avoided_pct_mean']:.1f}% of VLM calls with no loss of answer "
            f"validity ({best_safe['answer_validity_mean']:.3f} vs "
            f"{baseline['answer_validity_mean']:.3f} without)."
        )
    elif best_safe:
        rec = "cut"
        reason = (
            f"the best lossless configuration avoids only "
            f"{best_safe['calls_avoided_pct_mean']:.1f}% of calls. The scheduler already runs at "
            f"{baseline['calls_per_min_mean']:.1f} calls/min, so that saving is a rounding error "
            f"against the complexity of a second correctness-critical component."
        )
    else:
        rec = "cut"
        reason = (
            f"every cache configuration costs answer validity "
            f"(best {best_any['answer_validity_mean']:.3f} vs "
            f"{baseline['answer_validity_mean']:.3f} without a cache). A false hit is a "
            f"confidently wrong answer the system cannot detect."
        )

    return {
        "recommendation": rec,
        "reason": reason,
        "baseline_calls_per_min": baseline["calls_per_min_mean"],
        "baseline_validity": baseline["answer_validity_mean"],
        "best_lossless": best_safe,
        "best_any": best_any,
    }


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.cache_sweep",
        description="Phase 5: measure whether the semantic cache earns its place.",
        argv=argv,
    )
    runner = CacheSweepRunner(cfg=cfg, duration_s=cfg.run.duration_s,
                             metrics_out=cfg.run.metrics_out,
                             headless=cfg.run.headless, seed=cfg.seed)
    code = runner.run()
    v = runner.recorder.finalize()["extra"]["phase5_cache"]["verdict"]
    print(f"\nRECOMMENDATION: {v['recommendation'].upper()} — {v['reason']}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
