"""The headline figure: accuracy vs VLM-calls-per-minute.

`PROMPT.md`: *the headline figure: accuracy vs. VLM-calls-per-minute Pareto, all policies on one
plot, publication quality — axis labels, error bars over repeated runs, clearly marked baselines.*

Error bars are spread **across clips**. The policies are deterministic given a trace, so repeating
a run changes nothing; what varies is which scene you point them at, and that is the variation a
reader should see.

    python -m peripheral.viz.phase4_chart --sweep results/phase4_sweep.json \
        --out results/phase4_pareto.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {
    "fixed_interval":    ("#718096", "o", "fixed interval (baseline)"),
    "motion_threshold":  ("#dd6b20", "s", "motion threshold"),
    "embedding_novelty": ("#2b6cb0", "^", "embedding novelty"),
    "information_gain":  ("#2f855a", "D", "information gain"),
    "learned":           ("#c53030", "v", "learned"),
}
GRID = "#cbd5e0"
TARGET_ACC = 0.85
TARGET_CALL_FRACTION = 0.20


def load(p: Path) -> dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _rows(doc: dict, split: str) -> list[dict]:
    return [r for r in doc["extra"]["phase4_sweep"]["results"] if r["split"] == split]


def build(doc: dict, out: Path, split: str = "held_out") -> Path:
    sweep = doc["extra"]["phase4_sweep"]
    rows = _rows(doc, split)
    oracle = next(
        (r for r in rows if r["kind"] == "fixed_interval" and r["operating_point"] == 0.0), None
    )
    oracle_cpm = oracle["calls_per_min_mean"] if oracle else None

    fig = plt.figure(figsize=(16, 10), dpi=130)
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.30, top=0.86, bottom=0.07,
                          left=0.06, right=0.97)

    best = min(
        (r for r in rows if r["kind"] != "fixed_interval" and r["answer_validity_mean"] >= 0.9999),
        key=lambda r: r["calls_per_min_mean"], default=None,
    )
    cheapest_fixed = min(
        (r for r in rows if r["kind"] == "fixed_interval" and r["answer_validity_mean"] >= 0.9999),
        key=lambda r: r["calls_per_min_mean"], default=None,
    )
    headline = ""
    if best and cheapest_fixed:
        headline = (
            f"\n{STYLE[best['kind']][2]} holds a valid answer on every frame at "
            f"{best['calls_per_min_mean']:.1f} calls/min "
            f"({best['call_fraction_mean'] * 100:.2f}% of the oracle) — "
            f"{cheapest_fixed['calls_per_min_mean'] / best['calls_per_min_mean']:.1f}x cheaper "
            f"than fixed interval for the same result"
        )

    # The same comparison across all six clips, because the held-out advantage does not survive it
    # and a figure that hid that would be overclaiming.
    all_rows = _rows(doc, "all")
    caveat = ""
    best_all = min(
        (r for r in all_rows if r["kind"] != "fixed_interval"
         and r["answer_validity_mean"] >= 0.9999),
        key=lambda r: r["calls_per_min_mean"], default=None,
    )
    fixed_all = min(
        (r for r in all_rows if r["kind"] == "fixed_interval"
         and r["answer_validity_mean"] >= 0.9999),
        key=lambda r: r["calls_per_min_mean"], default=None,
    )
    if best_all and fixed_all:
        caveat = (
            f"\nBUT across all 6 clips the advantage does not hold: fixed interval reaches perfect "
            f"validity at {fixed_all['calls_per_min_mean']:.0f} calls/min vs "
            f"{best_all['calls_per_min_mean']:.0f} for {STYLE[best_all['kind']][2]} — "
            "a single global threshold does not generalise across scenes"
        )

    fig.suptitle(
        "Phase 4 — answer validity vs. VLM invocations, on held-out clips" + headline + caveat +
        "\nerror bars are spread across clips · latency is NOT measured here (see Phase 3)",
        fontsize=12.5, fontweight="bold", y=0.99,
    )

    ceiling = (sweep.get("text_agreement_ceiling") or {}).get("mean")
    _panel_pareto(fig.add_subplot(gs[0:2, 0:2]), rows, oracle_cpm)
    _panel_false_triggers(fig.add_subplot(gs[0, 2]), doc)
    _panel_ftr_vs_calls(fig.add_subplot(gs[1, 2]), rows)
    _panel_recall(fig.add_subplot(gs[2, 0]), rows)
    _panel_text_agreement(fig.add_subplot(gs[2, 1]), rows, ceiling)
    _panel_weights(fig.add_subplot(gs[2, 2]), sweep.get("learned_policy", {}))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _panel_pareto(ax, rows, oracle_cpm) -> None:
    for kind, (color, marker, label) in STYLE.items():
        pts = sorted([r for r in rows if r["kind"] == kind],
                     key=lambda r: r["calls_per_min_mean"])
        if not pts:
            continue
        x = np.array([p["calls_per_min_mean"] for p in pts])
        y = np.array([p["answer_validity_mean"] for p in pts])
        yerr = np.array([p["answer_validity_std"] for p in pts])
        xerr = np.array([p["calls_per_min_std"] for p in pts])
        ls = "--" if kind == "fixed_interval" else "-"
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, color=color, marker=marker, ls=ls,
                    lw=1.8, ms=7, capsize=3, elinewidth=1.0, label=label, zorder=3,
                    alpha=0.95)

    # The exit criterion, drawn as a target box.
    if oracle_cpm:
        budget = TARGET_CALL_FRACTION * oracle_cpm
        ax.axvline(budget, color="#805ad5", ls=":", lw=2)
        ax.axhline(TARGET_ACC, color="#805ad5", ls=":", lw=2)
        ax.add_patch(plt.Rectangle((0, TARGET_ACC), budget, 1.02 - TARGET_ACC,
                                   facecolor="#805ad5", alpha=0.10, zorder=0))
        ax.text(budget * 0.55, TARGET_ACC - 0.02,
                f"exit criterion: >={TARGET_ACC:.0%} validity\n"
                f"at <={TARGET_CALL_FRACTION:.0%} of oracle calls ({budget:.0f}/min)",
                fontsize=8.5, color="#553c9a", fontweight="bold", ha="center", va="top")

    ax.set_xscale("log")
    ax.set_xlabel("VLM invocations per minute (log scale) — the cost axis")
    ax.set_ylabel("answer validity\n(fraction of frames whose held answer describes the current scene)")
    ax.set_title("THE HEADLINE: accuracy vs. cost", fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.3, color=GRID, which="both")
    ax.set_ylim(0.3, 1.04)


def _panel_false_triggers(ax, doc) -> None:
    """Calls fired on the probe clips, where by construction nothing semantic ever happens."""
    per_clip = doc["extra"]["phase4_sweep"]["per_clip"]
    probes = ("lighting_drift", "rapid_motion")
    kinds = [k for k in STYLE if k != "fixed_interval"]

    # Compare at each policy's operating point closest to a 60 calls/min budget.
    bars, labels, colors = [], [], []
    for kind in kinds:
        rows = [r for r in per_clip if r["policy"].startswith(kind.split("_")[0])
                or r["policy"] == kind]
        rows = [r for r in per_clip if r["clip"] in probes and _kind_of(r["policy"]) == kind]
        if not rows:
            continue
        target = min(rows, key=lambda r: abs(r["calls_per_min"] - 60.0))
        matched = [r for r in rows if r["operating_point"] == target["operating_point"]]
        bars.append(float(np.mean([r["n_calls"] for r in matched])))
        labels.append(STYLE[kind][2].replace(" ", "\n"))
        colors.append(STYLE[kind][0])

    if not bars:
        ax.text(0.5, 0.5, "no probe data", ha="center", transform=ax.transAxes)
        return
    ax.bar(range(len(bars)), bars, 0.6, color=colors)
    for i, v in enumerate(bars):
        ax.text(i, v + max(bars) * 0.03, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels)), labels, fontsize=7.5)
    ax.set_ylabel("calls on probe clips")
    ax.set_title("False triggers where NOTHING happens\n(lighting drift + rapid motion, "
                 "matched budget)", fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, color=GRID)


def _panel_ftr_vs_calls(ax, rows) -> None:
    for kind, (color, marker, label) in STYLE.items():
        pts = sorted([r for r in rows if r["kind"] == kind and
                      r["false_trigger_rate_mean"] is not None],
                     key=lambda r: r["calls_per_min_mean"])
        if not pts:
            continue
        ax.plot([p["calls_per_min_mean"] for p in pts],
                [p["false_trigger_rate_mean"] for p in pts],
                color=color, marker=marker, ms=5, lw=1.5,
                ls="--" if kind == "fixed_interval" else "-", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("calls per minute (log)")
    ax.set_ylabel("false-trigger rate")
    ax.set_title("Fraction of calls that bought nothing", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, color=GRID, which="both")


def _panel_text_agreement(ax, rows, ceiling) -> None:
    """The secondary metric, plotted against the ceiling it can actually reach."""
    for kind, (color, marker, label) in STYLE.items():
        pts = sorted([r for r in rows if r["kind"] == kind],
                     key=lambda r: r["calls_per_min_mean"])
        if not pts:
            continue
        ax.plot([p["calls_per_min_mean"] for p in pts], [p["accuracy_mean"] for p in pts],
                color=color, marker=marker, ms=5, lw=1.5,
                ls="--" if kind == "fixed_interval" else "-")
    if ceiling:
        ax.axhline(ceiling, color="#805ad5", ls=":", lw=2)
        ax.text(ax.get_xlim()[0], ceiling + 0.01,
                f" ceiling {ceiling:.3f} — the model's own reproducibility",
                fontsize=8, color="#553c9a", fontweight="bold", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("calls per minute (log)")
    ax.set_ylabel("text agreement (content-F1)")
    ax.set_title("Secondary metric: text agreement.\nDecays with staleness even when no event "
                 "was missed", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3, color=GRID, which="both")


def _panel_recall(ax, rows) -> None:
    for kind, (color, marker, label) in STYLE.items():
        pts = sorted([r for r in rows if r["kind"] == kind and
                      r["event_recall_mean"] is not None],
                     key=lambda r: r["calls_per_min_mean"])
        if not pts:
            continue
        ax.plot([p["calls_per_min_mean"] for p in pts],
                [p["event_recall_mean"] for p in pts],
                color=color, marker=marker, ms=5, lw=1.5,
                ls="--" if kind == "fixed_interval" else "-", label=label)
    ax.set_xscale("log")
    ax.set_xlabel("calls per minute (log)")
    ax.set_ylabel("event recall")
    ax.set_title("Semantic events actually caught", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, color=GRID, which="both")
    ax.legend(fontsize=7, loc="lower right")


def _panel_staleness(ax, rows) -> None:
    for kind, (color, marker, label) in STYLE.items():
        pts = sorted([r for r in rows if r["kind"] == kind],
                     key=lambda r: r["calls_per_min_mean"])
        if not pts:
            continue
        ax.plot([p["calls_per_min_mean"] for p in pts],
                [p["staleness_ms_mean"] for p in pts],
                color=color, marker=marker, ms=5, lw=1.5,
                ls="--" if kind == "fixed_interval" else "-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("calls per minute (log)")
    ax.set_ylabel("mean answer staleness (ms, log)")
    ax.set_title("How old the held answer is", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, color=GRID, which="both")


def _panel_weights(ax, learned) -> None:
    names = learned.get("feature_names") or []
    weights = learned.get("weights") or []
    if not names:
        ax.text(0.5, 0.5, "learned policy not fitted", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return
    y = np.arange(len(names))
    colors = ["#2b6cb0" if w >= 0 else "#c53030" for w in weights]
    ax.barh(y, weights, 0.6, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel("weight")
    ax.set_title(
        f"Learned policy\n(fit on {len(learned.get('train_clips', []))} training clips, "
        f"{learned.get('n_positive', 0)} positives)",
        fontsize=10.5, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, color=GRID)


def _kind_of(policy_name: str) -> str:
    if policy_name.startswith("fixed") or policy_name == "oracle":
        return "fixed_interval"
    return policy_name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Phase 4 headline Pareto figure.")
    p.add_argument("--sweep", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", default="held_out", choices=["held_out", "train", "all"])
    a = p.parse_args(argv)
    out = build(load(a.sweep), a.out, split=a.split)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
