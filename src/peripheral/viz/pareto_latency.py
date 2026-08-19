"""Figure 2: accuracy vs. p95 photon-to-answer latency.

`PROMPT.md` names two figures this project exists to produce. Figure 1 is accuracy vs
VLM-calls-per-minute (`phase4_chart`). This is figure 2.

**What is measured, and what is not.** The full three-way cross (model × quantization × scheduler)
was not run: scoring a scheduler needs a per-frame oracle for that model, which is 4,320 VLM calls
per configuration — about six hours for the eight configurations in the Phase 3 sweep. So this plots
the two dimensions that *were* measured, and says so on the figure rather than interpolating a
surface nobody ran:

* **left** — model × quantization: measured p95 TTFT against quantization fidelity, with the
  400 ms target and the 12 GB budget marked.
* **right** — the scheduler dimension at the chosen configuration: answer validity against
  end-to-end p95 photon-to-answer latency, where latency is fixed by the model and the scheduler
  moves cost, not latency.

    python -m peripheral.viz.pareto_latency --bench results/phase3_vlm_bench.json \
        --sweep results/phase4_sweep.json --pipeline results/phase3_chosen_pipeline.json \
        --out results/pareto_latency.png
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

TARGET_TTFT_MS = 400.0
FAMILY_COLOR = {
    "SmolVLM2-500M": "#2b6cb0",
    "SmolVLM2-2.2B": "#2f855a",
    "Qwen3-VL-2B": "#c53030",
    "Qwen3-VL-4B": "#6b46c1",
}
POLICY_COLOR = {
    "fixed_interval": "#718096",
    "motion_threshold": "#dd6b20",
    "embedding_novelty": "#2b6cb0",
    "information_gain": "#2f855a",
    "learned": "#c53030",
}
GRID = "#cbd5e0"


def load(p: Path) -> dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build(bench: dict, sweep: dict, pipeline: dict | None, out: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), dpi=130)

    p95_answer = None
    if pipeline:
        pa = pipeline["metrics"].get("photon_to_answer_ms")
        p95_answer = pa["p95"] if pa else None

    fig.suptitle(
        "Figure 2 — accuracy vs. latency.  Latency is set by the model; the scheduler moves cost.\n"
        "The full model x quantization x scheduler cross was NOT run: each cell needs a per-frame "
        "oracle (4,320 VLM calls, ~6 h for 8 configs).",
        fontsize=12, fontweight="bold", y=0.99,
    )

    _panel_models(ax1, bench)
    _panel_scheduler(ax2, sweep, p95_answer)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _panel_models(ax, bench) -> None:
    rows = [r for r in bench["extra"]["vlm_bench"]["configs"] if "error" not in r]
    # Family references all sit at exactly 1.0 by definition, so stagger their labels downward in
    # latency order or they overprint each other.
    refs = sorted([r for r in rows if r["label"] == r.get("family_reference")],
                  key=lambda z: z["ttft_p95_ms"])
    ref_rank = {r["label"]: i for i, r in enumerate(refs)}

    for r in rows:
        q = r.get("quality_vs_family_reference") or {}
        f1 = q.get("content_f1_mean")
        is_ref = r["label"] == r.get("family_reference")
        y = 1.0 if is_ref else f1
        if y is None:
            continue
        color = FAMILY_COLOR.get(r["family"], "#718096")
        ax.scatter(r["ttft_p95_ms"], y, s=60 + r["peak_vram_gb"] * 55, color=color,
                   marker="*" if is_ref else "o", edgecolor="white", linewidth=1.4, zorder=3,
                   alpha=0.95)
        dy = -(13 + 13 * ref_rank[r["label"]]) if is_ref else 7
        ax.annotate(f"{r['family']} {r['quant']}", (r["ttft_p95_ms"], y),
                    textcoords="offset points", xytext=(8, dy), fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78))

    ax.axvline(TARGET_TTFT_MS, color="#dd6b20", ls="--", lw=2)
    ax.text(TARGET_TTFT_MS - 8, 0.72, f"target p95 < {TARGET_TTFT_MS:.0f} ms", rotation=90,
            fontsize=8.5, color="#dd6b20", fontweight="bold", ha="right", va="bottom")
    ax.set_xlabel("p95 TTFT, ms (frame in hand -> first token, includes JPEG + HTTP + vision encode)")
    ax.set_ylabel("quantization fidelity\n(content-F1 vs the family's highest-precision variant)")
    ax.set_title("model x quantization  ·  marker size = peak VRAM  ·  ★ = family reference",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, TARGET_TTFT_MS * 1.12)
    ax.grid(alpha=0.3, color=GRID)


def _panel_scheduler(ax, sweep, p95_answer) -> None:
    rows = [r for r in sweep["extra"]["phase4_sweep"]["results"] if r["split"] == "held_out"]
    for kind, color in POLICY_COLOR.items():
        pts = sorted([r for r in rows if r["kind"] == kind],
                     key=lambda r: r["calls_per_min_mean"])
        if not pts:
            continue
        ax.plot([p["calls_per_min_mean"] for p in pts],
                [p["answer_validity_mean"] for p in pts],
                marker="o", ms=5, lw=1.6, color=color,
                ls="--" if kind == "fixed_interval" else "-",
                label=kind.replace("_", " "))

    ax.set_xscale("log")
    ax.set_xlabel("VLM invocations per minute (log)")
    ax.set_ylabel("answer validity vs. per-frame oracle")
    title = "scheduler dimension, at the chosen configuration"
    if p95_answer:
        title += f"\nlatency is fixed at p95 photon-to-answer {p95_answer:.0f} ms for every point"
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0.3, 1.04)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, color=GRID, which="both")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build figure 2: accuracy vs latency.")
    p.add_argument("--bench", required=True, type=Path)
    p.add_argument("--sweep", required=True, type=Path)
    p.add_argument("--pipeline", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    out = build(load(a.bench), load(a.sweep),
                load(a.pipeline) if a.pipeline and a.pipeline.exists() else None, a.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
