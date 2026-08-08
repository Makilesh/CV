"""The Phase 3 slow-tier chart: what each model x quantization costs and what it keeps.

    python -m peripheral.viz.phase3_chart \
        --bench results/phase3_vlm_bench.json \
        --kv results/phase3_kv_reuse.json \
        --out results/phase3_slow_tier.png
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
VRAM_LIMIT_GB = 11.94

FAMILY_COLOR = {
    "SmolVLM2-500M": "#2b6cb0",
    "SmolVLM2-2.2B": "#2f855a",
    "Qwen3-VL-2B": "#c53030",
    "Qwen3-VL-4B": "#6b46c1",
}
GRID = "#cbd5e0"


def load(p: Path) -> dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _rows(bench: dict) -> list[dict]:
    return [r for r in bench["extra"]["vlm_bench"]["configs"] if "error" not in r]


def _color(r: dict) -> str:
    return FAMILY_COLOR.get(r.get("family", ""), "#718096")


def build(bench: dict, kv: dict | None, out: Path) -> Path:
    rows = sorted(_rows(bench), key=lambda r: r["ttft_p50_ms"])
    fig = plt.figure(figsize=(16, 10), dpi=130)
    gs = fig.add_gridspec(3, 3, hspace=0.70, wspace=0.30, top=0.87, bottom=0.07,
                          left=0.06, right=0.97)

    passing = [r for r in rows if r["ttft_p95_ms"] <= TARGET_TTFT_MS]
    fig.suptitle(
        "Phase 3 slow tier — small VLMs x GGUF quantization on a 12 GB laptop GPU\n"
        f"{len(passing)} of {len(rows)} configurations meet the p95 TTFT < {TARGET_TTFT_MS:.0f} ms "
        "target · frame-in-hand to first token, including JPEG encode, HTTP and vision encoding",
        fontsize=13.5, fontweight="bold", y=0.975,
    )

    _panel_ttft(fig.add_subplot(gs[0, :2]), rows)
    _panel_vram(fig.add_subplot(gs[0, 2]), rows)
    _panel_pareto(fig.add_subplot(gs[1, :2]), rows)
    _panel_tokens(fig.add_subplot(gs[1, 2]), rows)
    _panel_kv(fig.add_subplot(gs[2, :2]), kv)
    _panel_load(fig.add_subplot(gs[2, 2]), rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _panel_ttft(ax, rows) -> None:
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    p50 = [r["ttft_p50_ms"] for r in rows]
    p95 = [r["ttft_p95_ms"] for r in rows]

    ax.barh(y, p50, 0.62, color=[_color(r) for r in rows], label="p50")
    ax.barh(y, np.array(p95) - np.array(p50), 0.62, left=p50, color="black", alpha=0.25,
            label="p50 to p95")
    for i, (a, b) in enumerate(zip(p50, p95)):
        ax.text(b + 8, i, f"{a:.0f} / {b:.0f}", va="center", fontsize=8)

    ax.axvline(TARGET_TTFT_MS, color="#dd6b20", ls="--", lw=2)
    ax.text(TARGET_TTFT_MS + 6, len(rows) - 0.4, f"target p95 < {TARGET_TTFT_MS:.0f} ms",
            fontsize=8.5, color="#dd6b20", fontweight="bold", rotation=90, va="top")
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel("time to first token, ms (frame in hand -> first token)")
    ax.set_title("TTFT — everything between the frame and the first token is inside this number",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, color=GRID)


def _panel_vram(ax, rows) -> None:
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    vram = [r["peak_vram_gb"] for r in rows]
    ax.barh(y, vram, 0.62, color=[_color(r) for r in rows])
    for i, v in enumerate(vram):
        ax.text(v + 0.12, i, f"{v:.2f}", va="center", fontsize=8)
    ax.axvline(VRAM_LIMIT_GB, color="#c53030", ls="--", lw=2)
    ax.text(VRAM_LIMIT_GB - 0.3, len(rows) - 0.4, f"{VRAM_LIMIT_GB:.1f} GB", fontsize=8.5,
            color="#c53030", fontweight="bold", rotation=90, va="top", ha="right")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("peak VRAM, GB (NVML board-wide)")
    ax.set_xlim(0, VRAM_LIMIT_GB * 1.05)
    ax.set_title("The binding constraint", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, color=GRID)


def _panel_pareto(ax, rows) -> None:
    """Fidelity to the family's highest-precision variant vs the cost of getting it."""
    # Every family reference sits at exactly 1.0 by definition, so their labels would pile up on
    # one line. Stagger them downward in TTFT order.
    refs = sorted([r for r in rows if r["label"] == r.get("family_reference")],
                  key=lambda z: z["ttft_p95_ms"])
    ref_rank = {r["label"]: i for i, r in enumerate(refs)}

    for r in rows:
        q = r.get("quality_vs_family_reference") or {}
        f1 = q.get("content_f1_mean")
        if f1 is None:
            continue
        is_ref = r["label"] == r.get("family_reference")
        ax.scatter(r["ttft_p95_ms"], f1, s=190 if is_ref else 150, color=_color(r),
                   marker="*" if is_ref else "o", edgecolor="white", linewidth=1.5, zorder=3)
        dy = -(12 + 13 * ref_rank[r["label"]]) if is_ref else 6
        ax.annotate(r["label"], (r["ttft_p95_ms"], f1), textcoords="offset points",
                    xytext=(9, dy), fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    ax.axvline(TARGET_TTFT_MS, color="#dd6b20", ls="--", lw=2)
    ax.set_xlabel("p95 TTFT (ms)")
    ax.set_ylabel("content F1 vs the family's\nhighest-precision variant")
    ax.set_title(
        "Quantization fidelity vs latency  (★ = the reference each family is scored against, "
        "so it sits at 1.0 by definition)",
        fontsize=10.5, fontweight="bold",
    )
    ax.grid(alpha=0.3, color=GRID)


def _panel_tokens(ax, rows) -> None:
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    tps = [r["tokens_per_s"] for r in rows]
    ax.barh(y, tps, 0.62, color=[_color(r) for r in rows])
    for i, v in enumerate(tps):
        ax.text(v + 1, i, f"{v:.0f}", va="center", fontsize=8)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("tokens / second")
    ax.set_title("Decode throughput", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, color=GRID)


def _panel_kv(ax, kv) -> None:
    if not kv:
        ax.text(0.5, 0.5, "KV-reuse benchmark not supplied", ha="center", transform=ax.transAxes)
        ax.axis("off")
        return
    arms = [a for a in kv["extra"]["kv_reuse"]["arms"] if "error" not in a]
    labels = [a["label"] for a in arms]
    x = np.arange(len(arms))
    ttft = [a["ttft_p50_ms"] if a["ttft_p50_ms"] is not None else 0 for a in arms]
    total = [a["total_p50_ms"] for a in arms]

    w = 0.38
    ax.bar(x - w / 2, ttft, w, label="TTFT p50", color="#2b6cb0")
    ax.bar(x + w / 2, total, w, label="complete answer p50", color="#a0aec0")
    for i, (t, tt) in enumerate(zip(ttft, total)):
        if t > 0:
            ax.text(i - w / 2, t + 6, f"{t:.0f}", ha="center", fontsize=8, fontweight="bold")
        else:
            ax.text(i - w / 2, 12, "no first\ntoken", ha="center", fontsize=7.5,
                    color="#c53030", fontweight="bold")
        ax.text(i + w / 2, tt + 6, f"{tt:.0f}", ha="center", fontsize=8)

    ax.set_xticks(x, [l.replace(", ", ",\n") for l in labels], fontsize=8)
    ax.set_ylabel("milliseconds")
    ax.set_title(
        "KV-cache reuse and streaming decode — image-first leaves no cacheable prefix, "
        "because image tokens differ every frame",
        fontsize=10.5, fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, color=GRID)


def _panel_load(ax, rows) -> None:
    labels = [r["label"] for r in rows]
    y = np.arange(len(rows))
    load_s = [r["load_time_s"] for r in rows]
    ax.barh(y, load_s, 0.62, color=[_color(r) for r in rows])
    for i, v in enumerate(load_s):
        ax.text(v + 0.15, i, f"{v:.1f}s", va="center", fontsize=8)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("seconds")
    ax.set_title("Cold load time\n(startup cost, outside the measured window)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, color=GRID)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Phase 3 slow-tier chart.")
    p.add_argument("--bench", required=True, type=Path)
    p.add_argument("--kv", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    out = build(load(a.bench), load(a.kv) if a.kv and a.kv.exists() else None, a.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
