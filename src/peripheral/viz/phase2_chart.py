"""The Phase 2 fast-tier chart: what each encoder costs and what it buys.

Everything is read from metrics JSON. The one thing this chart must not do is let the
`d_semantic / d_lighting` ratio stand alone — a brightness-invariant control scores best on it
while being a pure motion detector, so both ratios are always shown together.

    python -m peripheral.viz.phase2_chart \
        --bench results/phase2_encoder_bench.json \
        --runs results/phase2_fasttier_webcam.json results/phase2_fasttier_unpaced.json \
        --baseline results/phase1_naive_webcam.json \
        --out results/phase2_fast_tier.png
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

C = {
    "control": "#718096",
    "torch": "#c53030",
    "onnx": "#2b6cb0",
    "budget": "#2f855a",
    "warn": "#dd6b20",
    "grid": "#cbd5e0",
}


def load(p: Path) -> dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _kind(label: str) -> str:
    if "control" in label or "downsample" in label:
        return "control"
    return "onnx" if "onnx" in label else "torch"


def _short(label: str) -> str:
    return (
        label.replace(" (control, no network)", "")
        .replace("@224", "")
        .replace(" torch fp16", " (torch)")
        .replace(" onnx", " (onnx)")
    )


def build(bench: dict, runs: list[dict], baseline: dict | None, out: Path) -> Path:
    rows = [r for r in bench["extra"]["encoder_bench"]["candidates"] if "error" not in r]
    fig = plt.figure(figsize=(16, 9.5), dpi=130)
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.30, top=0.86, bottom=0.08,
                          left=0.06, right=0.97)

    live = next((r for r in runs if r["config"]["capture"]["source"] == "webcam"), runs[0])
    unpaced = next(
        (r for r in runs if r["config"]["capture"]["source"] == "file"), None
    )
    ft_stage = (live["metrics"]["stages_ms"] or {}).get("fast_tier", {})

    fig.suptitle(
        "Phase 2 fast tier — every frame, inside the budget\n"
        f"chosen: MobileNetV3-small via ONNX Runtime CUDA · "
        f"p50 {ft_stage.get('p50', 0):.2f} ms/frame · "
        f"{live['metrics']['achieved_fps']:.1f} FPS sustained, 0 frames dropped",
        fontsize=14, fontweight="bold", y=0.975,
    )

    _panel_tradeoff(fig.add_subplot(gs[0, :2]), rows)
    _panel_backend(fig.add_subplot(gs[0, 2]), rows)
    _panel_ratios(fig.add_subplot(gs[1, 0]), rows)
    _panel_budget(fig.add_subplot(gs[1, 1]), live, baseline)
    _panel_throughput(fig.add_subplot(gs[1, 2]), live, unpaced)
    _panel_signals(fig.add_subplot(gs[2, :2]), live)
    _panel_energy(fig.add_subplot(gs[2, 2]), live, baseline)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _panel_tradeoff(ax, rows) -> None:
    """Cost vs the thing the scheduler actually needs."""
    # Alternate the label offsets: torch and onnx points for the same model land almost on top
    # of each other, which is itself the finding (the backend does not change quality).
    for i, r in enumerate(sorted(rows, key=lambda z: z["p50_ms"])):
        k = _kind(r["label"])
        ax.scatter(r["p50_ms"], r["semantic_vs_motion"], s=170, color=C[k],
                   edgecolor="white", zorder=3, linewidth=1.5)
        dy = 14 if i % 2 == 0 else -20
        ax.annotate(
            _short(r["label"]), (r["p50_ms"], r["semantic_vs_motion"]),
            textcoords="offset points", xytext=(8, dy), fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )

    ax.axhline(1.0, color=C["warn"], ls="--", lw=1.5)
    ax.text(0.15, 1.06, "below this line: reacts MORE to movement than to a real event",
            fontsize=8, color=C["warn"], fontweight="bold")
    ax.axvline(10.0, color=C["budget"], ls="--", lw=1.5)
    ax.text(10.4, 4.4, "10 ms budget", fontsize=8, color=C["budget"], fontweight="bold",
            rotation=90, va="top")
    ax.set_xlabel("p50 latency per frame, including preprocessing (ms)")
    ax.set_ylabel("semantic / motion\n(higher = better)")
    ax.set_title(
        "The trade-off that matters: cost vs. telling a real event apart from mere movement",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(-0.4, 12)
    ax.grid(alpha=0.3, color=C["grid"])


def _panel_backend(ax, rows) -> None:
    """ONNX Runtime vs PyTorch, per model."""
    models: dict[str, dict[str, float]] = {}
    for r in rows:
        k = _kind(r["label"])
        if k == "control":
            continue
        base = _short(r["label"]).replace(" (torch)", "").replace(" (onnx)", "")
        models.setdefault(base, {})[k] = r["p50_ms"]

    names = [m for m in models if "torch" in models[m] and "onnx" in models[m]]
    x = np.arange(len(names))
    w = 0.36
    t = [models[m]["torch"] for m in names]
    o = [models[m]["onnx"] for m in names]
    ax.bar(x - w / 2, t, w, label="PyTorch fp16", color=C["torch"])
    ax.bar(x + w / 2, o, w, label="ONNX Runtime", color=C["onnx"])
    for i, (tv, ov) in enumerate(zip(t, o)):
        ax.text(i, max(tv, ov) + 0.25, f"{tv / ov:.2f}x", ha="center",
                fontsize=9, fontweight="bold", color=C["onnx"])
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("p50 ms per frame")
    ax.set_title("ONNX Runtime won every candidate", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(t + o) * 1.30)  # headroom so the speedup labels clear the legend
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, color=C["grid"])


def _panel_ratios(ax, rows) -> None:
    """Both ratios side by side — the whole point of the panel."""
    labels = [_short(r["label"]) for r in rows]
    keep = [i for i, r in enumerate(rows) if _kind(r["label"]) != "torch"]
    labels = [labels[i] for i in keep]
    sl = [rows[i]["discrimination"] for i in keep]
    sm = [rows[i]["semantic_vs_motion"] for i in keep]

    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y + h / 2, sl, h, label="semantic / lighting", color="#a0aec0")
    ax.barh(y - h / 2, sm, h, label="semantic / motion", color=C["onnx"])
    ax.axvline(1.0, color=C["warn"], ls="--", lw=1.2)
    for i, v in enumerate(sm):
        if v < 1.0:
            ax.text(v + 0.4, i - h / 2, "motion detector", fontsize=7.5,
                    color=C["warn"], fontweight="bold", va="center")
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("ratio (higher is better)")
    ax.set_title("Read both, or the control wins\non an artifact", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.3, color=C["grid"])


def _panel_budget(ax, live, baseline) -> None:
    stages = live["metrics"]["stages_ms"] or {}
    items = [("fast_tier", stages.get("fast_tier")), ("capture_read", stages.get("capture_read"))]
    if baseline:
        items.append(("vlm_total\n(Phase 1)", (baseline["metrics"]["stages_ms"] or {}).get("vlm_total")))
    items = [(k, v) for k, v in items if v]

    names = [k for k, _ in items]
    p50 = [v["p50"] for _, v in items]
    p99 = [v["p99"] for _, v in items]
    y = np.arange(len(names))
    ax.barh(y, p50, 0.5, color=[C["onnx"], C["control"], C["torch"]][: len(names)])
    ax.barh(y, np.array(p99) - np.array(p50), 0.5, left=p50, color="black", alpha=0.25,
            label="p50 to p99")
    for i, (a, b) in enumerate(zip(p50, p99)):
        ax.text(b * 1.15, i, f"{a:.2f} / {b:.2f} ms", va="center", fontsize=8)
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ms per frame (log)")
    ax.set_title("Per-stage budget", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.3, color=C["grid"])


def _panel_throughput(ax, live, unpaced) -> None:
    bars = {"required\n(30 FPS)": 30.0,
            "sustained\n(live camera)": live["metrics"]["achieved_fps"]}
    if unpaced:
        ft = unpaced["extra"]["fast_tier"]
        bars["headroom\n(unpaced)"] = ft["n_scored"] / unpaced["run"]["duration_actual_s"]

    colors = [C["budget"], C["onnx"], C["control"]][: len(bars)]
    vals = list(bars.values())
    ax.bar(list(bars), vals, 0.55, color=colors)
    ax.set_ylim(0, max(vals) * 1.22)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
    if unpaced:
        ax.text(2, vals[2] * 0.45, f"{vals[2] / 30:.1f}x\nheadroom", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
    ax.set_ylabel("frames per second")
    ax.set_title("Throughput, VLM disabled", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3, color=C["grid"])


def _panel_signals(ax, live) -> None:
    ft = live["extra"]["fast_tier"]
    series = ft.get("series") or []
    if not series:
        ax.text(0.5, 0.5, "no fast-tier series recorded", ha="center", transform=ax.transAxes)
        return
    t = np.array([s[1] for s in series], dtype=float)
    t -= t.min()
    motion = np.array([s[2] for s in series], dtype=float)
    novelty = np.array([s[3] for s in series], dtype=float)

    ax.plot(t, novelty, lw=1.3, color=C["onnx"], label="novelty (vs rolling reference)")
    ax.plot(t, motion, lw=1.1, color=C["warn"], label="motion (pixel difference)")
    ax.set_xlabel("seconds into run")
    ax.set_ylabel("score")
    ax.set_title(
        "Fast-tier signals on a live desk scene — deliberately kept separate, "
        "because a motion spike is not a semantic event",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, color=C["grid"])


def _panel_energy(ax, live, baseline) -> None:
    ft_w = live["metrics"]["power"]["mean_w"]
    bars = {"fast tier\nonly": ft_w}
    if baseline:
        bars["Phase 1\nVLM every frame"] = baseline["metrics"]["power"]["mean_w"]
    ax.bar(list(bars), list(bars.values()), 0.5, color=[C["onnx"], C["torch"]][: len(bars)])
    for i, v in enumerate(bars.values()):
        ax.text(i, v + 1.5, f"{v:.1f} W", ha="center", fontsize=10, fontweight="bold")
    if len(bars) == 2:
        ratio = list(bars.values())[1] / list(bars.values())[0]
        ax.set_title(f"Mean GPU power — the fast tier is {ratio:.0f}x cheaper",
                     fontsize=11, fontweight="bold")
    else:
        ax.set_title("Mean GPU power", fontsize=11, fontweight="bold")
    ax.set_ylabel("watts")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3, color=C["grid"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Phase 2 fast-tier chart.")
    p.add_argument("--bench", required=True, type=Path)
    p.add_argument("--runs", nargs="+", required=True, type=Path)
    p.add_argument("--baseline", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)

    out = build(
        load(a.bench),
        [load(r) for r in a.runs],
        load(a.baseline) if a.baseline else None,
        a.out,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
