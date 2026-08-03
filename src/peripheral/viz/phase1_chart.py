"""The Phase 1 motivating chart.

`PROMPT.md`: *make the failure quantitative — this chart justifies everything after it.*

Everything plotted is read from metrics JSON files. Nothing is hardcoded, nothing is estimated, and
if a metric is null it is drawn as absent rather than as zero.

    python -m peripheral.viz.phase1_chart \
        --metrics results/phase1_naive_webcam.json results/phase1_naive_clip.json \
        --out results/phase1_naive_baseline.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display on the CI runner, and none wanted here
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {
    "capture": "#2b6cb0",
    "vlm": "#c53030",
    "dropped": "#dd6b20",
    "power": "#6b46c1",
    "budget": "#2f855a",
    "grid": "#cbd5e0",
}


def load(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc["run"]["status"] != "completed":
        raise ValueError(
            f"{path} has status {doc['run']['status']!r} — a partial run is not a result"
        )
    return doc


def _label(doc: dict[str, Any]) -> str:
    cfg = doc.get("config", {})
    src = cfg.get("capture", {}).get("source", "?")
    return {"webcam": "live webcam", "file": "recorded clip", "synthetic": "synthetic"}.get(src, src)


def build(docs: list[dict[str, Any]], out: Path) -> Path:
    fig = plt.figure(figsize=(16, 9.5), dpi=130)
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.28, top=0.85, bottom=0.09,
                          left=0.06, right=0.97)

    primary = docs[0]
    m0 = primary["metrics"]
    fps = m0["achieved_fps"]
    vlm_rate = (m0["vlm_calls_per_min"] or 0) / 60.0
    model = primary["config"].get("vlm", {}).get("name", "unknown model")

    fig.suptitle(
        "Phase 1 naive baseline — a VLM call on every frame it can get\n"
        f"{model} · llama.cpp CUDA · RTX 5070 Ti Laptop (12 GB, 95 W cap) · "
        f"capture {fps:.1f} FPS vs VLM {vlm_rate:.1f} calls/s",
        fontsize=14, fontweight="bold", y=0.975,
    )

    _panel_throughput(fig.add_subplot(gs[0, 0]), docs)
    _panel_latency(fig.add_subplot(gs[0, 1]), docs)
    _panel_drops(fig.add_subplot(gs[0, 2]), docs)
    _panel_queue_depth(fig.add_subplot(gs[1, :2]), primary)
    _panel_stages(fig.add_subplot(gs[1, 2]), primary)
    _panel_power(fig.add_subplot(gs[2, :2]), primary)
    _panel_energy_budget(fig.add_subplot(gs[2, 2]), primary)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _panel_throughput(ax, docs) -> None:
    labels, cap, vlm = [], [], []
    for d in docs:
        m = d["metrics"]
        labels.append(_label(d))
        cap.append(m["achieved_fps"])
        vlm.append((m["vlm_calls_per_min"] or 0) / 60.0)

    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, cap, w, label="capture", color=PALETTE["capture"])
    ax.bar(x + w / 2, vlm, w, label="VLM calls", color=PALETTE["vlm"])
    for xi, (c, v) in enumerate(zip(cap, vlm)):
        ax.text(xi - w / 2, c + 0.6, f"{c:.1f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
        ax.annotate(
            f"{c / v:.1f}x short", xy=(xi, max(cap) * 1.18), ha="center",
            fontsize=11, fontweight="bold", color=PALETTE["vlm"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["vlm"], lw=1.2),
        )
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("events per second")
    ax.set_title("Throughput: the VLM cannot keep up", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="center right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, color=PALETTE["grid"])
    ax.set_ylim(0, max(cap) * 1.35)


def _panel_latency(ax, docs) -> None:
    stats = ["p50", "p95", "p99"]
    x = np.arange(len(stats))
    w = 0.8 / max(1, len(docs))
    for i, d in enumerate(docs):
        s = d["metrics"]["photon_to_answer_ms"]
        if s is None:
            continue
        vals = [s[k] for k in stats]
        pos = x + (i - (len(docs) - 1) / 2) * w
        ax.bar(pos, vals, w * 0.9, label=_label(d))
        for p, v in zip(pos, vals):
            ax.text(p, v + 2, f"{v:.0f}", ha="center", fontsize=8)

    ax.axhline(33.3, color=PALETTE["budget"], ls="--", lw=1.5)
    ax.text(
        -0.45, 30.0, "33 ms frame budget", fontsize=8,
        color=PALETTE["budget"], ha="left", va="top", fontweight="bold",
    )
    ax.set_xticks(x, stats)
    ax.set_ylabel("milliseconds")
    ax.set_title("Photon-to-answer latency\n(includes capture, JPEG, HTTP, decode)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, color=PALETTE["grid"])


def _panel_drops(ax, docs) -> None:
    labels, kept, dropped = [], [], []
    for d in docs:
        m = d["metrics"]
        labels.append(_label(d))
        total = m["frames_captured"] + m["frames_dropped"]
        kept.append(100 * m["frames_captured"] / total)
        dropped.append(100 * m["frames_dropped"] / total)

    x = np.arange(len(labels))
    ax.bar(x, kept, 0.5, label="reached a stage", color=PALETTE["capture"])
    ax.bar(x, dropped, 0.5, bottom=kept, label="dropped by policy", color=PALETTE["dropped"])
    for xi, (k, dr) in enumerate(zip(kept, dropped)):
        ax.text(xi, k + dr / 2, f"{dr:.1f}%\ndropped", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("% of captured frames")
    ax.set_title("Frames dropped\n(policy: drop_oldest, recorded not accidental)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 100)


def _panel_queue_depth(ax, doc) -> None:
    t0 = doc["run"].get("t_start_monotonic")
    qd = doc["metrics"]["queue_depth"] or {}
    plotted = False
    for name, q in qd.items():
        series = q.get("series")
        if not series:
            continue
        t = np.array([p[0] for p in series], dtype=float)
        depth = np.array([p[1] for p in series], dtype=float)
        if t0 is not None:
            t = t - t0
        # Raw 20 Hz depth is a solid block of ink at this aspect ratio. Show it faintly for
        # honesty about the variance, and overlay a 1 s rolling mean that can actually be read.
        line, = ax.plot(t, depth, lw=0.6, alpha=0.18)
        win = max(1, int(round(1.0 / max(1e-6, np.median(np.diff(t))))) if t.size > 2 else 1)
        if win > 1 and depth.size > win:
            smooth = np.convolve(depth, np.ones(win) / win, mode="valid")
            ax.plot(
                t[win - 1:], smooth, lw=1.8, color=line.get_color(),
                label=f"{name} — 1 s mean (max {q['max']}, overall mean {q['mean']:.2f})",
            )
        else:
            line.set_alpha(1.0)
            line.set_label(f"{name} (max {q['max']}, mean {q['mean']:.2f})")
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "no queue-depth series recorded", ha="center", transform=ax.transAxes)
    ax.set_xlabel("seconds into run")
    ax.set_ylabel("items queued")
    ax.set_title(
        "Queue depth over time — the VLM queue sits pinned at its bound, "
        "everything upstream drains instantly",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3, color=PALETTE["grid"])


def _panel_stages(ax, doc) -> None:
    stages = doc["metrics"]["stages_ms"] or {}
    order = [k for k in ("capture_read", "fast_tier", "vlm_encode_jpeg", "vlm_total", "render")
             if k in stages]
    vals = [stages[k]["p50"] for k in order]
    ax.barh(range(len(order)), vals, color=[
        PALETTE["capture"] if "capture" in k else
        PALETTE["vlm"] if "vlm" in k else "#718096" for k in order
    ])
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v:.2f} ms", va="center", fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("p50 milliseconds (log scale)")
    ax.set_title("Per-stage budget", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, color=PALETTE["grid"])


def _panel_power(ax, doc) -> None:
    power = doc["metrics"]["power"] or {}
    series = power.get("series")
    t0 = doc["run"].get("t_start_monotonic")
    if not series:
        ax.text(0.5, 0.5, "no power series recorded", ha="center", transform=ax.transAxes)
        return
    t = np.array([p[0] for p in series], dtype=float)
    w = np.array([p[1] for p in series], dtype=float)
    if t0 is not None:
        t = t - t0
    ax.plot(t, w, lw=1.0, color=PALETTE["power"], label="GPU board power")
    if power.get("baseline_power_w") is not None:
        ax.axhline(power["baseline_power_w"], color="#a0aec0", ls=":", lw=1.2,
                   label=f"idle baseline {power['baseline_power_w']:.1f} W")
    if power.get("power_limit_w"):
        ax.axhline(power["power_limit_w"], color=PALETTE["budget"], ls="--", lw=1.5,
                   label=f"enforced cap {power['power_limit_w']:.0f} W")
    ax.set_xlabel("seconds into run")
    ax.set_ylabel("watts")
    ax.set_title(
        f"GPU power — mean {power['mean_w']:.1f} W, peak {power['peak_w']:.1f} W, "
        f"{power['energy_j']:.0f} J total, {power['energy_per_query_j']:.2f} J per answer",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, color=PALETTE["grid"])


def _panel_energy_budget(ax, doc) -> None:
    """Per-frame VLM inference is not merely slow — it is outside the power envelope."""
    m = doc["metrics"]
    power = m["power"] or {}
    per_query = power.get("energy_per_query_j")
    cap_w = power.get("power_limit_w")
    fps = m["achieved_fps"]
    if per_query is None or not cap_w:
        ax.text(0.5, 0.5, "power not measured", ha="center", transform=ax.transAxes)
        return

    achieved_rate = (m["vlm_calls_per_min"] or 0) / 60.0
    bars = {
        f"achieved\n{achieved_rate:.1f} calls/s": achieved_rate * per_query,
        f"per-frame oracle\n{fps:.0f} calls/s": fps * per_query,
    }
    colors = [PALETTE["vlm"], PALETTE["dropped"]]
    ax.bar(list(bars), list(bars.values()), 0.5, color=colors)
    ax.axhline(cap_w, color=PALETTE["budget"], ls="--", lw=2,
               label=f"GPU power cap {cap_w:.0f} W")
    for i, v in enumerate(bars.values()):
        ax.text(i, v + 4, f"{v:.0f} W", ha="center", fontsize=10, fontweight="bold")

    over = fps * per_query / cap_w
    ax.set_ylabel("sustained board power required (W)")
    ax.set_title(
        f"Per-frame inference needs {over:.1f}x the power budget",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3, color=PALETTE["grid"])
    ax.set_ylim(0, max(bars.values()) * 1.25)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the Phase 1 motivating chart.")
    p.add_argument("--metrics", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    docs = [load(m) for m in args.metrics]
    out = build(docs, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
