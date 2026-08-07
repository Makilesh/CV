"""Markdown tables generated from metrics JSON.

`RESULTS.md` is a written report, but its tables are generated rather than typed, so the numbers
in the prose cannot drift away from the runs that produced them. Phase 6 reuses this.

    python -m peripheral.viz.tables --bench results/phase3_vlm_bench.json \
        --kv results/phase3_kv_reuse.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TARGET_TTFT_MS = 400.0


def load(p: Path) -> dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def vlm_bench_table(doc: dict) -> str:
    rows = [r for r in doc["extra"]["vlm_bench"]["configs"] if "error" not in r]
    rows.sort(key=lambda r: (r["family"], r["ttft_p95_ms"]))

    out = [
        _row(["model", "quant", "file", "peak VRAM", "load", "TTFT p50", "**TTFT p95**",
              "tok/s", "fidelity F1", "self-consist."]),
        _row(["---"] * 10),
    ]
    for r in rows:
        q = r.get("quality_vs_family_reference") or {}
        is_ref = r["label"] == r.get("family_reference")
        f1 = q.get("content_f1_mean")
        f1s = "— *(reference)*" if is_ref else (f"{f1:.3f}" if f1 is not None else "—")
        p95 = r["ttft_p95_ms"]
        mark = "" if p95 <= TARGET_TTFT_MS else " ❌"
        out.append(_row([
            r["family"],
            r["quant"],
            f"{r['model_file_gb']:.2f} GB",
            f"{r['peak_vram_gb']:.2f} GB",
            f"{r['load_time_s']:.1f} s",
            f"{r['ttft_p50_ms']:.0f} ms",
            f"**{p95:.0f} ms**{mark}",
            f"{r['tokens_per_s']:.0f}",
            f1s,
            f"{r['self_consistency']:.2f}" if r.get("self_consistency") is not None else "—",
        ]))
    return "\n".join(out)


def kv_reuse_table(doc: dict) -> str:
    arms = [a for a in doc["extra"]["kv_reuse"]["arms"] if "error" not in a]
    out = [
        _row(["arm", "prompt order", "streaming", "TTFT p50", "TTFT p95",
              "complete answer p50", "vs arm A"]),
        _row(["---"] * 7),
    ]
    for a in arms:
        ttft = a["ttft_p50_ms"]
        speed = a.get("ttft_speedup_vs_A")
        out.append(_row([
            a["label"],
            "image first" if a["image_first"] else "text first",
            "yes" if a["stream"] else "no",
            f"{ttft:.0f} ms" if ttft is not None else "*n/a — no first token*",
            f"{a['ttft_p95_ms']:.0f} ms" if a["ttft_p95_ms"] is not None else "*n/a*",
            f"{a['total_p50_ms']:.0f} ms",
            f"{speed:.2f}x" if speed else "—",
        ]))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Emit RESULTS.md tables from metrics JSON.")
    p.add_argument("--bench", type=Path)
    p.add_argument("--kv", type=Path)
    a = p.parse_args(argv)

    if a.bench and a.bench.exists():
        print("### Model x quantization sweep\n")
        print(vlm_bench_table(load(a.bench)))
        print()
    if a.kv and a.kv.exists():
        print("### KV-cache reuse and streaming decode\n")
        print(kv_reuse_table(load(a.kv)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
