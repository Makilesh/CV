"""Phase 3 slow-tier sweep: small VLMs x GGUF quantization levels.

For each configuration: **load time, peak VRAM, TTFT, tokens/sec, and answer quality on frames from
a real clip.** One `step()` per configuration, so the bounded-runner contract still holds and a
sweep that runs out of time records what it got rather than losing everything.

Two things this deliberately does:

* **TTFT is measured from the frame in hand, not from the request.** JPEG encode, base64, HTTP and
  SSE framing are all inside the number, because they are all between the camera and the answer.
  The Phase 1 finding stands: there is a ~15 ms llama-server scheduling floor underneath all of it.
* **The prompt is fixed and the frames come from a clip**, so differences across the table are the
  model and the quantization, not the question.

    python -m peripheral.cli.vlm_bench --duration 5400 --headless \
        --metrics-out results/phase3_vlm_bench.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..capture import FileSource
from ..runtime import BoundedRunner
from ..telemetry.clock import now
from ..vlm.llama_server import LlamaServerClient
from ..vlm.quality import noise_floor, score_against_reference, self_consistency
from ._args import parse_and_load


class VlmBenchRunner(BoundedRunner):
    name = "phase3_vlm_bench"

    def setup(self) -> None:
        b = self.cfg.vlm_bench
        _require_idle_gpu(self.power, float(b.get("max_foreign_vram_gb", 1.5)))
        clip = Path(b.clip)
        if not clip.exists():
            raise FileNotFoundError(f"benchmark clip not found: {clip}")

        # Unpaced: this is an offline benchmark, not a streaming run.
        src = FileSource(clip, realtime=False)
        src.open()
        frames, i = [], 0
        while len(frames) < int(b.n_frames):
            f = src.read()
            if f is None:
                break
            if i % int(b.frame_stride) == 0:
                frames.append(f.image)
            i += 1
        src.close()
        if len(frames) < 4:
            raise RuntimeError(f"clip yielded only {len(frames)} frames")
        self.frames = frames
        self.prompt = str(b.prompt)
        self.max_tokens = int(b.max_tokens)
        self.configs = [dict(c) for c in b.configs]
        self.results: list[dict[str, Any]] = []
        self._i = 0
        self.recorder.note(
            f"quality set: {len(frames)} frames from {clip}, prompt={self.prompt!r}"
        )

    def step(self) -> bool:
        if self._i >= len(self.configs):
            return False
        spec = self.configs[self._i]
        self._i += 1
        label = spec["label"]

        model, mmproj = Path(spec["model"]), Path(spec["mmproj"])
        if not model.exists() or not mmproj.exists():
            self.recorder.note(f"config {label!r} skipped: weights not present")
            self.results.append({"label": label, "error": "weights not present"})
            return True

        try:
            row = self._bench_one(spec)
        except Exception as exc:  # noqa: BLE001 - one bad config must not kill the sweep
            self.recorder.note(f"config {label!r} failed: {type(exc).__name__}: {exc}")
            self.results.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
            return True

        self.results.append(row)
        print(
            f"{label:34s} load {row['load_time_s']:5.1f}s  VRAM {row['peak_vram_gb']:5.2f}GB  "
            f"TTFT p50 {row['ttft_p50_ms']:6.1f} p95 {row['ttft_p95_ms']:6.1f}  "
            f"{row['tokens_per_s']:5.1f} tok/s",
            file=sys.stderr,
        )
        return True

    def _bench_one(self, spec: dict[str, Any]) -> dict[str, Any]:
        v = self.cfg.vlm
        vram_before = _vram_used_gb(self.power)

        client = LlamaServerClient(
            binary=v.binary,
            model=spec["model"],
            mmproj=spec["mmproj"],
            host=v.host,
            port=int(spec.get("port", v.port)),
            n_gpu_layers=int(spec.get("n_gpu_layers", v.n_gpu_layers)),
            ctx_size=int(spec.get("ctx_size", v.ctx_size)),
            jpeg_quality=int(v.jpeg_quality),
            autostart=True,
            startup_timeout_s=float(v.startup_timeout_s),
            extra_args=list(spec.get("extra_args", []) or []),
            model_name=spec["label"],
        )
        client.start()
        try:
            vram_after_load = _vram_used_gb(self.power)

            # Warmup: the first call carries graph construction and allocator growth that would
            # otherwise land in the p95 and be reported as steady-state TTFT.
            for _ in range(int(self.cfg.vlm_bench.warmup)):
                client.describe(self.frames[0], self.prompt, max_tokens=self.max_tokens)

            ttft, total, tok_s, answers = [], [], [], []
            peak_vram = max(vram_before, vram_after_load)
            for img in self.frames:
                t0 = now()  # frame in hand: JPEG encode and transport are inside the number
                r = client.describe(img, self.prompt, max_tokens=self.max_tokens)
                wall = (now() - t0) * 1000.0
                if r.ttft_ms is not None:
                    # r.ttft_ms starts at request send; add our encode cost to get frame->token.
                    ttft.append(r.ttft_ms + r.encode_ms)
                total.append(wall)
                if r.n_tokens and r.total_ms > 0:
                    tok_s.append(r.n_tokens / (r.total_ms / 1000.0))
                answers.append(r.text)
                peak_vram = max(peak_vram, _vram_used_gb(self.power))

            # Second full pass over the SAME frames. This is not redundancy — it establishes this
            # configuration's noise floor, without which the fidelity column cannot be read (see
            # peripheral.vlm.quality.noise_floor).
            repeat = [
                client.describe(img, self.prompt, max_tokens=self.max_tokens).text
                for img in self.frames
            ]
        finally:
            client.stop()

        a = np.asarray(ttft) if ttft else np.array([np.nan])
        return {
            "label": spec["label"],
            "family": spec.get("family"),
            "quant": spec.get("quant"),
            "params": spec.get("params"),
            "model_file_gb": round(Path(spec["model"]).stat().st_size / 1024**3, 2),
            "mmproj_file_gb": round(Path(spec["mmproj"]).stat().st_size / 1024**3, 2),
            "load_time_s": round(client.load_time_s or 0.0, 2),
            "peak_vram_gb": round(peak_vram, 2),
            "vram_delta_gb": round(vram_after_load - vram_before, 2),
            "ttft_p50_ms": round(float(np.percentile(a, 50)), 1),
            "ttft_p95_ms": round(float(np.percentile(a, 95)), 1),
            "ttft_p99_ms": round(float(np.percentile(a, 99)), 1),
            "ttft_min_ms": round(float(a.min()), 1),
            "total_p50_ms": round(float(np.percentile(np.asarray(total), 50)), 1),
            "tokens_per_s": round(float(np.mean(tok_s)), 1) if tok_s else 0.0,
            "n_frames": len(self.frames),
            "answers": answers,
            "self_consistency": self_consistency(answers[: len(repeat)], repeat),
            "noise_floor": noise_floor(answers[: len(repeat)], repeat),
            "sample_answer": answers[0][:200] if answers else "",
        }

    def teardown(self) -> None:
        # Fidelity is scored against the highest-precision variant of the SAME family, so the
        # number isolates quantization damage rather than mixing in cross-model differences.
        ok = [r for r in self.results if "error" not in r]
        by_family: dict[str, list[dict]] = {}
        for r in ok:
            by_family.setdefault(r.get("family") or r["label"], []).append(r)

        precision_rank = {"f16": 0, "F16": 0, "Q8_0": 1, "Q5_K_M": 2, "Q4_K_M": 3}
        for family, rows in by_family.items():
            rows.sort(key=lambda z: precision_rank.get(z.get("quant"), 99))
            ref = rows[0]
            for r in rows:
                r["quality_vs_family_reference"] = score_against_reference(
                    r["answers"], ref["answers"]
                )
                r["family_reference"] = ref["label"]

        for r in ok:
            r.pop("answers", None)

        self.recorder.record_extra(
            "vlm_bench",
            {
                "configs": self.results,
                "prompt": self.prompt,
                "n_frames": len(self.frames),
                "quality_definition": (
                    "Fidelity to the highest-precision variant of the SAME model family on the "
                    "same frames. Measures quantization damage, NOT correctness — there are no "
                    "ground-truth annotations for these clips until Phase 4."
                ),
                "ttft_definition": (
                    "Frame in hand -> first generated token. Includes JPEG encode, base64, HTTP, "
                    "SSE framing, vision encoding and prefill. Excludes camera capture, which "
                    "Phase 1 measured at 32.09 ms p50 and which photon-to-first-token adds."
                ),
            },
        )


def _vram_used_gb(sampler: Any) -> float:
    """Board-wide NVML VRAM. Board-wide is what the 12 GB limit actually is."""
    s = sampler._read() if getattr(sampler, "available", False) else None
    return (s.mem_used_b / 1024**3) if s else 0.0


def _require_idle_gpu(sampler: Any, max_foreign_gb: float) -> None:
    """Refuse to benchmark on a GPU somebody else is already using.

    Learned the hard way on 2026-08-08: an unrelated process was holding ~9.5 GB, and the whole
    sweep completed successfully with plausible-looking numbers that were entirely wrong — peak
    VRAM read ~11.8 GB for every configuration including a 500M model that actually needs 1.5 GB,
    and TTFT was ~3x its true value from memory pressure. Nothing failed; the results were just
    silently invalid, which is the most dangerous kind of wrong.

    VRAM here is board-wide because that is what the 12 GB limit actually is, so contention is
    indistinguishable from our own usage after the fact. It has to be caught before the run.
    """
    if not getattr(sampler, "available", False):
        return
    used = _vram_used_gb(sampler)
    if used <= max_foreign_gb:
        return

    detail = ""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.stdout.strip():
            detail = "\nProcesses currently on the GPU:\n  " + "\n  ".join(
                out.stdout.strip().splitlines()
            )
    except Exception:  # noqa: BLE001 - diagnostics are best-effort
        pass

    raise RuntimeError(
        f"GPU already has {used:.2f} GB in use (limit for a clean run: {max_foreign_gb:.2f} GB). "
        "Benchmarking against a contended GPU produces plausible numbers that are wrong: peak "
        "VRAM is board-wide so it absorbs the other process, and memory pressure inflates TTFT. "
        "Free the GPU and re-run, or raise vlm_bench.max_foreign_vram_gb to override deliberately."
        + detail
    )


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.vlm_bench",
        description="Phase 3 sweep: small VLMs x GGUF quantization.",
        argv=argv,
    )
    runner = VlmBenchRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    code = runner.run()
    print(f"{len(runner.results)} configurations benchmarked", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
