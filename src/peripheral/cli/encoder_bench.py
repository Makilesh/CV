"""Phase 2 encoder sweep: latency and discrimination quality, per candidate.

`PROMPT.md`: *if the encoder is the bottleneck, benchmark smaller options before accepting it —
don't accept a 15 ms encoder without telling me what a 5 ms one costs in quality.* This is the run
that produces that answer.

**The quality measure is chosen to match what the scheduler actually needs.** A generic embedding
benchmark (ImageNet accuracy, retrieval mAP) would say nothing about our failure mode. Phase 4's
headline risk is a false trigger on lighting drift carrying no semantic event, so each encoder is
scored on how far its embedding moves under three kinds of change:

* **lighting** — gamma and brightness shifts of the same frame. Should move the embedding *little*.
* **motion**   — consecutive real frames. Small movement, no semantic event. Should move it little.
* **semantic** — a synthetic object occluding ~12% of the frame. Should move it *a lot*.

The headline number is the **discrimination ratio** `d_semantic / d_lighting`: how many times
further a real event pushes the embedding than a lighting change does. A pure motion detector
scores near 1 and is exactly the thing we must beat.

    python -m peripheral.cli.encoder_bench --duration 900 --headless \
        --metrics-out results/phase2_encoder_bench.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..capture import FileSource
from ..fasttier.encoders import (
    DownsampleEncoder,
    Encoder,
    OnnxEncoder,
    TorchEncoder,
    export_onnx,
)
from ..fasttier.scoring import cosine_distance
from ..runtime import BoundedRunner
from ..telemetry.clock import now
from ._args import parse_and_load


# --- perturbations ---------------------------------------------------------------------------
def gamma_shift(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def brightness_shift(img: np.ndarray, delta: int) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=1.0, beta=delta)


def insert_object(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A synthetic semantic event: an opaque textured object appears, covering ~12% of the frame.

    Synthetic rather than staged so it is reproducible and seeded. It is a deliberately *easy*
    semantic event — if an encoder cannot separate this from a gamma shift, it certainly cannot
    separate a subtle one.
    """
    out = img.copy()
    h, w = out.shape[:2]
    bw, bh = int(w * 0.38), int(h * 0.32)
    x = int(rng.integers(0, max(1, w - bw)))
    y = int(rng.integers(0, max(1, h - bh)))
    patch = rng.integers(0, 255, (bh, bw, 3), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (9, 9), 0)
    cv2.rectangle(patch, (0, 0), (bw - 1, bh - 1), (20, 20, 20), 6)
    out[y:y + bh, x:x + bw] = patch
    return out


# --- candidates ------------------------------------------------------------------------------
def candidates(cfg: Any) -> list[dict[str, Any]]:
    """The sweep. Each entry becomes one `step()` of the bounded run."""
    return [dict(c) for c in cfg.encoder_bench.candidates]


def build(spec: dict[str, Any], models_dir: Path) -> Encoder:
    kind = spec["kind"]
    if kind == "downsample":
        return DownsampleEncoder(size=spec.get("size", 32))
    if kind == "torch":
        return TorchEncoder(
            model_name=spec["model"],
            input_size=spec.get("input_size", 224),
            dtype=spec.get("dtype", "fp16"),
            clip_norm=spec.get("clip_norm", False),
        )
    if kind == "onnx":
        path = models_dir / "onnx" / f"{spec['model'].replace('.', '_')}_{spec.get('input_size', 224)}.onnx"
        export_onnx(spec["model"], path, input_size=spec.get("input_size", 224))
        return OnnxEncoder(
            path,
            input_size=spec.get("input_size", 224),
            providers=spec.get("providers"),
            clip_norm=spec.get("clip_norm", False),
            label=f"{spec['model']}@{spec.get('input_size', 224)}-onnx",
            allow_cpu_fallback=spec.get("allow_cpu_fallback", False),
        )
    raise ValueError(f"unknown candidate kind {kind!r}")


class EncoderBenchRunner(BoundedRunner):
    name = "phase2_encoder_bench"

    def setup(self) -> None:
        self.models_dir = Path(self.cfg.paths.models_dir)
        self.rng = np.random.default_rng(self.seed)
        self.specs = candidates(self.cfg)
        self.results: list[dict[str, Any]] = []
        self._i = 0

        clip = Path(self.cfg.encoder_bench.clip)
        if not clip.exists():
            raise FileNotFoundError(
                f"benchmark clip not found: {clip} — record one with peripheral.cli.record_clip"
            )
        # Load frames once, unpaced: this is an offline benchmark, not a streaming run, and
        # wall-clock pacing here would only measure the clip's frame rate.
        src = FileSource(clip, realtime=False)
        src.open()
        stride = int(self.cfg.encoder_bench.frame_stride)
        want = int(self.cfg.encoder_bench.n_frames)
        frames, i = [], 0
        while len(frames) < want:
            f = src.read()
            if f is None:
                break
            if i % stride == 0:
                frames.append(f.image)
            i += 1
        src.close()
        if len(frames) < 4:
            raise RuntimeError(f"clip yielded only {len(frames)} frames")
        self.frames = frames
        self.recorder.note(
            f"quality set: {len(frames)} frames from {clip} (stride {stride})"
        )

    def step(self) -> bool:
        if self._i >= len(self.specs):
            return False
        spec = self.specs[self._i]
        self._i += 1
        label = spec.get("label", f"{spec.get('model', spec['kind'])}")
        try:
            enc = build(spec, self.models_dir)
        except Exception as exc:  # noqa: BLE001 - one unavailable candidate must not kill the sweep
            self.recorder.note(f"candidate {label!r} unavailable: {type(exc).__name__}: {exc}")
            self.results.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
            return True

        enc.warmup(n=int(self.cfg.encoder_bench.warmup))
        latency = self._latency(enc)
        quality = self._quality(enc)
        row = {"label": label, **enc.describe(), **latency, **quality}
        self.results.append(row)
        self.recorder.record_stage(f"encode/{label}", latency["p50_ms"])
        print(
            f"{label:44s} p50 {latency['p50_ms']:6.2f} ms  p95 {latency['p95_ms']:6.2f}  "
            f"disc {quality['discrimination']:5.2f}x",
            file=sys.stderr,
        )
        return True

    def _latency(self, enc: Encoder) -> dict[str, Any]:
        """Per-frame latency including preprocessing. Never model-only."""
        reps = int(self.cfg.encoder_bench.latency_reps)
        samples = []
        for i in range(reps):
            img = self.frames[i % len(self.frames)]
            t0 = now()
            enc.encode(img)
            samples.append((now() - t0) * 1000.0)
        a = np.asarray(samples)
        return {
            "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "mean_ms": round(float(a.mean()), 3),
            "max_fps": round(1000.0 / float(np.percentile(a, 50)), 1),
            "n_reps": reps,
        }

    def _quality(self, enc: Encoder) -> dict[str, Any]:
        """How far does the embedding move under lighting, motion and semantic change?"""
        rng = np.random.default_rng(self.seed)
        d_light, d_motion, d_semantic = [], [], []

        for i in range(len(self.frames) - 1):
            base = self.frames[i]
            e0 = enc.encode(base)

            for pert in (
                gamma_shift(base, 0.6),
                gamma_shift(base, 1.6),
                brightness_shift(base, 40),
                brightness_shift(base, -40),
            ):
                d_light.append(cosine_distance(e0, enc.encode(pert)))

            d_motion.append(cosine_distance(e0, enc.encode(self.frames[i + 1])))
            d_semantic.append(cosine_distance(e0, enc.encode(insert_object(base, rng))))

        med_l = float(np.median(d_light))
        med_m = float(np.median(d_motion))
        med_s = float(np.median(d_semantic))
        return {
            "d_lighting_median": round(med_l, 5),
            "d_motion_median": round(med_m, 5),
            "d_semantic_median": round(med_s, 5),
            # The number that matters: how much further a real event moves the embedding than a
            # lighting change does. A motion detector scores ~1.
            "discrimination": round(med_s / med_l, 3) if med_l > 1e-9 else float("inf"),
            "semantic_vs_motion": round(med_s / med_m, 3) if med_m > 1e-9 else float("inf"),
            "n_pairs": len(d_semantic),
        }

    def teardown(self) -> None:
        ok = [r for r in self.results if "error" not in r]
        self.recorder.record_extra(
            "encoder_bench",
            {
                "candidates": self.results,
                "quality_definition": {
                    "lighting": "gamma 0.6/1.6 and brightness +/-40 on the same frame",
                    "motion": "consecutive frames from the clip",
                    "semantic": "synthetic opaque object occluding ~12% of the frame",
                    "discrimination": "d_semantic_median / d_lighting_median; higher is better, "
                                      "a pure motion detector scores ~1",
                },
                "n_candidates": len(self.results),
                "n_failed": len(self.results) - len(ok),
            },
        )


def main(argv: list[str] | None = None) -> int:
    args, cfg = parse_and_load(
        prog="peripheral.cli.encoder_bench",
        description="Phase 2 encoder sweep: latency and lighting-vs-semantic discrimination.",
        argv=argv,
    )
    runner = EncoderBenchRunner(
        cfg=cfg,
        duration_s=cfg.run.duration_s,
        metrics_out=cfg.run.metrics_out,
        headless=cfg.run.headless,
        seed=cfg.seed,
    )
    code = runner.run()
    print(json.dumps(runner.results, indent=2)[:400], file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
