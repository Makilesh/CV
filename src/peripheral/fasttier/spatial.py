"""Phase 8b: spatial novelty — the fix the 8a diagnosis actually calls for.

**The failure 8a found.** On `object_events` the novelty produced by a real event (median 0.0732)
sits *below* the scene's own quiet background (p95 0.1238). No threshold can separate them, so the
Phase 4 loss to a plain timer was never a calibration problem.

**Why.** `FastTier` embeds the whole frame and pools it to one vector. A person moving through the
middle of the shot moves that vector a lot; an object appearing in one corner moves it a little.
Global pooling averages the event away — the signal is diluted by exactly the irrelevant motion the
scheduler is supposed to ignore.

**The fix.** Keep the encoder's spatial feature map instead of pooling it, give **each cell its own
rolling reference**, and score novelty as the strongest *local* change rather than the average one.
A corner object then spikes its own cell at full strength regardless of what the rest of the frame
is doing.

This costs no extra inference: the pooled vector was already computed from this feature map, so we
are declining to throw information away rather than paying for more.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .encoders import IMAGENET_MEAN, IMAGENET_STD, _ensure_cuda_dlls, _preprocess


def export_spatial_onnx(
    model_name: str, out_path: str | Path, input_size: int = 224, opset: int = 17
) -> Path:
    """Export a timm backbone that emits its **feature map**, not a pooled vector."""
    import timm
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    # features_only gives the pyramid; we take the last (deepest, most semantic) stage.
    model = timm.create_model(model_name, pretrained=True, features_only=True).eval()

    class LastStage(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[-1]

    wrapped = LastStage(model).eval()
    dummy = torch.zeros(1, 3, input_size, input_size)
    with torch.inference_mode():
        shape = tuple(wrapped(dummy).shape)
    torch.onnx.export(
        wrapped, dummy, str(out_path),
        input_names=["input"], output_names=["features"],
        opset_version=opset, dynamo=False,
    )
    print(f"exported {out_path.name} with feature map {shape}")
    return out_path


class SpatialEncoder:
    """Returns an L2-normalised feature map, one unit vector per spatial cell."""

    def __init__(
        self,
        onnx_path: str | Path,
        input_size: int = 224,
        providers: Sequence[str] | None = None,
        allow_cpu_fallback: bool = False,
    ) -> None:
        _ensure_cuda_dlls()
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"spatial ONNX model not found: {self.onnx_path}")
        self.input_size = int(input_size)
        self._mean, self._std = IMAGENET_MEAN, IMAGENET_STD

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        wanted = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        available = ort.get_available_providers()
        use = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=use)
        self.providers = self.session.get_providers()

        # Same rule as OnnxEncoder: a silent CPU fallback would report a CPU number as a GPU one.
        if not allow_cpu_fallback:
            asked = [p for p in wanted if p != "CPUExecutionProvider"]
            if asked and not any(p in self.providers for p in asked):
                raise RuntimeError(
                    f"ONNX Runtime fell back to CPU: asked {asked}, bound {self.providers}"
                )

        self._in = self.session.get_inputs()[0].name
        self._out = self.session.get_outputs()[0].name
        probe = self.encode_spatial(np.zeros((64, 64, 3), np.uint8))
        self.grid = probe.shape[:2]
        self.dim = probe.shape[2]
        self.name = f"{self.onnx_path.stem}-spatial{self.grid[0]}x{self.grid[1]}"

    def encode_spatial(self, bgr: np.ndarray) -> np.ndarray:
        """(H, W, C) with every cell L2-normalised."""
        arr = _preprocess(bgr, self.input_size, self._mean, self._std)
        feat = self.session.run([self._out], {self._in: arr})[0][0]   # (C, H, W)
        feat = np.transpose(feat, (1, 2, 0)).astype(np.float32)        # (H, W, C)
        norms = np.linalg.norm(feat, axis=-1, keepdims=True)
        return feat / np.maximum(norms, 1e-12)

    def encode(self, bgr: np.ndarray) -> np.ndarray:
        """Pooled vector, so this can stand in for a plain Encoder where one is expected."""
        spatial = self.encode_spatial(bgr)
        v = spatial.reshape(-1, spatial.shape[-1]).mean(axis=0)
        n = np.linalg.norm(v)
        return (v / n if n > 0 else v).astype(np.float32)

    def warmup(self, n: int = 10, size: tuple[int, int] = (480, 640)) -> None:
        dummy = np.random.default_rng(0).integers(0, 255, (*size, 3), dtype=np.uint8)
        for _ in range(n):
            self.encode_spatial(dummy)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "backend": "onnx-spatial", "path": str(self.onnx_path),
            "input_size": self.input_size, "grid": list(self.grid), "dim": self.dim,
            "providers": list(self.providers),
        }


class PatchRollingReference:
    """One exponential reference **per spatial cell**, with a half-life in seconds.

    The pooled version answers "how different does the scene look overall". This answers "how
    different does the *most changed part* of the scene look", which is the question a scheduler
    watching for localised events actually needs.
    """

    def __init__(self, half_life_s: float = 2.0, min_updates: int = 5, top_k: int = 3) -> None:
        self.half_life_s = float(half_life_s)
        self.min_updates = int(min_updates)
        self.top_k = int(top_k)
        self._ref: np.ndarray | None = None
        self._t_last: float | None = None
        self.n_updates = 0

    def reset(self) -> None:
        self._ref = None
        self._t_last = None
        self.n_updates = 0

    def update(self, spatial: np.ndarray, t: float) -> None:
        if self._ref is None:
            self._ref = spatial.astype(np.float32).copy()
            self._t_last = t
            self.n_updates = 1
            return
        dt = max(0.0, t - (self._t_last or t))
        alpha = 1.0 - 0.5 ** (dt / self.half_life_s) if self.half_life_s > 0 else 1.0
        self._ref = (1.0 - alpha) * self._ref + alpha * spatial
        norms = np.linalg.norm(self._ref, axis=-1, keepdims=True)
        self._ref = self._ref / np.maximum(norms, 1e-12)
        self._t_last = t
        self.n_updates += 1

    def distance(self, spatial: np.ndarray) -> float:
        """Mean cosine distance over the `top_k` most-changed cells.

        Top-k rather than the single max: one cell can spike on sensor noise, but a real object
        covers several. k=3 of 49 cells is still local enough that a corner event is not averaged
        away, while being robust to a single noisy cell.
        """
        if self._ref is None or self.n_updates < self.min_updates:
            return 0.0
        per_cell = 1.0 - np.sum(self._ref * spatial, axis=-1)   # (H, W)
        flat = np.sort(per_cell.reshape(-1))[::-1]
        k = max(1, min(self.top_k, flat.size))
        return float(np.clip(flat[:k].mean(), 0.0, 2.0))

    def per_cell_distance(self, spatial: np.ndarray) -> np.ndarray:
        if self._ref is None:
            return np.zeros(spatial.shape[:2], np.float32)
        return (1.0 - np.sum(self._ref * spatial, axis=-1)).astype(np.float32)

    @property
    def ready(self) -> bool:
        return self._ref is not None and self.n_updates >= self.min_updates


class SpatialFastTier:
    """Fast tier producing **patch novelty** alongside the existing signals.

    Drop-in for `FastTier`: same `process(frame) -> FrameScores` contract, with `novelty` now the
    top-k local score. The pooled global novelty is kept in `meta` so the two can be compared on
    the same run rather than across runs.
    """

    def __init__(
        self,
        encoder: SpatialEncoder,
        motion_size: int = 64,
        half_life_s: float = 2.0,
        top_k: int = 3,
        keep_embedding: bool = False,
    ) -> None:
        from .scoring import MotionScorer, RollingReference

        self.encoder = encoder
        self.motion = MotionScorer(size=motion_size)
        self.reference = PatchRollingReference(half_life_s=half_life_s, top_k=top_k)
        self.global_reference = RollingReference(half_life_s=half_life_s)
        self.keep_embedding = keep_embedding
        self._prev_pooled: np.ndarray | None = None

    def reset(self) -> None:
        self.motion.reset()
        self.reference.reset()
        self.global_reference.reset()
        self._prev_pooled = None

    def process(self, frame: Any) -> Any:
        from .scoring import FrameScores

        motion = self.motion.score(frame.image)
        spatial = self.encoder.encode_spatial(frame.image)

        pooled = spatial.reshape(-1, spatial.shape[-1]).mean(axis=0)
        n = np.linalg.norm(pooled)
        pooled = (pooled / n if n > 0 else pooled).astype(np.float32)

        novelty = self.reference.distance(spatial)
        global_novelty = self.global_reference.distance(pooled)
        scene_change = (
            float(1.0 - np.dot(self._prev_pooled, pooled))
            if self._prev_pooled is not None else 0.0
        )

        # Stream time, not capture time — see Frame.t_presentation.
        self.reference.update(spatial, frame.t_stream)
        self.global_reference.update(pooled, frame.t_stream)
        self._prev_pooled = pooled

        return FrameScores(
            frame_id=frame.frame_id,
            motion=motion,
            novelty=novelty,
            scene_change=scene_change,
            embedding=pooled if self.keep_embedding else None,
            meta={"global_novelty": global_novelty},
        )
