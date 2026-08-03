"""Small embedding encoders for the fast tier.

The fast tier runs on **every** frame inside a ~10 ms budget (Phase 1 measured `vlm_total` at
97.66 ms p50 and the rest of the pipeline at ~1 ms, so this is the room available). Its job is to
produce a cheap embedding that a scheduler can compare against a rolling reference.

Every encoder here reports latency **including preprocessing** — resize, colour conversion and
normalisation are our cost, and a model-only number is a marketing number.

Backends:

* ``torch``  — timm model on CUDA, fp16 or fp32.
* ``onnx``   — the same graph exported and run through ONNX Runtime (CUDA EP). `PROMPT.md` asks us
  to prefer ONNX Runtime here and to measure both, so both exist and neither is assumed faster.

`DownsampleEncoder` is the deliberate control: 32x32 grayscale, no network at all. It costs
essentially nothing and represents what plain frame differencing can see. If a learned encoder
cannot beat it on the lighting-vs-semantic discrimination that the scheduler actually needs, the
learned encoder has not earned its milliseconds.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

_ORT_DLLS_READY = False


def _ensure_cuda_dlls() -> str | None:
    """Point ONNX Runtime at CUDA DLLs before it is imported. Windows-specific, and load-bearing.

    Measured 2026-08-04: the PyPI `onnxruntime-gpu` 1.28 wheel is built against CUDA 13 and fails
    to load its CUDA provider here (`cublasLt64_13.dll` missing), silently falling back to CPU —
    the session still reports success, so a CPU number can masquerade as a GPU one. The CUDA 13
    runtime wheels NVIDIA publishes are Linux-only, so there is no pip route to satisfy 1.28.

    We pin `onnxruntime-gpu==1.26.0` (CUDA 12) and reuse the CUDA 12.8 + cuDNN 9 DLLs that torch
    already ships, which are known-good on this machine. Returns the directory added, or None.
    """
    global _ORT_DLLS_READY
    if _ORT_DLLS_READY:
        return None
    _ORT_DLLS_READY = True
    if not hasattr(os, "add_dll_directory"):
        return None
    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
            return str(lib)
    except Exception:  # noqa: BLE001 - absence of torch is not fatal for a CPU-only run
        return None
    return None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CLIP_MEAN = np.array([0.4815, 0.4578, 0.4082], dtype=np.float32)
CLIP_STD = np.array([0.2686, 0.2613, 0.2758], dtype=np.float32)


class Encoder(ABC):
    """Maps a BGR uint8 frame to an L2-normalised embedding."""

    name: str = "encoder"
    dim: int = 0
    input_size: int = 224

    @abstractmethod
    def encode(self, bgr: np.ndarray) -> np.ndarray:
        """Return a float32 unit vector. Includes all preprocessing."""

    def encode_batch(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self.encode(f) for f in frames])

    def warmup(self, n: int = 10, size: tuple[int, int] = (480, 640)) -> None:
        """Burn in lazy CUDA kernels and allocator paths so they don't land in the p99."""
        dummy = np.random.default_rng(0).integers(0, 255, (*size, 3), dtype=np.uint8)
        for _ in range(n):
            self.encode(dummy)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "dim": self.dim, "input_size": self.input_size}


def _preprocess(
    bgr: np.ndarray, size: int, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """BGR uint8 HxWx3 -> normalised NCHW float32. cv2 rather than torchvision: measurably faster
    on CPU and it keeps the tensor library out of the per-frame hot path."""
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


class DownsampleEncoder(Encoder):
    """The control: 32x32 grayscale, flattened, mean-centred and L2-normalised.

    No network, no GPU. This is what frame differencing sees, expressed as an embedding so it can
    be scored with exactly the same machinery as the learned encoders.
    """

    def __init__(self, size: int = 32) -> None:
        self.name = f"downsample{size}"
        self.input_size = size
        self.dim = size * size

    def encode(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
        gray -= gray.mean()
        n = np.linalg.norm(gray)
        return gray / n if n > 0 else gray


class TorchEncoder(Encoder):
    """timm model on CUDA."""

    def __init__(
        self,
        model_name: str,
        input_size: int = 224,
        device: str = "cuda",
        dtype: str = "fp16",
        clip_norm: bool = False,
    ) -> None:
        import timm
        import torch

        self._torch = torch
        self.name = f"{model_name}@{input_size}-{dtype}-torch"
        self.model_name = model_name
        self.input_size = int(input_size)
        self.device = device
        self.dtype_str = dtype
        self._dtype = torch.float16 if dtype == "fp16" else torch.float32
        self._mean = CLIP_MEAN if clip_norm else IMAGENET_MEAN
        self._std = CLIP_STD if clip_norm else IMAGENET_STD

        kwargs: dict[str, Any] = {"pretrained": True, "num_classes": 0}
        if "vit" in model_name and "dinov2" in model_name:
            # DINOv2's default 518px input means 1369 patches — far outside a 10 ms budget.
            # timm interpolates the position embedding for a smaller input.
            kwargs["img_size"] = self.input_size
        self.model = timm.create_model(model_name, **kwargs)
        self.model.eval().to(device=device, dtype=self._dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

        with torch.inference_mode():
            probe = torch.zeros(1, 3, self.input_size, self.input_size,
                                device=device, dtype=self._dtype)
            self.dim = int(self.model(probe).shape[-1])

    def encode(self, bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        arr = _preprocess(bgr, self.input_size, self._mean, self._std)
        with torch.inference_mode():
            t = torch.from_numpy(arr).to(self.device, dtype=self._dtype, non_blocking=True)
            out = self.model(t)
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return out.float().cpu().numpy()[0]

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "backend": "torch",
            "model": self.model_name,
            "dtype": self.dtype_str,
            "device": self.device,
            "params_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 2),
        }


class OnnxEncoder(Encoder):
    """The same graph through ONNX Runtime."""

    def __init__(
        self,
        onnx_path: str | Path,
        input_size: int = 224,
        providers: Sequence[str] | None = None,
        clip_norm: bool = False,
        label: str | None = None,
        allow_cpu_fallback: bool = False,
    ) -> None:
        _ensure_cuda_dlls()
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        self.input_size = int(input_size)
        self._mean = CLIP_MEAN if clip_norm else IMAGENET_MEAN
        self._std = CLIP_STD if clip_norm else IMAGENET_STD

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        wanted = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        available = ort.get_available_providers()
        use = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=use)
        self.providers = self.session.get_providers()

        self._in = self.session.get_inputs()[0].name
        self._out = self.session.get_outputs()[0].name
        self.dim = int(self.session.get_outputs()[0].shape[-1])
        self.name = label or f"{self.onnx_path.stem}@{input_size}-onnx"

    def encode(self, bgr: np.ndarray) -> np.ndarray:
        arr = _preprocess(bgr, self.input_size, self._mean, self._std)
        out = self.session.run([self._out], {self._in: arr})[0][0]
        n = np.linalg.norm(out)
        return (out / n if n > 0 else out).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "backend": "onnx",
            "path": str(self.onnx_path),
            "providers": list(self.providers),
        }


def export_onnx(
    model_name: str, out_path: str | Path, input_size: int = 224, opset: int = 17
) -> Path:
    """Export a timm model to ONNX at fixed batch 1.

    Batch is fixed because the fast tier is strictly per-frame — a dynamic batch axis costs
    optimisation opportunities for a flexibility we never use.
    """
    import timm
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    kwargs: dict[str, Any] = {"pretrained": True, "num_classes": 0}
    if "dinov2" in model_name:
        kwargs["img_size"] = input_size
    model = timm.create_model(model_name, **kwargs).eval()

    dummy = torch.zeros(1, 3, input_size, input_size)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["embedding"],
        opset_version=opset,
        dynamo=False,
    )
    return out_path


def build_encoder(cfg: Any) -> Encoder:
    """Construct the encoder named by `cfg.fasttier.encoder`. Nothing hardcoded."""
    ft = cfg["fasttier"] if not hasattr(cfg, "fasttier") else cfg.fasttier
    kind = str(ft["encoder"])

    if kind == "downsample":
        return DownsampleEncoder(size=ft.get("downsample_size", 32))
    if kind == "torch":
        return TorchEncoder(
            model_name=ft["model"],
            input_size=ft.get("input_size", 224),
            device=ft.get("device", "cuda"),
            dtype=ft.get("dtype", "fp16"),
            clip_norm=ft.get("clip_norm", False),
        )
    if kind == "onnx":
        return OnnxEncoder(
            onnx_path=ft["onnx_path"],
            input_size=ft.get("input_size", 224),
            providers=ft.get("providers"),
            clip_norm=ft.get("clip_norm", False),
        )
    raise ValueError(f"unknown fasttier.encoder {kind!r}; expected downsample, torch or onnx")
