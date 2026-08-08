"""Fast tier — runs on every frame inside a ~10 ms budget. Never blocks, never calls the VLM."""

from .encoders import (
    DownsampleEncoder,
    Encoder,
    OnnxEncoder,
    TorchEncoder,
    build_encoder,
    export_onnx,
)
from .scoring import (
    FastTier,
    FrameScores,
    MotionScorer,
    RollingReference,
    cosine_distance,
)

__all__ = [
    "DownsampleEncoder",
    "Encoder",
    "FastTier",
    "FrameScores",
    "MotionScorer",
    "OnnxEncoder",
    "RollingReference",
    "TorchEncoder",
    "build_encoder",
    "cosine_distance",
    "export_onnx",
]
