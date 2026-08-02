"""Capture — frame sources. Phase 0 synchronous; Phase 1 wraps these in a capture thread."""

from .sources import (
    BACKENDS,
    FileSource,
    Frame,
    FrameSource,
    SyntheticSource,
    WebcamSource,
    build_source,
)

__all__ = [
    "BACKENDS",
    "FileSource",
    "Frame",
    "FrameSource",
    "SyntheticSource",
    "WebcamSource",
    "build_source",
]
