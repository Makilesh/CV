"""Shared test fixtures."""

from __future__ import annotations

import pytest

from peripheral.cli._args import load_config


@pytest.fixture
def synthetic_cfg():
    """Config using the synthetic source — deterministic, no camera, CPU-only."""
    return load_config(
        "config",
        [
            "capture=synthetic",
            "capture.target_fps=60",
            "run.duration_s=0.5",
            "telemetry.power_sample_hz=20",
        ],
    )


@pytest.fixture(scope="session")
def has_webcam() -> bool:
    import cv2

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    if ok:
        ok, _ = cap.read()
    cap.release()
    return bool(ok)
