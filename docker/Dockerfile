# Peripheral — REPLAY / EVAL PATH ONLY. CPU-only. Not the demo.
#
# WHY THIS CONTAINER DOES NOT RUN THE DEMO, stated plainly rather than discovered by a user:
#
#   * The live demo needs a webcam. Passing a USB camera into a WSL2-backed Docker container on
#     Windows requires usbipd-win, a kernel with USB/IP, and a working v4l2 shim — and even then
#     `CAP_DSHOW` does not exist inside Linux, so the measured backend choice (STATUS.md D1) would
#     not apply. The demo is a native Windows program and stays one.
#   * The slow tier needs CUDA. GPU passthrough works, but the measured numbers in RESULTS.md come
#     from a specific driver, a specific llama.cpp build and a 95 W power cap. A container that
#     *looks* like it reproduces them while running on different hardware would be worse than no
#     container at all.
#
# So this image reproduces the part that is genuinely portable: the replay harness, the scheduler
# policies, the simulator and the test suite, all on CPU. It gates CORRECTNESS, not performance.
FROM python:3.12-slim

# OpenCV needs these even for headless video decode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only wheel set. torch is the CPU build here on purpose: nothing in the eval path needs CUDA,
# and pulling the cu128 wheel would add ~3 GB to reproduce numbers this container must not claim.
COPY docker/requirements-eval.txt ./
RUN pip install --no-cache-dir -r requirements-eval.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY pyproject.toml ./

ENV PYTHONPATH=/app/src
ENV PERIPHERAL_CONFIG_DIR=/app/configs

# Default: the correctness gate. Excludes anything needing a GPU, a camera or model weights.
CMD ["python", "-m", "pytest", "tests/", "-q", "-m", "not gpu and not webcam and not vlm and not slow"]
