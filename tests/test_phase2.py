"""Phase 2 exit criteria.

`PROMPT.md`: *test asserting sustained >=30 FPS over a 30-second headless run, with per-stage
timings in the metrics JSON.*

Per STATUS.md Q1 that criterion is read as **>=29.5 FPS on the live camera** (whose sensor caps at
~30 and cannot exceed it) **plus >=30 FPS unpaced**, which is the number that actually shows
headroom.

Phase 2 must show:
  1. the scorers behave — motion is lighting-sensitive, novelty accumulates drift, the rolling
     reference is frame-rate independent;
  2. an ONNX session that silently fell back to CPU is an error, not a result;
  3. the encoder choice is defended by the recorded benchmark, not by assertion;
  4. sustained frame rate with per-stage timings in the metrics JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from peripheral.capture import Frame
from peripheral.cli._args import load_config
from peripheral.fasttier import (
    DownsampleEncoder,
    FastTier,
    MotionScorer,
    RollingReference,
    cosine_distance,
)
from peripheral.runtime import EXIT_OK
from peripheral.telemetry.schema import validate_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
rng = np.random.default_rng(1337)


def _img(seed: int = 0, w: int = 320, h: int = 240) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _frame(i: int, img: np.ndarray, t: float | None = None) -> Frame:
    return Frame(frame_id=i, t_capture=float(i) / 30.0 if t is None else t, image=img)


# --------------------------------------------------------------------------------------------
# 1. Scorers
# --------------------------------------------------------------------------------------------
def test_motion_is_zero_on_the_first_frame():
    """No previous frame means no motion — not an exception, and not a spurious spike."""
    assert MotionScorer().score(_img(1)) == 0.0


def _scene(x: int, y: int, w: int = 320, h: int = 240) -> np.ndarray:
    """Flat grey background with a white block at (x, y) — a structured, movable object.

    Not random noise: two noise frames both blur toward the same mean grey, so differencing them
    correctly reports a small number and would make this test assert the wrong thing.
    """
    img = np.full((h, w, 3), 128, np.uint8)
    cv2.rectangle(img, (x, y), (x + 90, y + 90), (255, 255, 255), -1)
    return img


def test_motion_detects_change_and_ignores_a_static_scene():
    m = MotionScorer()
    img = _scene(20, 20)
    m.score(img)
    assert m.score(img) == pytest.approx(0.0, abs=1e-6), "a static scene must not read as motion"
    assert m.score(_scene(200, 120)) > 0.05, "an object moving across the frame must read as motion"


def test_motion_is_fooled_by_a_brightness_change():
    """The premise of the whole fast tier: pixel differencing cannot tell light from event.

    This is not a defect to fix here — it is why an embedding exists downstream.
    """
    m = MotionScorer()
    img = _img(4)
    m.score(img)
    brighter = cv2.convertScaleAbs(img, alpha=1.0, beta=40)
    assert m.score(brighter) > 0.1, "a pure lighting change should light up a motion detector"


def test_rolling_reference_is_not_ready_until_it_has_settled():
    """An unsettled reference must report 0.0 novelty rather than a large meaningless distance."""
    ref = RollingReference(half_life_s=2.0, min_updates=5)
    e = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(3):
        ref.update(e, t=i * 0.033)
    assert not ref.ready
    assert ref.distance(np.array([0.0, 1.0, 0.0], dtype=np.float32)) == 0.0


def test_rolling_reference_tracks_a_settled_scene_to_zero_novelty():
    ref = RollingReference(half_life_s=0.5, min_updates=3)
    e = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    for i in range(30):
        ref.update(e, t=i * 0.033)
    assert ref.distance(e) == pytest.approx(0.0, abs=1e-5)


def test_rolling_reference_half_life_is_in_seconds_not_frames():
    """Behaviour must not silently change with capture rate.

    The same elapsed time at 15 FPS and 60 FPS must leave the reference in the same place, or
    every threshold tuned in Phase 4 would be secretly a function of frame rate.
    """
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)

    def drift(fps: float) -> float:
        ref = RollingReference(half_life_s=1.0, min_updates=1)
        ref.update(a, t=0.0)
        n = int(2.0 * fps)
        for i in range(1, n + 1):
            ref.update(b, t=i / fps)
        return ref.distance(b)

    assert drift(15.0) == pytest.approx(drift(60.0), abs=0.02)


def test_novelty_accumulates_slow_drift_that_frame_to_frame_change_misses():
    """A slow drift must become visible even though no single frame looks different."""
    ref = RollingReference(half_life_s=1.0, min_updates=2)
    prev = None
    per_frame = []
    for i in range(60):
        angle = i * 0.02
        e = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        if prev is not None:
            per_frame.append(cosine_distance(prev, e))
        ref.update(e, t=i / 30.0)
        prev = e
    final = np.array([np.cos(60 * 0.02), np.sin(60 * 0.02)], dtype=np.float32)
    assert max(per_frame) < 1e-3, "no single frame should look different"
    assert ref.distance(final) > max(per_frame), "but the drift must accumulate in the reference"


def test_fast_tier_produces_all_three_signals():
    ft = FastTier(encoder=DownsampleEncoder(16), motion_size=32, half_life_s=1.0)
    scores = [ft.process(_frame(i, _img(i))) for i in range(10)]
    assert scores[0].motion == 0.0 and scores[0].scene_change == 0.0
    last = scores[-1]
    assert last.motion > 0 and last.scene_change > 0
    assert 0.0 <= last.novelty <= 2.0
    assert all(s.frame_id == i for i, s in enumerate(scores))


def test_fast_tier_reset_clears_history():
    ft = FastTier(encoder=DownsampleEncoder(16))
    for i in range(5):
        ft.process(_frame(i, _img(i)))
    ft.reset()
    assert ft.process(_frame(99, _img(99))).motion == 0.0


def test_embeddings_are_unit_norm():
    enc = DownsampleEncoder(16)
    for s in range(5):
        assert np.linalg.norm(enc.encode(_img(s))) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------------------------
# 2. A CPU fallback is an error, not a result
# --------------------------------------------------------------------------------------------
@pytest.mark.gpu
def test_onnx_refuses_to_report_cpu_numbers_as_gpu(tmp_path):
    """ORT creates a working session on CPU when a GPU provider's DLLs are missing.

    Measured 2026-08-04: `onnxruntime-gpu` 1.28 needs CUDA 13 DLLs that have no Windows pip
    route, so it fell back silently and CPU latencies were reported as GPU ones. The encoder now
    refuses unless CPU is asked for explicitly.
    """
    from peripheral.fasttier.encoders import OnnxEncoder, export_onnx

    onnx = tmp_path / "m.onnx"
    try:
        export_onnx("mobilenetv3_small_100", onnx, input_size=224)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot export ONNX here: {exc}")

    with pytest.raises(RuntimeError, match="fell back to CPU"):
        OnnxEncoder(onnx, input_size=224, providers=["NotARealProvider"])

    # Asking for CPU explicitly is fine.
    enc = OnnxEncoder(onnx, input_size=224, providers=["CPUExecutionProvider"],
                      allow_cpu_fallback=True)
    assert np.linalg.norm(enc.encode(_img(7))) == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------------------------
# 3. The encoder choice is defended by the benchmark
# --------------------------------------------------------------------------------------------
def test_config_uses_the_encoder_the_benchmark_chose():
    cfg = load_config("config", [])
    assert cfg.fasttier.encoder == "onnx", "ONNX Runtime beat PyTorch on every candidate"
    assert cfg.fasttier.model == "mobilenetv3_small_100"
    assert cfg.fasttier.half_life_s > 0


def test_benchmark_shows_the_control_is_a_motion_detector():
    """The finding that justifies paying for an encoder at all.

    `d_semantic / d_lighting` alone rewards brightness invariance, which the mean-centred,
    L2-normalised control has by construction. Its `semantic / motion` below 1 is what exposes it.
    """
    path = RESULTS / "phase2_encoder_bench.json"
    if not path.exists():
        pytest.skip("encoder benchmark not run in this checkout")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["label"]: r for r in doc["extra"]["encoder_bench"]["candidates"] if "error" not in r}

    control = next(r for k, r in rows.items() if "control" in k)
    assert control["semantic_vs_motion"] < 1.0, (
        "the control should react more to movement than to a semantic event"
    )
    chosen = next(r for k, r in rows.items() if "mobilenetv3_small" in k and "onnx" in k)
    assert chosen["semantic_vs_motion"] > 1.0
    assert chosen["semantic_vs_motion"] > control["semantic_vs_motion"] * 5
    assert chosen["p50_ms"] < 10.0, "the fast tier budget is ~10 ms per frame"


def test_benchmark_measured_both_backends():
    path = RESULTS / "phase2_encoder_bench.json"
    if not path.exists():
        pytest.skip("encoder benchmark not run in this checkout")
    doc = json.loads(path.read_text(encoding="utf-8"))
    labels = [r["label"] for r in doc["extra"]["encoder_bench"]["candidates"]]
    assert any("torch" in l for l in labels) and any("onnx" in l for l in labels), (
        "PROMPT.md asks for both to be measured, not for ORT to be assumed faster"
    )


# --------------------------------------------------------------------------------------------
# 4. THE EXIT CRITERION
# --------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.gpu
def test_fast_tier_sustains_frame_rate_with_per_stage_timings(tmp_path):
    """30-second headless run, VLM disabled, per-stage timings in the metrics JSON."""
    from peripheral.cli import fast_tier

    clip = REPO_ROOT / "data" / "clips" / "desk_60s.mp4"
    onnx = REPO_ROOT / "models" / "onnx" / "mobilenetv3_small_100_224.onnx"
    if not clip.exists() or not onnx.exists():
        pytest.skip("clip or exported ONNX encoder not present")

    out = tmp_path / "ft.json"
    code = fast_tier.main([
        "--duration", "30", "--headless", "--metrics-out", str(out),
        "-o", "capture=file", "-o", f"capture.path={clip.as_posix()}",
    ])
    assert code == EXIT_OK

    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_metrics(doc)
    m = doc["metrics"]

    assert m["achieved_fps"] >= 30.0, f"only sustained {m['achieved_fps']} FPS"
    assert m["frames_dropped"] == 0, "a fast tier inside budget should not drop frames"

    stages = m["stages_ms"]
    assert "fast_tier" in stages, "per-stage timings are half the exit criterion"
    assert stages["fast_tier"]["n"] > 800, "the fast tier must have scored ~900 frames"
    assert stages["fast_tier"]["p50"] < 10.0, "fast tier must fit its ~10 ms per-frame budget"

    ft = doc["extra"]["fast_tier"]
    for signal in ("motion", "novelty", "scene_change"):
        assert ft[signal]["max"] > 0, f"{signal} never moved — the scorer is not wired up"


@pytest.mark.gpu
def test_recorded_runs_meet_the_criterion():
    """The committed Phase 2 runs, checked as artifacts rather than re-run."""
    live = RESULTS / "phase2_fasttier_webcam.json"
    unpaced = RESULTS / "phase2_fasttier_unpaced.json"
    if not live.exists():
        pytest.skip("Phase 2 runs not generated in this checkout")

    d = json.loads(live.read_text(encoding="utf-8"))
    validate_metrics(d)
    # The camera caps at ~30 FPS, so live can never exceed it (STATUS.md Q1).
    assert d["metrics"]["achieved_fps"] >= 29.5
    assert d["metrics"]["stages_ms"]["fast_tier"]["p50"] < 10.0

    if unpaced.exists():
        u = json.loads(unpaced.read_text(encoding="utf-8"))
        rate = u["extra"]["fast_tier"]["n_scored"] / u["run"]["duration_actual_s"]
        assert rate >= 30.0, f"unpaced throughput {rate:.1f} FPS shows no headroom"


@pytest.mark.gpu
def test_the_phase2_chart_exists():
    if not (RESULTS / "phase2_encoder_bench.json").exists():
        pytest.skip("Phase 2 artifacts not generated in this checkout")
    png = RESULTS / "phase2_fast_tier.png"
    assert png.exists(), "run peripheral.viz.phase2_chart to build it"
    assert png.stat().st_size > 50_000
