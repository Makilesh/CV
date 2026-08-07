"""Phase 3 exit criteria.

`PROMPT.md`: *comparison table in `RESULTS.md`, a chosen configuration defended against the 12 GB
constraint, and a test asserting the TTFT target* — **p95 TTFT under 400 ms**, photon-to-first-token
including vision encoding.

Phase 3 must show:
  1. quality scoring behaves, and is honest about measuring fidelity rather than correctness;
  2. prompt ordering — the thing that decides whether KV reuse is possible — is what we think;
  3. the chosen configuration fits 12 GB with room for the fast tier;
  4. p95 photon-to-first-token is under 400 ms **in the real pipeline**, not just in a bench loop.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from peripheral.cli._args import load_config
from peripheral.runtime import EXIT_OK
from peripheral.telemetry.schema import validate_metrics
from peripheral.vlm.llama_server import LlamaServerClient
from peripheral.vlm.quality import (
    content_f1,
    is_degenerate,
    score_against_reference,
    self_consistency,
    similarity,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
TARGET_TTFT_MS = 400.0
VRAM_LIMIT_GB = 11.94


# --------------------------------------------------------------------------------------------
# 1. Quality scoring
# --------------------------------------------------------------------------------------------
def test_identical_answers_score_perfectly():
    a = "A person sits at a desk in front of a monitor."
    assert similarity(a, a) == pytest.approx(1.0)
    assert content_f1(a, a) == pytest.approx(1.0)


def test_scoring_ignores_phrasing_but_not_content():
    ref = "A person sits at a desk in front of a monitor."
    reworded = "At a desk, in front of the monitor, a person is sitting."
    different = "An empty kitchen with a bowl of fruit on the counter."
    assert content_f1(ref, reworded) > 0.9, "word order must not be penalised"
    assert content_f1(ref, different) < 0.2, "different content must score low"


def test_stopwords_do_not_inflate_agreement():
    """Shared boilerplate must not make two unrelated answers look similar."""
    a = "The image shows a red bicycle leaning on a wall."
    b = "The image shows a plate of pasta on a table."
    assert content_f1(a, b) < 0.3


def test_degenerate_answers_are_detected():
    assert is_degenerate("")
    assert is_degenerate("the")
    assert is_degenerate("a a a a a a a a"), "a looping answer is degenerate"
    assert not is_degenerate("A person sits at a desk working on a laptop computer.")


def test_degenerate_answers_are_counted_separately_from_agreement():
    """Two degenerate answers agree with each other; that must not read as quality."""
    res = score_against_reference(["a a a a", "b b b b"], ["a a a a", "b b b b"])
    assert res["similarity_mean"] == pytest.approx(1.0)
    assert res["degenerate_rate"] == pytest.approx(1.0), "and the degeneracy must be visible"


def test_empty_population_scores_null_not_zero():
    res = score_against_reference([], [])
    assert res["n"] == 0
    assert res["similarity_mean"] is None


def test_self_consistency_flags_nondeterminism():
    assert self_consistency(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    assert self_consistency(["a", "b"], ["a", "c"]) == pytest.approx(0.5)


# --------------------------------------------------------------------------------------------
# 2. Prompt ordering decides whether KV reuse is possible at all
# --------------------------------------------------------------------------------------------
def _b64() -> str:
    return base64.b64encode(b"not-a-real-jpeg").decode()


def test_text_first_ordering_puts_a_cacheable_prefix_before_the_image():
    """Image tokens differ every frame, so anything after the image is unreusable."""
    p = LlamaServerClient.build_payload(_b64(), "What is happening?", system="You are a monitor.")
    content = p["messages"][-1]["content"]
    assert p["messages"][0]["role"] == "system"
    assert content[0]["type"] == "text", "text must precede the image to be cacheable"
    assert content[1]["type"] == "image_url"


def test_image_first_ordering_leaves_nothing_cacheable():
    p = LlamaServerClient.build_payload(_b64(), "What is happening?", image_first=True)
    content = p["messages"][-1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "text"


def test_payload_defaults_are_deterministic_and_streaming():
    p = LlamaServerClient.build_payload(_b64(), "q")
    assert p["temperature"] == 0.0, "a quality sweep must not sample"
    assert p["stream"] is True


# --------------------------------------------------------------------------------------------
# 3. The chosen configuration is defended
# --------------------------------------------------------------------------------------------
def test_chosen_config_fits_the_12gb_constraint():
    """The chosen model must leave room for the fast tier, the display and the OS."""
    path = RESULTS / "phase3_vlm_bench.json"
    if not path.exists():
        pytest.skip("Phase 3 sweep not run in this checkout")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["label"]: r for r in doc["extra"]["vlm_bench"]["configs"] if "error" not in r}

    cfg = load_config("config", [])
    # configs/vlm/<chosen>.yaml carries `bench_label`, which is the row it was chosen from.
    label = cfg.vlm.get("bench_label")
    assert label, "the chosen VLM config must name the sweep row that justifies it"
    row = rows.get(label)
    assert row is not None, f"chosen config {label!r} is not in the sweep it claims to come from"

    assert row["peak_vram_gb"] < VRAM_LIMIT_GB, "chosen config exceeds available VRAM"
    # Phase 2 measured the fast tier at 0.61 GB peak; leave clear headroom above the VLM.
    assert row["peak_vram_gb"] < VRAM_LIMIT_GB - 1.5, (
        "chosen config leaves no room for the fast tier and the display"
    )


def test_sweep_covered_multiple_models_and_quant_levels():
    path = RESULTS / "phase3_vlm_bench.json"
    if not path.exists():
        pytest.skip("Phase 3 sweep not run in this checkout")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in doc["extra"]["vlm_bench"]["configs"] if "error" not in r]
    assert len({r["family"] for r in rows}) >= 3, "PROMPT.md asks for 3-4 models"
    assert len({r["quant"] for r in rows}) >= 2, "and multiple quantization levels"
    for r in rows:
        assert r["peak_vram_gb"] > 0 and r["load_time_s"] > 0 and r["tokens_per_s"] > 0


def test_results_md_has_the_comparison_table():
    md = REPO_ROOT / "RESULTS.md"
    if not (RESULTS / "phase3_vlm_bench.json").exists():
        pytest.skip("Phase 3 sweep not run in this checkout")
    assert md.exists(), "Phase 3 exit criterion is a comparison table in RESULTS.md"
    text = md.read_text(encoding="utf-8")
    assert "TTFT" in text and "VRAM" in text
    assert "|" in text, "the comparison table should be a table"


# --------------------------------------------------------------------------------------------
# 4. THE EXIT CRITERION — p95 TTFT under 400 ms in the real pipeline
# --------------------------------------------------------------------------------------------
@pytest.mark.vlm
def test_recorded_pipeline_run_meets_the_ttft_target():
    """photon-to-first-token p95 < 400 ms, measured through the full pipeline.

    The bench loop measures frame-in-hand to first token. This asserts the target on the metric
    that actually matters — from the moment OpenCV returns the frame, through queues, JPEG
    encoding, HTTP, vision encoding and prefill.
    """
    path = RESULTS / "phase3_chosen_pipeline.json"
    if not path.exists():
        pytest.skip("chosen-config pipeline run not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    validate_metrics(doc)
    ttft = doc["metrics"]["photon_to_first_token_ms"]
    assert ttft is not None, "no VLM calls were made"
    assert ttft["p95"] < TARGET_TTFT_MS, (
        f"p95 photon-to-first-token {ttft['p95']} ms exceeds the {TARGET_TTFT_MS} ms target"
    )


@pytest.mark.slow
@pytest.mark.vlm
def test_ttft_target_live(tmp_path):
    """The same assertion, run fresh against the recorded clip."""
    from peripheral.cli import naive_baseline

    cfg = load_config("config", [])
    clip = REPO_ROOT / "data" / "clips" / "desk_60s.mp4"
    if not Path(cfg.vlm.model).exists() or not clip.exists():
        pytest.skip("chosen model weights or clip not present")

    out = tmp_path / "ttft.json"
    code = naive_baseline.main([
        "--duration", "25", "--headless", "--metrics-out", str(out),
        "-o", "capture=file", "-o", f"capture.path={clip.as_posix()}",
    ])
    assert code == EXIT_OK

    doc = json.loads(out.read_text(encoding="utf-8"))
    validate_metrics(doc)
    ttft = doc["metrics"]["photon_to_first_token_ms"]
    assert ttft is not None and ttft["n"] > 20, "not enough calls to trust a p95"
    assert ttft["p95"] < TARGET_TTFT_MS, (
        f"p95 photon-to-first-token {ttft['p95']} ms exceeds the {TARGET_TTFT_MS} ms target"
    )


@pytest.mark.vlm
def test_kv_reuse_was_measured_before_and_after():
    path = RESULTS / "phase3_kv_reuse.json"
    if not path.exists():
        pytest.skip("KV-reuse benchmark not run in this checkout")
    doc = json.loads(path.read_text(encoding="utf-8"))
    arms = [a for a in doc["extra"]["kv_reuse"]["arms"] if "error" not in a]
    assert len(arms) >= 3, "before/after needs at least the naive and reusing arms"
    assert any(a["image_first"] for a in arms), "the naive image-first baseline must be measured"
    assert any(not a["stream"] for a in arms), "streaming decode needs a non-streaming comparison"
    non_streaming = next(a for a in arms if not a["stream"])
    assert non_streaming["ttft_p50_ms"] is None, (
        "a non-streaming call has no first token before the answer completes"
    )
