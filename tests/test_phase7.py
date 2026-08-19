"""Phase 7 exit criteria.

`PROMPT.md`: *a fresh clone plus documented setup reaches a working live demo, and the README
communicates the result without anyone running anything.*

The second half is the harder one to test, so these check the properties that make it true: that the
README carries the claim, both figures, the hardware caveats and — critically — the **non-claims**,
and that the shipped artifacts exist. Most of these run on CPU with no weights, so CI enforces them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from peripheral.cli._args import build_parser, load_config
from peripheral.viz.hud import HudState, draw

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
RESULTS_MD = REPO_ROOT / "RESULTS.md"
RESULTS = REPO_ROOT / "results"


def _readme() -> str:
    if not README.exists():
        pytest.skip("README.md not present")
    return README.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------
# 1. The README communicates the result without running anything
# --------------------------------------------------------------------------------------------
def test_readme_leads_with_the_claim():
    text = _readme()
    head = text[:1200]
    assert "datacenter" in head.lower() and "laptop" in head.lower(), (
        "the claim this project defends should be at the top"
    )


def test_readme_shows_both_figures_and_the_demo_gif():
    text = _readme()
    for asset in ("phase4_pareto.png", "pareto_latency.png", "demo.gif"):
        assert asset in text, f"README does not reference {asset}"
    # Above the fold: both figures inside the first ~40% of the document.
    cut = int(len(text) * 0.4)
    assert "phase4_pareto.png" in text[:cut] and "pareto_latency.png" in text[:cut], (
        "both Pareto charts must be above the fold"
    )


def test_readme_states_the_hardware_and_that_runs_were_plugged_in():
    text = _readme()
    assert "95 W" in text, "the power cap is load-bearing for every energy number"
    assert re.search(r"plugged in", text, re.I), (
        "energy numbers are meaningless on battery and the README must say so"
    )
    assert "11.94 GB" in text or "12 GB" in text


def test_readme_carries_the_non_claims_above_the_fold():
    """The honesty requirement, and the one most likely to rot. It must not become a footnote."""
    text = _readme()
    idx = text.lower().find("what was *not* established")
    if idx < 0:
        idx = text.lower().find("not established")
    assert idx > 0, "README must state what was NOT established"
    assert idx < len(text) * 0.5, "the non-claims must be above the fold, not buried at the end"

    for required in ("fixed_interval", "generalise", "OVO-Bench"):
        assert required in text, f"README omits the {required!r} limitation"


def test_readme_does_not_overclaim_novelty():
    """STATUS.md D-list and PROMPT.md are explicit: our differentiator is the measurement."""
    text = _readme()
    assert "Dispider" in text, "prior art must be named"
    assert re.search(r"not novel in architecture|not pretend otherwise|contribution here is",
                     text, re.I), "README must position the contribution honestly"


def test_readme_documents_the_docker_split():
    text = _readme()
    assert "no demo container" in text.lower() or "deliberately no demo" in text.lower(), (
        "the Docker split must be documented rather than shipped broken"
    )
    assert "correctness, not performance" in text.lower()


def test_readme_reports_the_bugs():
    """Both timing bugs changed reported numbers; hiding them would misrepresent the result."""
    text = _readme()
    assert re.search(r"bugs? (are|is) part of the result|Two bugs", text, re.I)
    assert "20 ms" in text, "the future-evidence bug should be quantified"


# --------------------------------------------------------------------------------------------
# 2. Shipped artifacts
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["phase4_pareto.png", "pareto_latency.png"])
def test_figures_exist(name):
    if not (RESULTS / "phase4_sweep.json").exists():
        pytest.skip("results not generated in this checkout")
    path = RESULTS / name
    assert path.exists(), f"{name} missing — regenerate with peripheral.viz"
    assert path.stat().st_size > 40_000


def test_demo_gif_exists_and_is_small_enough_to_render():
    if not (RESULTS / "demo_frames").exists():
        pytest.skip("demo frames not recorded in this checkout")
    gif = RESULTS / "demo.gif"
    assert gif.exists(), "run peripheral.viz.make_gif"
    assert gif.stat().st_size < 15_000_000, "a GIF this large will not render in a README"


def test_ci_workflow_is_cpu_only_and_says_so():
    wf = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert wf.exists(), "Phase 7 requires GitHub Actions CI"
    text = wf.read_text(encoding="utf-8")
    assert "correctness" in text.lower() and "not performance" in text.lower(), (
        "CI must state that it gates correctness, not performance"
    )
    for marker in ("gpu", "webcam", "vlm", "slow"):
        assert marker in text, f"CI must exclude the {marker!r} marker"
    assert "test_phase6" in text, "the anti-cheat suite must run as its own visible step"


def test_docker_eval_image_does_not_pretend_to_run_the_demo():
    compose = REPO_ROOT / "docker" / "docker-compose.yml"
    assert compose.exists()
    text = compose.read_text(encoding="utf-8")
    assert "demo" not in re.sub(r"#.*", "", text).lower().split("services:")[1].split("\n  ")[0] \
        or "no `demo` service" in text, "there must be no demo service"


def test_eval_container_requirements_exclude_gpu_stacks():
    """Check the actual requirements, not the comments that explain what is excluded."""
    raw = (REPO_ROOT / "docker" / "requirements-eval.txt").read_text(encoding="utf-8")
    pinned = [
        line.strip() for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    joined = "\n".join(pinned)
    for forbidden in ("onnxruntime-gpu", "cu128", "bitsandbytes", "timm", "transformers"):
        assert forbidden not in joined, f"the CPU eval image must not pull {forbidden}"
    assert any(p.startswith("torch==") and "+cu" not in p for p in pinned), (
        "torch must be the CPU build in the eval container"
    )


# --------------------------------------------------------------------------------------------
# 3. The demo obeys the same contract as everything else
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("flag", ["--duration", "--headless", "--metrics-out"])
def test_demo_honours_the_bounded_runner_contract(flag):
    """A demo that cannot be measured is a screenshot."""
    opts = {s for a in build_parser("p", "d")._actions for s in a.option_strings}
    assert flag in opts


def test_demo_config_uses_the_chosen_system_not_a_showcase():
    cfg = load_config("config", [])
    assert cfg.demo.policy_kind in {"embedding_novelty", "motion_threshold", "learned",
                                    "fixed_interval", "information_gain"}
    assert cfg.vlm.name, "the demo must run the chosen VLM configuration"


def test_ci_fasttier_profile_cannot_be_mistaken_for_a_quality_result():
    """CI runs the control encoder; its own config must say it is not a quality configuration."""
    text = (REPO_ROOT / "configs" / "fasttier" / "ci.yaml").read_text(encoding="utf-8")
    cfg = load_config("config", ["fasttier=ci"])
    assert cfg.fasttier.encoder == "downsample"
    assert "motion detector" in text.lower() or "cannot produce a quality result" in text.lower()


# --------------------------------------------------------------------------------------------
# 4. The HUD renders
# --------------------------------------------------------------------------------------------
def test_hud_draws_without_mutating_the_frame():
    import numpy as np

    frame = np.full((240, 320, 3), 60, np.uint8)
    before = frame.copy()
    state = HudState(fps=30.0, novelty=0.2, threshold=0.12, calls_per_min=7.5, n_calls=3,
                     answer="a person at a desk", answer_age_s=1.2, last_ttft_ms=164.0,
                     model="Qwen3-VL-4B", policy="embedding_novelty @ 0.12")
    for v in (0.01, 0.05, 0.2, 0.3):
        state.novelty_history.append(v)
        state.fire_history.append(v > 0.12)

    out = draw(frame, state)
    assert out.shape == frame.shape
    assert np.array_equal(frame, before), "draw() must not modify the frame it is given"
    assert not np.array_equal(out, frame), "the HUD must actually draw something"


def test_hud_survives_an_empty_state():
    """First frames have no answer and no history; the HUD must not crash before the first call."""
    import numpy as np

    out = draw(np.zeros((120, 160, 3), np.uint8), HudState())
    assert out.shape == (120, 160, 3)
