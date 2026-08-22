"""Phase 8 exit criteria.

Phase 8 asked whether the scheduler thesis could be rescued. Its exit criterion was: *an adaptive
policy beats `fixed_interval` at a matched budget on all six clips.* **It failed**, and these tests
pin both what was gained and what was not, so neither can quietly drift.

The phase was allowed to fail by design. What must not happen is the failure being forgotten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from peripheral.eval.simulate import simulate
from peripheral.eval.traces import ClipTrace
from peripheral.fasttier.spatial import PatchRollingReference
from peripheral.scheduler.policies import AdaptiveNoveltyPolicy, FrameContext

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
TRACE_DIR = REPO_ROOT / "data" / "traces"


def _ctx(**kw) -> FrameContext:
    base = dict(frame_idx=0, t=0.0, motion=0.0, novelty=0.0, scene_change=0.0,
                embedding=None, t_last_call=None, n_calls=0)
    base.update(kw)
    return FrameContext(**base)


def _cells(h=7, w=7, c=8, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(h, w, c)).astype(np.float32)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


# --------------------------------------------------------------------------------------------
# 1. Patch novelty — the 8b signal
# --------------------------------------------------------------------------------------------
def test_patch_reference_reports_zero_until_settled():
    ref = PatchRollingReference(half_life_s=2.0, min_updates=5)
    a = _cells()
    for i in range(3):
        ref.update(a, t=i * 0.033)
    assert not ref.ready
    assert ref.distance(_cells(seed=1)) == 0.0


def test_patch_novelty_survives_a_localised_change_that_pooling_averages_away():
    """The whole point of 8b, as a test.

    One cell of 49 changes completely. Pooled novelty divides that by 49 and barely moves; top-k
    patch novelty sees it at close to full strength. This is the `object_events` failure — an object
    in a corner against a person moving through the middle — in miniature.
    """
    base = _cells(seed=0)
    ref = PatchRollingReference(half_life_s=2.0, min_updates=2, top_k=3)
    for i in range(10):
        ref.update(base, t=i * 0.033)

    changed = base.copy()
    changed[0, 0] = -changed[0, 0]          # one cell flipped: cosine distance 2.0 there
    patch_score = ref.distance(changed)

    def pooled(x):
        v = x.reshape(-1, x.shape[-1]).mean(axis=0)
        return v / max(np.linalg.norm(v), 1e-12)

    pooled_score = float(1.0 - np.dot(pooled(base), pooled(changed)))
    assert patch_score > 5 * pooled_score, (
        f"patch {patch_score:.4f} should dominate pooled {pooled_score:.4f} for a local change"
    )


def test_patch_novelty_ignores_a_single_noisy_cell_less_than_a_real_object():
    """top_k=3 is the compromise: one spiking cell must not score like three."""
    base = _cells(seed=2)
    ref = PatchRollingReference(half_life_s=2.0, min_updates=2, top_k=3)
    for i in range(10):
        ref.update(base, t=i * 0.033)

    one = base.copy()
    one[3, 3] = -one[3, 3]
    three = base.copy()
    for cell in ((3, 3), (3, 4), (4, 3)):
        three[cell] = -three[cell]

    assert ref.distance(three) > ref.distance(one), "a larger object must score higher"


def test_patch_reference_half_life_is_in_seconds():
    """Same property the pooled reference has, and for the same reason."""
    a, b = _cells(seed=3), _cells(seed=4)

    def drift(fps: float) -> float:
        ref = PatchRollingReference(half_life_s=1.0, min_updates=1)
        ref.update(a, t=0.0)
        for i in range(1, int(2.0 * fps) + 1):
            ref.update(b, t=i / fps)
        return ref.distance(b)

    assert drift(15.0) == pytest.approx(drift(60.0), abs=0.03)


# --------------------------------------------------------------------------------------------
# 2. Adaptive quantile policy
# --------------------------------------------------------------------------------------------
def test_adaptive_policy_is_a_rate_controller():
    """q maps onto call rate: a higher quantile must fire strictly less often."""
    rng = np.random.default_rng(0)
    novelty = rng.random(1200)

    def n_calls(q: float) -> int:
        p = AdaptiveNoveltyPolicy(q, window_s=8.0, min_gap_s=0.0, min_samples=30)
        p.reset()
        t_last, calls = None, 0
        for i, v in enumerate(novelty):
            t = i / 30.0
            if p.decide(_ctx(frame_idx=i, t=t, novelty=float(v), t_last_call=t_last)).fire:
                calls += 1
                t_last = t
        return calls

    assert n_calls(0.80) > n_calls(0.95) > n_calls(0.99)


def test_adaptive_policy_calibrates_to_scene_scale():
    """The property a global threshold lacks: two scenes at different novelty scales, same rate."""
    rng = np.random.default_rng(1)
    quiet = rng.random(900) * 0.05          # a low-novelty scene
    busy = rng.random(900) * 0.8            # a high-novelty scene

    def n_calls(series) -> int:
        p = AdaptiveNoveltyPolicy(0.95, window_s=8.0, min_gap_s=0.0, min_samples=30)
        p.reset()
        t_last, calls = None, 0
        for i, v in enumerate(series):
            t = i / 30.0
            if p.decide(_ctx(frame_idx=i, t=t, novelty=float(v), t_last_call=t_last)).fire:
                calls += 1
                t_last = t
        return calls

    a, b = n_calls(quiet), n_calls(busy)
    assert abs(a - b) / max(a, b) < 0.25, (
        f"a scale-free policy should fire at a similar rate on both scenes, got {a} vs {b}"
    )


def test_adaptive_policy_respects_the_rate_limit_and_warmup():
    p = AdaptiveNoveltyPolicy(0.5, window_s=8.0, min_gap_s=0.5, min_samples=30)
    p.reset()
    assert p.decide(_ctx(t=0.0, novelty=0.5)).fire, "first frame always fires"
    assert not p.decide(_ctx(t=0.1, novelty=1.0, t_last_call=0.0)).fire, "rate limited"
    # Before the window fills there is no distribution to take a quantile of.
    assert not p.decide(_ctx(t=1.0, novelty=1.0, t_last_call=0.0)).fire


def test_adaptive_policy_sees_only_the_past():
    """It judges the current frame against a window that ends at the current frame."""
    p = AdaptiveNoveltyPolicy(0.9, window_s=1.0, min_gap_s=0.0, min_samples=5)
    p.reset()
    for i in range(60):
        p.decide(_ctx(frame_idx=i, t=i / 30.0, novelty=0.1, t_last_call=0.0))
    # A window of 1 s at 30 fps holds ~30 samples, not the whole 60-frame history.
    assert len(p._hist) <= 31


# --------------------------------------------------------------------------------------------
# 3. The trace signal view
# --------------------------------------------------------------------------------------------
def test_signal_view_swaps_novelty_and_shares_everything_else():
    path = TRACE_DIR / "object_events"
    if not ClipTrace.exists(path):
        pytest.skip("traces not built in this checkout")
    t = ClipTrace.load(path)
    if t.novelty_patch is None:
        pytest.skip("patch signals not computed; run peripheral.cli.add_spatial_signals")

    v = t.with_signal("patch")
    assert v is not t
    assert np.array_equal(v.novelty, t.novelty_patch)
    assert v.answers is t.answers, "the oracle answers must be shared, not copied or altered"
    assert np.array_equal(v.state_ids, t.state_ids)


def test_signal_view_rejects_an_unknown_signal():
    path = TRACE_DIR / "object_events"
    if not ClipTrace.exists(path):
        pytest.skip("traces not built in this checkout")
    with pytest.raises(ValueError, match="unknown signal"):
        ClipTrace.load(path).with_signal("telepathy")


# --------------------------------------------------------------------------------------------
# 4. THE EXIT CRITERION — which failed, and must stay recorded as failed
# --------------------------------------------------------------------------------------------
def _sweep(path: Path) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path.name} not generated in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))["extra"]["phase4_sweep"]["results"]


def _cheapest(rows, kind, split, bar=0.99):
    best = None
    for r in rows:
        if r["split"] != split or r["kind"] != kind or r["answer_validity_mean"] < bar:
            continue
        if best is None or r["calls_per_min_mean"] < best["calls_per_min_mean"]:
            best = r
    return best


@pytest.mark.parametrize("sweep", ["phase8_sweep_pooled.json", "phase8_sweep_patch.json"])
def test_no_content_policy_beats_the_timer_on_all_clips(sweep):
    """Phase 8's exit criterion, asserted in the direction the data actually went.

    If this ever starts failing, a content policy has finally beaten the timer across all six clips
    — which is the result the project wants, and RESULTS.md section 8 must then be rewritten rather
    than the test relaxed.
    """
    rows = _sweep(RESULTS / sweep)
    timer = _cheapest(rows, "fixed_interval", "all")
    assert timer is not None, "the timer baseline must reach the bar"

    winners = [
        k for k in ("adaptive_novelty", "embedding_novelty", "motion_threshold",
                    "information_gain", "learned")
        if (b := _cheapest(rows, k, "all")) and b["calls_per_min_mean"] < timer["calls_per_min_mean"]
    ]
    assert not winners, (
        f"a content policy now beats the timer on all clips ({winners}) — Phase 8's exit criterion "
        "has been met and RESULTS.md section 8 needs rewriting"
    )


def test_patch_signal_fixed_the_separability_failure():
    """8b's gain, which is real even though the end-to-end criterion failed."""
    pooled = RESULTS / "phase8_signal_diagnosis.json"
    patch = RESULTS / "phase8_diag_patch.json"
    if not (pooled.exists() and patch.exists()):
        pytest.skip("Phase 8 diagnoses not run in this checkout")

    def auc(path, clip):
        d = json.loads(path.read_text(encoding="utf-8"))["extra"]
        key = next(k for k in d if k.startswith("phase8_signal_diagnosis"))
        rows = {r["clip"]: r for r in d[key]["separability"]}
        return rows[clip]["auc_novelty"]

    before, after = auc(pooled, "object_events"), auc(patch, "object_events")
    assert after > before + 0.15, (
        f"patch novelty was supposed to rescue object_events separability: {before} -> {after}"
    )
    assert after > 0.85


def test_the_diagnosis_recorded_a_verdict_with_its_evidence():
    path = RESULTS / "phase8_signal_diagnosis.json"
    if not path.exists():
        pytest.skip("Phase 8a not run in this checkout")
    extra = json.loads(path.read_text(encoding="utf-8"))["extra"]
    key = next(k for k in extra if k.startswith("phase8_signal_diagnosis"))
    d = extra[key]
    assert d["verdict"] in {"signal_problem", "threshold_problem", "neither"}
    assert d["reason"]
    assert d["separability"], "the verdict must rest on per-clip separability, not a summary"
    assert "headroom" in d


def test_spatial_encoder_stays_inside_the_fast_tier_budget():
    """8b must not have bought separability with latency the fast tier cannot afford."""
    path = RESULTS / "phase8_spatial_signals.json"
    if not path.exists():
        pytest.skip("spatial signals not computed in this checkout")
    enc = json.loads(path.read_text(encoding="utf-8"))["extra"]["phase8_spatial_signals"]["encoder"]
    assert enc["grid"] == [7, 7], "a coarser grid would re-average local events away"
    assert "CUDAExecutionProvider" in enc["providers"]
