"""Phase 5 exit criteria.

`PROMPT.md`: *numbers in `RESULTS.md`. If the accuracy cost exceeds the latency benefit, recommend
cutting the feature — four solid components beat five with one that doesn't earn its place.*

So the exit criterion for this phase is **a defensible verdict backed by measurements**, not a
working cache. These tests check that the machinery is correct (so the verdict can be trusted) and
that the verdict is actually derived from the numbers rather than asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from peripheral.cache import SceneStateSummary, SemanticCache
from peripheral.eval.simulate_cache import simulate_with_cache
from peripheral.eval.traces import ClipTrace
from peripheral.scheduler.policies import FixedIntervalPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# --------------------------------------------------------------------------------------------
# 1. Cache mechanics
# --------------------------------------------------------------------------------------------
def test_empty_cache_misses():
    c = SemanticCache(threshold=0.9)
    assert c.lookup(_unit([1, 0, 0]), t=0.0).hit is False


def test_identical_embedding_hits_and_returns_its_evidence_frame():
    c = SemanticCache(threshold=0.9)
    e = _unit([1, 0, 0])
    c.store(e, "a desk with a mug", frame_idx=7, t=1.0)
    look = c.lookup(e, t=2.0)
    assert look.hit and look.answer == "a desk with a mug"
    assert look.entry.frame_idx == 7, "staleness must be measured from the ORIGINAL evidence frame"
    assert look.staleness_s == pytest.approx(1.0)


def test_dissimilar_embedding_misses():
    c = SemanticCache(threshold=0.9)
    c.store(_unit([1, 0, 0]), "a", frame_idx=0, t=0.0)
    assert c.lookup(_unit([0, 1, 0]), t=0.1).hit is False


def test_stale_entry_is_not_served_however_similar():
    """An embedding match says the scene LOOKS the same, not that an old description is still true."""
    c = SemanticCache(threshold=0.9, max_staleness_s=5.0)
    e = _unit([1, 0, 0])
    c.store(e, "a", frame_idx=0, t=0.0)
    assert c.lookup(e, t=4.9).hit is True
    assert c.lookup(e, t=5.1).hit is False


def test_eviction_keeps_the_entries_that_earn_their_place():
    """Least-recently-useful, not LRU: a rarely-but-reliably-reused state is worth keeping."""
    c = SemanticCache(threshold=0.99, max_entries=2)
    useful, filler1, filler2 = _unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])
    c.store(useful, "useful", 0, 0.0)
    c.lookup(useful, 0.1)          # earns a hit
    c.store(filler1, "f1", 1, 1.0)
    c.store(filler2, "f2", 2, 2.0)  # forces an eviction
    assert c.n_evictions == 1
    assert any(e.answer == "useful" for e in c.entries), "the entry with hits must survive"


def test_hit_rate_is_none_before_any_lookup():
    assert SemanticCache().hit_rate is None


# --------------------------------------------------------------------------------------------
# 2. Scene-state summary answers with no VLM call
# --------------------------------------------------------------------------------------------
def test_scene_summary_answers_without_a_vlm_call():
    s = SceneStateSummary()
    assert s.query(0.0)["answer"] is None
    s.update("a person at a desk", frame_idx=3, t=1.0, changed=True)
    s.update("a person at a desk", frame_idx=9, t=2.0, changed=False)
    q = s.query(5.0)
    assert q["answer"] == "a person at a desk"
    assert q["held_for_s"] == pytest.approx(4.0)
    assert q["confirmations"] == 2
    assert q["evidence_frame"] == 3, "evidence stays the frame the answer came from"


# --------------------------------------------------------------------------------------------
# 3. The simulation accounts for cache hits honestly
# --------------------------------------------------------------------------------------------
def _trace_with_return(n=300, fps=30.0) -> tuple[ClipTrace, np.ndarray]:
    """State 0 → 1 → 0: the scene RETURNS, which is the only case a cache can exploit."""
    states = np.array([0] * 100 + [1] * 100 + [0] * 100, dtype=np.int32)
    answers = ["an empty desk" if s == 0 else "a mug on the desk" for s in states]
    emb = np.stack([_unit([1, 0]) if s == 0 else _unit([0, 1]) for s in states])
    tr = ClipTrace(
        clip_name="ret", fps=fps, n_frames=n,
        motion=np.zeros(n, np.float32), novelty=np.zeros(n, np.float32),
        scene_change=np.zeros(n, np.float32), answers=answers, state_ids=states,
        encoder={}, model="fake", prompt="p",
    )
    return tr, emb


def test_cache_avoids_calls_when_the_scene_returns():
    """A correct cache is free: same answers, fewer calls.

    Validity is compared against the same policy with no cache rather than against 1.0 — a 1 s
    timer necessarily lags a state change by up to 30 frames, and that lag is the policy's cost,
    not the cache's.
    """
    tr, emb = _trace_with_return()
    cached = simulate_with_cache(tr, FixedIntervalPolicy(1.0), SemanticCache(threshold=0.99), emb,
                                 warmup_frames=0)
    plain = simulate_with_cache(tr, FixedIntervalPolicy(1.0), None, emb, warmup_frames=0)

    assert cached.n_cache_hits > 0
    assert cached.n_vlm_calls < cached.n_triggers
    assert cached.calls_avoided == cached.n_triggers - cached.n_vlm_calls
    assert cached.false_hits == 0, "returning to a genuinely identical state is not a false hit"
    assert cached.answer_validity == pytest.approx(plain.answer_validity), (
        "a cache that only serves correct entries must cost nothing in validity"
    )


def test_no_cache_means_every_trigger_costs_a_call():
    tr, emb = _trace_with_return()
    r = simulate_with_cache(tr, FixedIntervalPolicy(1.0), None, emb, warmup_frames=0)
    assert r.n_vlm_calls == r.n_triggers
    assert r.n_cache_hits == 0
    assert r.calls_avoided == 0


def test_a_loose_threshold_produces_false_hits_and_costs_validity():
    """The cost side of the trade: a hit on a merely similar scene is a confident wrong answer."""
    tr, emb = _trace_with_return()
    loose = simulate_with_cache(tr, FixedIntervalPolicy(1.0), SemanticCache(threshold=0.0), emb,
                                warmup_frames=0)
    assert loose.false_hits > 0
    assert loose.answer_validity < 1.0


def test_cache_hit_staleness_uses_the_original_evidence_frame():
    """A hit must not reset staleness to zero — that would hide exactly what needs auditing."""
    tr, emb = _trace_with_return()
    r = simulate_with_cache(tr, FixedIntervalPolicy(1.0), SemanticCache(threshold=0.99), emb,
                            warmup_frames=0)
    assert r.staleness_ms_mean > 0


# --------------------------------------------------------------------------------------------
# 4. THE EXIT CRITERION — a verdict derived from the numbers
# --------------------------------------------------------------------------------------------
def _phase5() -> dict:
    path = RESULTS / "phase5_cache.json"
    if not path.exists():
        pytest.skip("Phase 5 cache sweep not run in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))["extra"]["phase5_cache"]


def test_the_sweep_measured_a_no_cache_baseline():
    """Without a baseline, 'the cache avoided N calls' means nothing."""
    rows = [r for r in _phase5()["rows"] if r["split"] == "held_out"]
    assert any(r["cache_threshold"] is None for r in rows), "no no-cache baseline was run"


def test_the_sweep_explored_thresholds_where_the_cache_actually_fires():
    """A curve of all zeros proves nothing; the sweep must reach the region where hits happen."""
    rows = [r for r in _phase5()["rows"] if r["split"] == "held_out"]
    assert any((r["calls_avoided_pct_mean"] or 0) > 0 for r in rows), (
        "every threshold produced zero hits — the sweep never reached the operating region"
    )


def test_no_threshold_gives_benefit_without_cost():
    """The finding this phase turns on, asserted rather than described."""
    rows = [r for r in _phase5()["rows"] if r["split"] == "held_out"]
    baseline = next(r for r in rows if r["cache_threshold"] is None)
    useful_and_free = [
        r for r in rows
        if r["cache_threshold"] is not None
        and r["calls_avoided_pct_mean"] > 0
        and r["answer_validity_mean"] >= baseline["answer_validity_mean"] - 1e-9
    ]
    assert not useful_and_free, (
        "a cache configuration DOES avoid calls at no accuracy cost — the Phase 5 recommendation "
        f"should be revisited: {useful_and_free}"
    )


def test_the_verdict_is_cut_and_is_explained():
    v = _phase5()["verdict"]
    assert v["recommendation"] in {"keep", "cut", "inconclusive"}
    assert v["recommendation"] == "cut", "the measurements support cutting; if this changes, say so"
    assert v["reason"], "a verdict without a reason is an opinion"
    assert v["root_cause"], "the verdict must explain WHY, not just report that it failed"


def test_the_root_cause_is_backed_by_a_measurement():
    """The explanation rests on similarity available at trigger time — check it was measured."""
    kq = _phase5()["key_quality"]
    sat = kq["similarity_at_trigger"]
    assert kq["mean_auc"] is not None
    assert sat["median_best_similarity"] is not None
    # The cache is asked at moments far less similar than the same/different separation band.
    assert sat["median_best_similarity"] < min(
        v["same_state_p10"] for v in kq["per_clip"].values()
    ), "if trigger-time similarity were inside the separation band, the cache could work"
