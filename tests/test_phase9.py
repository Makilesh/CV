"""Phase 9 exit criteria — the regime where content-aware scheduling beats a timer.

Phase 8 failed and blamed event sparsity. Phase 9 swept sparsity across an order of magnitude, found
the timer still winning, and located the real variable: **background stability**. These tests pin
that finding against `results/phase9_sparsity.json` so it cannot silently rot, and pin the two
experiment-design bugs that produced clean-looking wrong numbers on the way there.

The exit criterion for the phase is `test_exit_criterion_content_beats_timer_on_a_static_background`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from peripheral.eval.clips import RECIPES

RESULTS = Path(__file__).resolve().parents[1] / "results" / "phase9_sparsity.json"

# The bar the sweep itself enforced. Repeated here so a change to one is visible against the other.
VALIDITY_BAR = 0.99


def _rows() -> dict[str, dict]:
    if not RESULTS.exists():
        pytest.skip(f"{RESULTS.name} not present — run peripheral.cli.sparsity_study")
    doc = json.loads(RESULTS.read_text(encoding="utf-8"))
    if doc["run"]["status"] != "completed":
        pytest.skip(f"sparsity study did not complete: {doc['run']['status']}")
    return {r["recipe"]: r for r in doc["extra"]["phase9_sparsity"]["rows"]}


def _calls(row: dict, policy: str) -> float | None:
    best = row["best"].get(policy)
    return None if best is None else best["calls_per_min"]


# -- the exit criterion ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recipe", "min_speedup"),
    [("frozen_sparse_2", 17.0), ("frozen_sparse_4", 17.0), ("frozen_sparse_8", 17.0)],
)
def test_exit_criterion_content_beats_timer_on_a_static_background(recipe, min_speedup):
    """The claim, in the regime it was always about.

    On a static-background camera the best content policy must hold the validity bar at a large
    multiple fewer VLM calls than a fixed-interval timer holding the same bar. Measured speedups are
    50-67x (motion) and 17-21x (embedding); the bar is set at the weaker of the two so the test
    passes on whichever policy wins, and so it is a floor rather than a transcription of the run.
    """
    row = _rows()[recipe]
    timer = _calls(row, "fixed_interval")
    assert timer is not None, "the timer must have a valid operating point to compare against"

    content = {
        k: _calls(row, k)
        for k in ("embedding_novelty", "motion_threshold", "adaptive_novelty")
        if _calls(row, k)
    }
    assert content, f"{recipe}: no content policy held validity >= {VALIDITY_BAR}"

    best_policy = min(content, key=lambda k: content[k])
    speedup = timer / content[best_policy]
    assert speedup >= min_speedup, (
        f"{recipe}: best content policy {best_policy} at {content[best_policy]:.2f} calls/min "
        f"vs timer {timer:.2f} — only {speedup:.1f}x, expected >= {min_speedup}x"
    )
    assert row["best"][best_policy]["event_recall"] == 1.0, "cheaper by missing events is not cheaper"


def test_every_reported_operating_point_actually_held_the_validity_bar():
    """A cheap call rate means nothing if it was bought below the bar the sweep claimed to enforce."""
    for recipe, row in _rows().items():
        for policy, best in row["best"].items():
            assert best["validity"] >= VALIDITY_BAR, (
                f"{recipe}/{policy}: reported at validity {best['validity']} < {VALIDITY_BAR}"
            )


# -- the finding that corrects Phase 8 ---------------------------------------------------------


def test_sparsity_alone_does_not_let_content_beat_the_timer():
    """Phase 8's prescription, refuted.

    It said: make events sparser and the timer must start wasting calls. On the live-background
    ladder it never does, at any sparsity from one event per 19 s to one per 150 s. This test fails
    the day that stops being true — which would be a real finding, not a regression.
    """
    rows = _rows()
    live = ["sparse_2", "sparse_4", "sparse_8", "sparse_16"]
    losers = []
    for recipe in live:
        row = rows[recipe]
        timer = _calls(row, "fixed_interval")
        content = [c for k in ("embedding_novelty", "motion_threshold", "adaptive_novelty")
                   if (c := _calls(row, k))]
        best = min(content) if content else float("inf")
        losers.append(best >= timer / 2.0)

    assert all(losers), (
        "a content policy now beats the timer by >2x on live-background footage at some sparsity — "
        "Phase 9's conclusion (background stability, not sparsity, is the variable) needs revisiting"
    )


def test_the_timer_oversampling_factor_is_flat_across_the_sparsity_ladder():
    """Why sparsity was the wrong variable: the timer's winning interval scales with the event rate.

    Phase 8 reasoned that sparser events would drive the timer's oversampling up until it had to
    waste calls. It does not move — which is exactly why sweeping sparsity changed nothing.
    """
    rows = _rows()
    factors = [
        rows[r]["timer_oversampling_x"]
        for r in ("sparse_2", "sparse_4", "sparse_8", "sparse_16")
        if rows[r]["timer_oversampling_x"] is not None
    ]
    assert len(factors) >= 3
    assert max(factors) / min(factors) <= 2.5, (
        f"oversampling now varies {min(factors)}-{max(factors)}x across the ladder; Phase 9 measured "
        "it flat at 75x, which is the basis for rejecting sparsity as the limiting variable"
    )


def test_lighting_drift_breaks_motion_but_not_the_embedding():
    """The one result where the learned fast tier earns its 2.95 ms.

    A frozen background makes plain pixel differencing optimal (false-trigger rate 0.000). Add a slow
    gamma drift and it collapses, while the embedding does not notice. This is Phase 2's
    `semantic/motion` ratio of 0.42 reproduced at clip scale.
    """
    rows = _rows()
    for still, drift in (("frozen_sparse_4", "frozen_drift_sparse_4"),
                         ("frozen_sparse_8", "frozen_drift_sparse_8")):
        m_still, m_drift = _calls(rows[still], "motion_threshold"), _calls(rows[drift], "motion_threshold")
        e_still, e_drift = _calls(rows[still], "embedding_novelty"), _calls(rows[drift], "embedding_novelty")
        assert None not in (m_still, m_drift, e_still, e_drift)

        assert m_drift >= 5.0 * m_still, (
            f"{drift}: motion cost only went {m_still:.2f} -> {m_drift:.2f} under lighting drift; "
            "measured 8.9-16.6x, and the two-tier argument rests on it degrading"
        )
        assert e_drift <= 1.25 * e_still, (
            f"{drift}: embedding cost went {e_still:.2f} -> {e_drift:.2f} under drift; it is supposed "
            "to be near brightness-invariant"
        )
        assert e_drift < m_drift, "under drift the embedding must be the cheaper signal, or it is not earning its latency"


def test_motion_is_the_better_signal_when_the_light_holds_still():
    """State the trade honestly in both directions, so the fast tier is not oversold.

    On a genuinely static scene, pixel differencing is not merely competitive — it is optimal, at a
    false-trigger rate of exactly zero. If this ever fails, RESULTS.md section 9 is overselling the
    encoder and must be corrected.
    """
    rows = _rows()
    for recipe in ("frozen_sparse_2", "frozen_sparse_4", "frozen_sparse_8"):
        motion = rows[recipe]["best"]["motion_threshold"]
        assert motion["false_trigger_rate"] == 0.0, (
            f"{recipe}: motion false-trigger rate is {motion['false_trigger_rate']}, not 0.0"
        )
        assert motion["calls_per_min"] <= _calls(rows[recipe], "embedding_novelty")


# -- the two experiment-design bugs -------------------------------------------------------------


def test_sparse_event_times_are_not_evenly_spaced():
    """Regression for the phase-lock artifact.

    Evenly spaced events let a fixed-interval timer whose period divides the spacing align with every
    one of them: `sparse_4` reported a timer cost of 1.0 calls/min instead of the honest 60.0, a 60x
    error with nothing logged. Randomised spacing is what makes the timer baseline fair, so it is
    tested rather than trusted.
    """
    make = RECIPES["sparse_8"]
    _fn, events, _purpose = make(300.0)
    times = np.array(sorted(e.t_start for e in events if e.semantic), dtype=float)
    assert len(times) >= 4, "sparse_8 should annotate its semantic events"

    gaps = np.diff(times)
    assert gaps.std() > 0.05 * gaps.mean(), (
        f"event gaps are near-uniform (std {gaps.std():.3f}s on mean {gaps.mean():.3f}s) — a timer "
        "can phase-lock to them and the measured baseline will be far too cheap"
    )


def test_the_study_run_was_not_silently_truncated():
    """Regression for the `duration_s` name collision.

    The runner stored a clip duration under the attribute holding the run deadline; a 3,600 s run
    stopped after 332 s and still wrote `status: completed`. A short run is legitimate *only* when
    the work actually ran out, which the runner records as an explicit note.
    """
    doc = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else pytest.skip("no results")
    run = doc["run"]
    if run["duration_actual_s"] < 0.9 * run["duration_requested_s"]:
        assert any("ended early" in n for n in run.get("notes", [])), (
            f"run stopped at {run['duration_actual_s']:.0f}s of {run['duration_requested_s']:.0f}s "
            "with no note explaining why — this is the silent-truncation bug"
        )
    n_rows = len(doc["extra"]["phase9_sparsity"]["rows"])
    assert n_rows == len(doc["config"]["phase9"]["recipes"]), (
        f"{n_rows} rows for {len(doc['config']['phase9']['recipes'])} configured recipes — "
        "the run ended before sweeping every clip"
    )


def test_no_fabricated_accuracy_is_reported():
    """No VLM ran, so text-agreement accuracy is unmeasured and must serialise as null, never 0.0."""
    doc = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else pytest.skip("no results")
    assert doc["metrics"].get("accuracy_vs_oracle") is None
    assert "no_oracle" in doc["extra"]["phase9_sparsity"]
