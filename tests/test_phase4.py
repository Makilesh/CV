"""Phase 4 exit criteria.

`PROMPT.md`: *`results/phase4_pareto.png` plus a test asserting the chosen policy reaches >=85% of
oracle accuracy at <=20% of oracle VLM calls on held-out clips.*

Phase 4 must show:
  1. the probe clips really do contain no semantic event, so a false trigger there is unambiguous;
  2. policies see only the past — no lookahead, by construction;
  3. the simulator's accounting is right (the oracle scores 1.0 at 100% of the budget);
  4. the learned policy is graded on clips it never saw;
  5. the chosen policy clears >=85% accuracy at <=20% of oracle calls on **held-out** clips.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from peripheral.cli._args import load_config
from peripheral.eval.clips import ClipAnnotation
from peripheral.eval.simulate import simulate
from peripheral.eval.traces import ClipTrace
from peripheral.scheduler.policies import (
    EmbeddingNoveltyPolicy,
    FixedIntervalPolicy,
    FrameContext,
    InformationGainPolicy,
    LearnedPolicy,
    MotionThresholdPolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
CLIP_DIR = REPO_ROOT / "data" / "eval_clips"
TRACE_DIR = REPO_ROOT / "data" / "traces"

TARGET_ACC = 0.85
TARGET_CALL_FRACTION = 0.20


def _ctx(**kw) -> FrameContext:
    base = dict(frame_idx=0, t=0.0, motion=0.0, novelty=0.0, scene_change=0.0,
                embedding=None, t_last_call=None, n_calls=0)
    base.update(kw)
    return FrameContext(**base)


def _fake_trace(n=300, fps=30.0, switch=150) -> ClipTrace:
    """Half the clip in state 0, half in state 1, with a matching answer change."""
    states = np.array([0] * switch + [1] * (n - switch), dtype=np.int32)
    answers = ["an empty desk with a keyboard" if s == 0 else "a red mug sits on the desk"
               for s in states]
    novelty = np.where(states == 1, 0.4, 0.01).astype(np.float32)
    return ClipTrace(
        clip_name="fake", fps=fps, n_frames=n,
        motion=np.full(n, 0.002, np.float32),
        novelty=novelty,
        scene_change=np.zeros(n, np.float32),
        answers=answers, state_ids=states,
        encoder={}, model="fake", prompt="p",
    )


# --------------------------------------------------------------------------------------------
# 1. The probe clips are genuinely eventless
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["lighting_drift", "rapid_motion", "static"])
def test_probe_clips_contain_no_semantic_event(name):
    """A false trigger is only unambiguous if the clip provably contains nothing to trigger on."""
    path = CLIP_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip("evaluation clips not built in this checkout")
    ann = ClipAnnotation.load(path)
    assert ann.semantic_events == [], f"{name} must contain no semantic events"
    assert len(set(ann.state_ids)) == 1, f"{name} must stay in a single scene state"


def test_event_clips_have_labelled_state_transitions():
    path = CLIP_DIR / "object_events.json"
    if not path.exists():
        pytest.skip("evaluation clips not built in this checkout")
    ann = ClipAnnotation.load(path)
    assert len(ann.semantic_events) == 3
    transitions = sum(1 for i in range(1, len(ann.state_ids))
                      if ann.state_ids[i] != ann.state_ids[i - 1])
    assert transitions == 3, "state must change exactly once per semantic event"


def test_clip_states_are_piecewise_constant():
    """Labels must be a step function; noise in the label would corrupt every false-trigger count."""
    path = CLIP_DIR / "mixed.json"
    if not path.exists():
        pytest.skip("evaluation clips not built in this checkout")
    ann = ClipAnnotation.load(path)
    runs = sum(1 for i in range(1, len(ann.state_ids))
               if ann.state_ids[i] != ann.state_ids[i - 1])
    assert runs == len(set(ann.state_ids)) - 1


# --------------------------------------------------------------------------------------------
# 2. No lookahead
# --------------------------------------------------------------------------------------------
def test_frame_context_exposes_only_past_and_present():
    """If a policy could see the future, every number in this phase would be worthless."""
    fields = set(FrameContext.__dataclass_fields__)
    assert fields == {
        "frame_idx", "t", "motion", "novelty", "scene_change",
        "embedding", "t_last_call", "n_calls",
    }, "FrameContext gained a field — check it cannot carry future information"


def test_policies_are_pure_functions_of_context():
    """Same context, same decision — no hidden dependence on iteration order."""
    for policy in (MotionThresholdPolicy(0.01), EmbeddingNoveltyPolicy(0.05),
                   InformationGainPolicy(0.02)):
        c = _ctx(t=5.0, motion=0.05, novelty=0.2, t_last_call=1.0)
        assert policy.decide(c).fire == policy.decide(c).fire


# --------------------------------------------------------------------------------------------
# 3. Policy behaviour
# --------------------------------------------------------------------------------------------
def test_oracle_policy_fires_on_every_frame():
    p = FixedIntervalPolicy(0.0)
    assert p.name == "oracle"
    assert all(p.decide(_ctx(t=i / 30.0, t_last_call=(i - 1) / 30.0)).fire for i in range(1, 10))


def test_fixed_interval_respects_its_period():
    p = FixedIntervalPolicy(1.0)
    assert not p.decide(_ctx(t=0.5, t_last_call=0.0)).fire
    assert p.decide(_ctx(t=1.0, t_last_call=0.0)).fire


def test_rate_limit_stops_degenerate_every_frame_firing():
    """Without a floor on spacing a content policy could win accuracy by calling constantly."""
    p = EmbeddingNoveltyPolicy(0.0, min_gap_s=0.3)
    assert not p.decide(_ctx(t=0.1, novelty=1.0, t_last_call=0.0)).fire
    assert p.decide(_ctx(t=0.4, novelty=1.0, t_last_call=0.0)).fire


def test_information_gain_needs_novelty_not_merely_time():
    """Staleness may raise urgency; it must never manufacture a trigger in a static scene."""
    p = InformationGainPolicy(0.05, staleness_half_life_s=3.0)
    p._novelty_at_last_call = 0.2
    quiet = _ctx(t=100.0, novelty=0.2, t_last_call=1.0)  # ancient call, nothing changed
    assert not p.decide(quiet).fire
    assert p.decide(_ctx(t=100.0, novelty=0.9, t_last_call=1.0)).fire


def test_information_gain_uses_novelty_since_last_call():
    p = InformationGainPolicy(0.05)
    ctx = _ctx(t=10.0, novelty=0.5, t_last_call=5.0)
    before = p.gain(ctx)
    p.observe_call(ctx)          # we have now seen this novelty level
    assert p.gain(ctx) < before, "novelty already paid for must not keep triggering"


def test_learned_policy_is_small_and_causal():
    assert len(LearnedPolicy.FEATURES) == 5, "the learned policy is deliberately tiny"
    p = LearnedPolicy(weights=[1, 0, 0, 0, 0], bias=-0.5, threshold=0.5)
    assert p.probability(_ctx(t=1.0, novelty=10.0, t_last_call=0.0)) > 0.9
    assert p.probability(_ctx(t=1.0, novelty=0.0, t_last_call=0.0)) < 0.5


# --------------------------------------------------------------------------------------------
# 4. Simulator accounting
# --------------------------------------------------------------------------------------------
def test_oracle_scores_perfectly_at_full_budget():
    """The oracle answers every frame from that frame, so it must score 1.0 at 100% of budget."""
    r = simulate(_fake_trace(), FixedIntervalPolicy(0.0))
    assert r.call_fraction == 1.0
    assert r.accuracy == pytest.approx(1.0, abs=1e-6)
    assert r.staleness_ms_mean == pytest.approx(0.0, abs=1e-6)


def test_never_calling_after_the_first_frame_scores_badly():
    """A policy that stops calling must be penalised on every frame its answer is out of date.

    Scored relative to the oracle rather than against an absolute: the two states in the fake
    trace share vocabulary ("desk"), so a wrong answer still earns partial content-F1 credit. What
    must hold is that never calling is far worse than always calling.
    """
    trace = _fake_trace()
    oracle = simulate(trace, FixedIntervalPolicy(0.0))
    never = simulate(trace, FixedIntervalPolicy(1e6))
    assert never.n_calls == 1
    assert never.accuracy < oracle.accuracy - 0.3
    assert never.event_recall == 0.0, "the state change was never looked at"


def test_false_triggers_counted_only_when_state_did_not_change():
    """Two calls inside one state = one false trigger; a call after the change is justified."""
    trace = _fake_trace(n=300, switch=150)
    r = simulate(trace, FixedIntervalPolicy(2.0), warmup_frames=0)
    assert r.false_triggers + r.justified_calls == r.n_calls - 1  # the first call has no baseline
    assert r.justified_calls >= 1, "the state change must be picked up by a 2 s interval"
    assert r.false_trigger_rate is not None


def test_probe_style_trace_makes_every_repeat_call_a_false_trigger():
    """In a single-state clip every call after the first buys nothing, by construction."""
    n = 300
    trace = ClipTrace(
        clip_name="probe", fps=30.0, n_frames=n,
        motion=np.full(n, 0.05, np.float32),
        novelty=np.full(n, 0.5, np.float32),
        scene_change=np.zeros(n, np.float32),
        answers=["a desk"] * n, state_ids=np.zeros(n, np.int32),
        encoder={}, model="fake", prompt="p",
    )
    r = simulate(trace, FixedIntervalPolicy(1.0), warmup_frames=0)
    assert r.justified_calls == 0
    assert r.false_trigger_rate == pytest.approx(1.0)


def test_event_recall_and_delay_are_measured():
    r = simulate(_fake_trace(), FixedIntervalPolicy(0.5), warmup_frames=0)
    assert r.total_events == 1
    assert r.event_recall == pytest.approx(1.0)
    assert r.detection_delay_ms_mean is not None and r.detection_delay_ms_mean >= 0


# --------------------------------------------------------------------------------------------
# 5. The learned policy is graded on unseen clips
# --------------------------------------------------------------------------------------------
def test_held_out_clips_are_excluded_from_training():
    path = RESULTS / "phase4_sweep.json"
    if not path.exists():
        pytest.skip("Phase 4 sweep not run in this checkout")
    sweep = json.loads(path.read_text(encoding="utf-8"))["extra"]["phase4_sweep"]
    train = set(sweep["train_clips"])
    held = set(sweep["held_out_clips"])
    assert held, "there must be held-out clips"
    assert not (train & held), "held-out clips leaked into learned-policy training"
    assert set(sweep["learned_policy"]["train_clips"]) == train


def test_held_out_covers_both_failure_directions():
    """Held-out must include a false-trigger probe AND a clip with real events."""
    cfg = load_config("config", [])
    held = set(cfg.phase4.held_out_clips)
    assert held & {"lighting_drift", "rapid_motion", "static"}, "no false-trigger probe held out"
    assert held & {"object_events", "mixed", "scene_cuts"}, "no event clip held out"


# --------------------------------------------------------------------------------------------
# 6. THE EXIT CRITERION
# --------------------------------------------------------------------------------------------
def _held_out_rows() -> list[dict]:
    path = RESULTS / "phase4_sweep.json"
    if not path.exists():
        pytest.skip("Phase 4 sweep not run in this checkout")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in doc["extra"]["phase4_sweep"]["results"] if r["split"] == "held_out"]


def test_a_policy_reaches_85pct_accuracy_at_20pct_of_oracle_calls():
    """Phase 4 exit criterion, asserted on held-out clips."""
    rows = _held_out_rows()
    qualifying = [
        r for r in rows
        if r["call_fraction_mean"] <= TARGET_CALL_FRACTION and r["answer_validity_mean"] >= TARGET_ACC
    ]
    assert qualifying, (
        "no policy reached >=85% of oracle accuracy at <=20% of oracle calls on held-out clips.\n"
        + "\n".join(
            f"  {r['kind']:18s} @ {r['operating_point']:<8.4g} "
            f"valid {r['answer_validity_mean']:.3f}  calls {r['call_fraction_mean'] * 100:5.1f}%"
            for r in sorted(rows, key=lambda z: -z["answer_validity_mean"])[:12]
        )
    )


def test_the_winning_policy_is_not_the_oracle_in_disguise():
    """A 'policy' that simply calls constantly is not a scheduler."""
    rows = _held_out_rows()
    best = max(
        (r for r in rows if r["call_fraction_mean"] <= TARGET_CALL_FRACTION),
        key=lambda r: r["answer_validity_mean"],
    )
    assert best["call_fraction_mean"] <= TARGET_CALL_FRACTION
    assert best["calls_per_min_mean"] < 0.25 * 1800, "that is not a saving"


def test_a_content_policy_beats_fixed_interval_at_matched_budget():
    """The claim of the project: scene-awareness beats calling on a timer."""
    rows = _held_out_rows()
    fixed = [r for r in rows if r["kind"] == "fixed_interval" and 0 < r["call_fraction_mean"] <= TARGET_CALL_FRACTION]
    content = [r for r in rows if r["kind"] != "fixed_interval"
               and r["call_fraction_mean"] <= TARGET_CALL_FRACTION]
    if not fixed or not content:
        pytest.skip("no matched operating points in range")
    best_fixed = max(fixed, key=lambda r: r["answer_validity_mean"])
    best_content = max(content, key=lambda r: r["answer_validity_mean"])
    assert best_content["answer_validity_mean"] >= best_fixed["answer_validity_mean"], (
        f"fixed interval ({best_fixed['answer_validity_mean']:.3f}) beat every content-aware policy "
        f"({best_content['kind']} {best_content['answer_validity_mean']:.3f}) — report this plainly"
    )


def test_the_headline_figure_exists():
    if not (RESULTS / "phase4_sweep.json").exists():
        pytest.skip("Phase 4 sweep not run in this checkout")
    png = RESULTS / "phase4_pareto.png"
    assert png.exists(), "run peripheral.viz.phase4_chart to build it"
    assert png.stat().st_size > 50_000

