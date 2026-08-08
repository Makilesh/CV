"""Phase 6 exit criteria — and the anti-cheat suite.

`PROMPT.md`: *Write a test that deliberately attempts to access a future frame and asserts it fails.
This is the most likely place the project silently cheats.*

So these tests are adversarial: they try to obtain information from the future by every route the
API permits, and assert that each one is refused. A passing suite is the only evidence that
"streaming" means anything in this repo.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from peripheral.cli._args import load_config
from peripheral.eval.replay import (
    FutureFrameError,
    Query,
    QueryTimeline,
    ReplayClock,
    ReplaySource,
    uniform_queries,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
CLIPS = REPO_ROOT / "data" / "eval_clips"


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory) -> Path:
    """A 60-frame, 30 FPS clip (2 seconds) with frame-index markers baked in.

    Small on purpose: these tests must attack the invariant, not spend a minute waiting.
    """
    path = tmp_path_factory.mktemp("replay") / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(60):
        img = np.full((48, 64, 3), i * 4 % 255, np.uint8)
        writer.write(img)
    writer.release()
    assert path.exists()
    return path


# --------------------------------------------------------------------------------------------
# THE EXIT CRITERION — deliberately attempt to access a future frame
# --------------------------------------------------------------------------------------------
def test_reading_a_future_frame_raises(tiny_clip):
    """The headline anti-cheat test. Reach forward; be refused."""
    src = ReplaySource(tiny_clip, clock=ReplayClock(speed=1.0))
    src.open()
    try:
        # At t~0 only frame 0 is due. Frame 59 is 1.97 s away.
        with pytest.raises(FutureFrameError) as exc:
            src.frame_at(59)
        assert "future" in str(exc.value).lower()
        assert src.stats.future_access_attempts == 1
    finally:
        src.close()


def test_every_future_index_is_refused_at_t0(tiny_clip):
    """Not just the far future — anything not yet due."""
    src = ReplaySource(tiny_clip, clock=ReplayClock(speed=1.0))
    src.open()
    try:
        refused = 0
        for i in range(1, 60):
            if src.is_available(i):
                continue
            with pytest.raises(FutureFrameError):
                src.frame_at(i)
            refused += 1
        assert refused > 50, "almost every frame should still be in the future at t=0"
    finally:
        src.close()


def test_past_frames_remain_accessible(tiny_clip):
    """The gate must block the future without breaking legitimate access to the past."""
    src = ReplaySource(tiny_clip, clock=ReplayClock(speed=10.0))  # 10x so the test is quick
    src.open()
    try:
        for _ in range(12):
            src.read()
        frame = src.frame_at(3)
        assert frame.frame_id == 3
        assert frame.image.shape == (48, 64, 3)
    finally:
        src.close()


def test_read_never_outruns_the_wall_clock(tiny_clip):
    """Gate 1: the decoder cannot run ahead, so frames delivered <= elapsed * fps."""
    src = ReplaySource(tiny_clip, clock=ReplayClock(speed=1.0))
    src.open()
    try:
        t0 = time.perf_counter()
        for _ in range(10):
            src.read()
            audit = src.audit()
            assert audit["within_wall_clock"], audit
        elapsed = time.perf_counter() - t0
        # 10 frames at 30 FPS cannot arrive faster than ~0.3 s of real time.
        assert elapsed >= 0.25, f"10 frames arrived in {elapsed:.3f}s — replay is not paced"
    finally:
        src.close()


def test_a_batch_reader_would_fail_the_audit(tiny_clip):
    """Prove the audit has teeth by forging the violation it exists to catch.

    A batch reader decodes the whole clip immediately. Simulated here by inflating the delivered
    count without advancing the clock — exactly the signature of processing a file as fast as it
    decodes and calling it streaming.
    """
    src = ReplaySource(tiny_clip, clock=ReplayClock(speed=1.0))
    src.open()
    try:
        src.read()
        assert src.audit()["within_wall_clock"] is True
        src.stats.frames_delivered = 60          # what a batch job would report
        assert src.audit()["within_wall_clock"] is False
    finally:
        src.close()


# --------------------------------------------------------------------------------------------
# Gate 3 — answers audited against their evidence
# --------------------------------------------------------------------------------------------
def test_answering_with_future_evidence_raises():
    """The subtle cheat: legitimate frame access, but an answer attributed to an earlier query."""
    tl = QueryTimeline(queries=[Query(t=1.0, text="q", query_id="q0")], strict=True)
    with pytest.raises(FutureFrameError) as exc:
        tl.answer(tl.queries[0], answer="a", evidence_frame=60,
                  evidence_t=2.0, answered_at=1.0)
    assert "future" in str(exc.value).lower()
    assert tl.summary()["future_evidence_violations"] == 1


def test_answering_with_past_evidence_is_fine_and_measures_staleness():
    tl = QueryTimeline(queries=[Query(t=1.0, text="q", query_id="q0")], strict=True)
    rec = tl.answer(tl.queries[0], answer="a", evidence_frame=15, evidence_t=0.5, answered_at=1.0)
    assert rec.staleness_s == pytest.approx(0.5)
    assert tl.summary()["future_evidence_violations"] == 0


def test_evidence_exactly_at_the_query_time_is_allowed():
    """A frame that arrived on the query's own timestamp is present, not future."""
    tl = QueryTimeline(queries=[Query(t=1.0, text="q", query_id="q0")], strict=True)
    rec = tl.answer(tl.queries[0], answer="a", evidence_frame=30, evidence_t=1.0, answered_at=1.0)
    assert rec.staleness_s == pytest.approx(0.0)


def test_a_query_between_two_frames_must_use_the_earlier_frame():
    """The bug Gate 3 actually caught in a real run, as a regression test.

    A query at t=20.000 with frames at 19.987 and 20.020 must be answered from 19.987. Answering it
    after processing the 20.020 frame gives it 20 ms of future evidence. Our own clips hid this
    because 30 fps puts a frame exactly on every 2-second query; StreamingBench's frame rate did
    not cooperate, and the gate fired on sample 41.
    """
    tl = QueryTimeline(queries=[Query(t=20.0, text="q", query_id="q0")], strict=True)

    # Correct: evidence from the frame that had already arrived.
    rec = tl.answer(tl.queries[0], "a", evidence_frame=599, evidence_t=19.987, answered_at=20.0)
    assert rec.staleness_s == pytest.approx(0.013, abs=1e-6)

    # Wrong: the next frame, 20 ms in the query's future.
    tl2 = QueryTimeline(queries=[Query(t=20.0, text="q", query_id="q0")], strict=True)
    with pytest.raises(FutureFrameError, match="20.020"):
        tl2.answer(tl2.queries[0], "a", evidence_frame=601, evidence_t=20.020, answered_at=20.0)


def test_non_strict_timeline_records_violations_instead_of_raising():
    """Used only for post-hoc auditing of an external system; never for our own measured runs."""
    tl = QueryTimeline(queries=[Query(t=1.0, text="q", query_id="q0")], strict=False)
    tl.answer(tl.queries[0], answer="a", evidence_frame=60, evidence_t=2.0, answered_at=1.0)
    assert tl.summary()["future_evidence_violations"] == 1


def test_queries_become_due_in_order_and_only_once():
    tl = QueryTimeline(queries=uniform_queries(7.0, 2.0, "q"))
    assert [q.t for q in tl.queries] == [2.0, 4.0, 6.0]
    assert len(tl.due(1.0)) == 0
    assert len(tl.due(2.0)) == 1
    tl.answer(tl.due(2.0)[0], "a", 1, 1.0, 2.0)
    assert len(tl.due(2.0)) == 0, "an answered query must not come due again"
    assert len(tl.due(5.0)) == 1


def test_no_query_is_scheduled_at_or_past_the_clip_end():
    """The last frame of an N-frame clip is due before `duration`, so a query at exactly `duration`
    could never come due and would be miscounted as an unanswered failure."""
    assert uniform_queries(6.0, 2.0, "q")[-1].t == 4.0
    assert all(q.t < 6.0 for q in uniform_queries(6.0, 2.0, "q"))


# --------------------------------------------------------------------------------------------
# Clock semantics
# --------------------------------------------------------------------------------------------
def test_clock_rejects_non_positive_speed():
    with pytest.raises(ValueError):
        ReplayClock(speed=0.0)


def test_clock_must_be_started_before_use():
    with pytest.raises(RuntimeError):
        ReplayClock().elapsed()


def test_sleep_until_actually_waits():
    clock = ReplayClock(speed=1.0)
    clock.start()
    t0 = time.perf_counter()
    clock.sleep_until(0.15)
    assert time.perf_counter() - t0 >= 0.13, "sleep_until returned early — replay is not paced"


def test_replay_speed_is_recorded_so_it_cannot_be_hidden(tiny_clip):
    src = ReplaySource(tiny_clip, clock=ReplayClock(speed=5.0))
    src.open()
    try:
        assert src.describe()["replay_speed"] == 5.0
        assert src.audit()["replay_speed"] == 5.0
    finally:
        src.close()


def test_measured_runs_use_realtime_speed():
    cfg = load_config("config", [])
    assert cfg.phase6.replay_speed == 1.0, "a measured replay must run at wall-clock rate"


# --------------------------------------------------------------------------------------------
# Recorded Phase 6 artifacts
# --------------------------------------------------------------------------------------------
def _replay_docs() -> list[dict]:
    paths = sorted(RESULTS.glob("phase6_replay*.json"))
    if not paths:
        pytest.skip("Phase 6 replay runs not present in this checkout")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def test_recorded_replays_passed_the_wall_clock_audit():
    for doc in _replay_docs():
        audit = doc["extra"]["phase6_replay"]["wall_clock_audit"]
        assert audit["within_wall_clock"], f"{doc['run']['name']}: {audit}"
        assert audit["replay_speed"] == 1.0
        assert audit["future_access_attempts"] == 0


def test_recorded_replays_had_no_future_evidence_violations():
    for doc in _replay_docs():
        q = doc["extra"]["phase6_replay"]["queries"]
        assert q["future_evidence_violations"] == 0, q["violations"]
        assert q["n_answered"] > 0, "no queries were answered — the run proves nothing"


def test_ablation_table_exists_and_includes_the_required_arms():
    path = RESULTS / "phase6_ablations.json"
    if not path.exists():
        pytest.skip("ablations not run in this checkout")
    rows = json.loads(path.read_text(encoding="utf-8"))["extra"]["phase6_ablations"]["rows"]
    names = {r["name"] for r in rows}
    for required in ("full_system", "no_fast_tier", "fixed_interval_matched"):
        assert required in names, f"ablation {required!r} missing"
