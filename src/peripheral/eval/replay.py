"""Wall-clock replay with **hard** no-future-frames enforcement.

`PROMPT.md` invariant 1, and its warning: *this is the single most likely way this project cheats
without noticing.* So the enforcement here is structural, not procedural. Three independent gates,
each of which would catch a violation the others might miss:

**Gate 1 — the decoder never runs ahead.** `ReplaySource` decodes frame *n* only once the replay
clock has passed frame *n*'s presentation timestamp. A future frame is not withheld from the
consumer; it does not exist in memory. There is nothing to accidentally read.

**Gate 2 — explicit access is checked.** `frame_at(i)` exists specifically so that a caller *can*
try to reach forward, and raises `FutureFrameError` when it does. Without a reachable API there
would be nothing for the anti-cheat test to attack, and an untested invariant is a hope.

**Gate 3 — answers are audited against their evidence.** Every answer carries the frame it was
derived from. `QueryTimeline.answer()` refuses an answer whose evidence timestamp is later than the
query time. This is the gate that catches the subtle version of the bug: a pipeline that legitimately
holds a frame, but attributes an answer to a query that was asked *before* that frame arrived.

A run may also be checked after the fact: `frames_delivered <= elapsed * fps + 1` must hold at all
times, because a source that delivered more frames than wall clock allows has run ahead by
definition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from ..capture.sources import Frame, FrameSource
from ..telemetry.clock import now


class FutureFrameError(RuntimeError):
    """Raised when something tries to read a frame that has not happened yet.

    This exception existing and being raised is a Phase 6 exit criterion. If it is ever caught and
    ignored, the project has stopped measuring streaming and started measuring batch processing.
    """


class ReplayClock:
    """Monotonic clock anchored at replay start.

    `speed` scales replay rate. It exists for tests, which cannot afford to spend 24 seconds of real
    time per clip — but **every measured run uses speed 1.0**, and the metrics file records the
    value so a non-realtime run can never be mistaken for a realtime one.
    """

    def __init__(self, speed: float = 1.0) -> None:
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = float(speed)
        self._t0: float | None = None

    def start(self) -> float:
        self._t0 = now()
        return self._t0

    @property
    def started(self) -> bool:
        return self._t0 is not None

    def elapsed(self) -> float:
        """Replay-time seconds since start."""
        if self._t0 is None:
            raise RuntimeError("clock not started")
        return (now() - self._t0) * self.speed

    def sleep_until(self, replay_t: float) -> None:
        """Block until replay time reaches `replay_t`.

        Sleeping is the point. A replay that does not wait is a batch job.
        """
        if self._t0 is None:
            raise RuntimeError("clock not started")
        target_wall = self._t0 + replay_t / self.speed
        remaining = target_wall - now()
        if remaining <= 0:
            return
        # time.sleep on Windows is coarse (~1-15 ms); spin the last stretch so pacing error does
        # not become measured jitter.
        if remaining > 0.002:
            time.sleep(remaining - 0.002)
        while now() < target_wall:
            pass


@dataclass
class ReplayStats:
    frames_delivered: int = 0
    frames_late: int = 0            # delivered after their due time by more than tolerance
    max_lateness_ms: float = 0.0
    future_access_attempts: int = 0
    elapsed_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ReplaySource(FrameSource):
    """A clip streamed at wall-clock rate, with future frames physically absent.

    Not a subclass of `FileSource`: that one is paced but permissive, and Phase 6 needs a source
    whose contract is enforcement rather than politeness.
    """

    name = "replay"

    def __init__(
        self,
        path: str | Path,
        clock: ReplayClock | None = None,
        fps: float | None = None,
        late_tolerance_ms: float = 50.0,
        strict: bool = True,
    ) -> None:
        self.path = Path(path)
        self.clock = clock or ReplayClock()
        self._fps_override = fps
        self.late_tolerance_ms = float(late_tolerance_ms)
        self.strict = bool(strict)

        self._cap: cv2.VideoCapture | None = None
        self._next_index = 0
        self.fps: float = 30.0
        self.n_frames_total: int = 0
        self.stats = ReplayStats()

    # -- lifecycle -----------------------------------------------------------------------
    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"clip not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open clip: {self.path}")
        declared = cap.get(cv2.CAP_PROP_FPS)
        self.fps = float(self._fps_override or (declared if declared and declared > 0 else 30.0))
        self.n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._cap = cap
        self._next_index = 0
        self.stats = ReplayStats()
        if not self.clock.started:
            self.clock.start()

    def close(self) -> None:
        if self._cap is not None:
            self.stats.elapsed_s = round(self.clock.elapsed(), 4) if self.clock.started else 0.0
            self._cap.release()
            self._cap = None

    # -- the contract --------------------------------------------------------------------
    def due_time(self, index: int) -> float:
        """Replay time at which frame `index` becomes available."""
        return index / self.fps

    def is_available(self, index: int) -> bool:
        return self.due_time(index) <= self.clock.elapsed()

    def read(self) -> Frame | None:
        """Return the next frame, **waiting until it is due**.

        Blocking here is the enforcement: the consumer cannot obtain frame *n* before wall clock
        reaches *n/fps*, because this call does not return until then.
        """
        if self._cap is None:
            raise RuntimeError("read() before open()")
        if self.n_frames_total and self._next_index >= self.n_frames_total:
            return None

        due = self.due_time(self._next_index)
        self.clock.sleep_until(due)

        ok, image = self._cap.read()
        t_capture = now()
        if not ok:
            return None

        lateness_ms = (self.clock.elapsed() - due) * 1000.0
        if lateness_ms > self.late_tolerance_ms:
            self.stats.frames_late += 1
        self.stats.max_lateness_ms = max(self.stats.max_lateness_ms, lateness_ms)

        frame = Frame(frame_id=self._next_index, t_capture=t_capture, image=image)
        self._next_index += 1
        self.stats.frames_delivered += 1
        return frame

    def frame_at(self, index: int) -> Frame:
        """Random access, **gated on the clock**.

        This method exists so the no-future-frames invariant is attackable and therefore testable.
        It is not used by the pipeline.
        """
        if self._cap is None:
            raise RuntimeError("frame_at() before open()")
        if index < 0:
            raise IndexError(f"negative frame index {index}")

        if not self.is_available(index):
            self.stats.future_access_attempts += 1
            raise FutureFrameError(
                f"frame {index} is due at t={self.due_time(index):.3f}s but replay time is "
                f"{self.clock.elapsed():.3f}s. Reading it would give the system information from "
                f"the future. This is invariant 1 and it is not negotiable."
            )

        pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self._cap.read()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        if not ok:
            raise IndexError(f"frame {index} could not be decoded")
        return Frame(frame_id=index, t_capture=now(), image=image)

    def audit(self) -> dict[str, Any]:
        """Post-hoc check: did the source ever deliver more frames than wall clock allowed?"""
        elapsed = self.clock.elapsed() if self.clock.started else 0.0
        allowed = elapsed * self.fps + 1  # +1 for the frame due at exactly t=0
        ok = self.stats.frames_delivered <= allowed
        return {
            "frames_delivered": self.stats.frames_delivered,
            "elapsed_s": round(elapsed, 4),
            "max_frames_allowed_by_clock": round(allowed, 2),
            "within_wall_clock": bool(ok),
            "replay_speed": self.clock.speed,
            "frames_late": self.stats.frames_late,
            "max_lateness_ms": round(self.stats.max_lateness_ms, 2),
            "future_access_attempts": self.stats.future_access_attempts,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.name,
            "path": str(self.path),
            "fps": self.fps,
            "n_frames_total": self.n_frames_total,
            "replay_speed": self.clock.speed,
            "strict": self.strict,
        }


@dataclass
class Query:
    """A question asked at a specific moment in replay time."""

    t: float
    text: str
    query_id: str = ""


@dataclass
class AnsweredQuery:
    query: Query
    answer: str | None
    evidence_frame: int | None
    evidence_t: float | None
    staleness_s: float | None
    answered_at: float


@dataclass
class QueryTimeline:
    """Timestamped queries, answered only from evidence that already existed.

    `answer()` is the third enforcement gate. A pipeline can be perfectly well-behaved about frame
    access and still cheat here, by attributing an answer derived from frame 300 to a query asked at
    frame 200's timestamp. This refuses that.
    """

    queries: list[Query] = field(default_factory=list)
    answered: list[AnsweredQuery] = field(default_factory=list)
    strict: bool = True
    violations: list[str] = field(default_factory=list)

    def due(self, replay_t: float) -> list[Query]:
        """Queries whose time has come and which have not been answered yet."""
        done = {a.query.query_id for a in self.answered}
        return [q for q in self.queries if q.t <= replay_t and q.query_id not in done]

    def answer(
        self,
        query: Query,
        answer: str | None,
        evidence_frame: int | None,
        evidence_t: float | None,
        answered_at: float,
    ) -> AnsweredQuery:
        if evidence_t is not None and evidence_t > query.t + 1e-9:
            msg = (
                f"query {query.query_id!r} asked at t={query.t:.3f}s was answered using evidence "
                f"from t={evidence_t:.3f}s — {(evidence_t - query.t) * 1000:.1f} ms in its future"
            )
            self.violations.append(msg)
            if self.strict:
                raise FutureFrameError(msg)

        rec = AnsweredQuery(
            query=query,
            answer=answer,
            evidence_frame=evidence_frame,
            evidence_t=evidence_t,
            staleness_s=None if evidence_t is None else round(query.t - evidence_t, 4),
            answered_at=round(answered_at, 4),
        )
        self.answered.append(rec)
        return rec

    def summary(self) -> dict[str, Any]:
        stale = [a.staleness_s for a in self.answered if a.staleness_s is not None]
        return {
            "n_queries": len(self.queries),
            "n_answered": len(self.answered),
            "n_unanswered": len(self.queries) - len(self.answered),
            "staleness_s_mean": round(float(np.mean(stale)), 4) if stale else None,
            "staleness_s_max": round(float(np.max(stale)), 4) if stale else None,
            "future_evidence_violations": len(self.violations),
            "violations": self.violations,
        }


def uniform_queries(duration_s: float, every_s: float, text: str) -> list[Query]:
    """A query every `every_s` seconds — a simple, reproducible interrogation schedule.

    The last frame of an N-frame clip is due at (N-1)/fps, strictly before `duration_s`, so a query
    scheduled at exactly `duration_s` could never come due and would be reported as unanswered.
    That is a scheduling artifact rather than a system failure, so it is excluded.
    """
    out: list[Query] = []
    i = 0
    while (i + 1) * every_s < duration_s - 1e-6:
        out.append(Query(t=(i + 1) * every_s, text=text, query_id=f"q{i}"))
        i += 1
    return out
