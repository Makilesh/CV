"""Bounded queues with an explicit, configurable backpressure policy.

Invariant 7: **the capture thread never blocks on inference.** Video is a real-time source — if a
consumer is slower than the camera, something must be discarded, and the only question is *what*
and whether it is recorded.

Three policies, all measurable:

* ``drop_oldest`` (default) — evict the stalest queued frame to make room for the newest. Correct
  for video understanding: an answer about a 2-second-old frame is worth less than one about the
  current frame. `queue.Queue` cannot express this, which is why this is a `deque` + `Condition`.
* ``drop_newest``   — discard the arriving frame, keep the backlog. Preserves temporal order at the
  cost of answering about the past. Useful as a comparison, rarely what we want.
* ``block``         — the producer waits. **Violates invariant 7.** Kept only so Phase 1 can show
  quantitatively what it costs, rather than asserting that it is bad.

Every drop is recorded with its stage and reason, so a dropped frame is always a policy decision
and never an accident.
"""

from __future__ import annotations

import threading
from collections import deque
from enum import Enum
from typing import Any, Callable


class BackpressurePolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


class BoundedQueue:
    """A bounded queue whose overflow behaviour is chosen, recorded and testable."""

    def __init__(
        self,
        name: str,
        maxsize: int,
        policy: BackpressurePolicy | str = BackpressurePolicy.DROP_OLDEST,
        on_drop: Callable[[Any, str, str], None] | None = None,
        on_depth: Callable[[str, int], None] | None = None,
    ) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.name = name
        self.maxsize = int(maxsize)
        self.policy = BackpressurePolicy(policy)
        self._on_drop = on_drop
        self._on_depth = on_depth

        self._items: deque[Any] = deque()
        self._cond = threading.Condition()
        self._closed = False

        self.n_put = 0
        self.n_dropped = 0
        self.peak_depth = 0

    def put(self, item: Any, frame_id: int | None = None, timeout: float = 1.0) -> bool:
        """Enqueue. Returns False if the item was dropped by policy.

        Never raises on a full queue — overflow is expected operation for a real-time source, not
        an error condition.
        """
        with self._cond:
            if self._closed:
                return False

            if len(self._items) >= self.maxsize:
                if self.policy is BackpressurePolicy.DROP_NEWEST:
                    self._record_drop(frame_id, "queue_full_drop_newest")
                    return False

                if self.policy is BackpressurePolicy.DROP_OLDEST:
                    stale = self._items.popleft()
                    self._record_drop(getattr(stale, "frame_id", None), "queue_full_drop_oldest")

                elif self.policy is BackpressurePolicy.BLOCK:
                    # The producer stalls here. This is the invariant-7 violation, retained so its
                    # cost can be measured rather than assumed.
                    ok = self._cond.wait_for(
                        lambda: self._closed or len(self._items) < self.maxsize, timeout=timeout
                    )
                    if self._closed:
                        return False
                    if not ok:
                        self._record_drop(frame_id, "block_timeout")
                        return False

            self._items.append(item)
            self.n_put += 1
            depth = len(self._items)
            self.peak_depth = max(self.peak_depth, depth)
            if self._on_depth is not None:
                self._on_depth(self.name, depth)
            self._cond.notify()
            return True

    def get(self, timeout: float = 0.1) -> Any | None:
        """Dequeue, or None on timeout/close. Consumers poll so they can honour a stop signal."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._items or self._closed, timeout=timeout):
                return None
            if self._items:
                item = self._items.popleft()
                self._cond.notify()
                return item
            return None

    def drain_newest(self) -> Any | None:
        """Take the freshest item and discard the rest.

        For a consumer that only ever wants the current state of the world — which is what a VLM
        answering "what is happening now" wants. Discards are recorded.
        """
        with self._cond:
            if not self._items:
                return None
            newest = self._items.pop()
            while self._items:
                stale = self._items.popleft()
                self._record_drop(getattr(stale, "frame_id", None), "superseded_by_newer")
            self._cond.notify()
            return newest

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def depth(self) -> int:
        with self._cond:
            return len(self._items)

    def _record_drop(self, frame_id: int | None, reason: str) -> None:
        """Caller already holds the lock."""
        self.n_dropped += 1
        if self._on_drop is not None:
            self._on_drop(frame_id, self.name, reason)

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {
                "name": self.name,
                "maxsize": self.maxsize,
                "policy": self.policy.value,
                "n_put": self.n_put,
                "n_dropped": self.n_dropped,
                "peak_depth": self.peak_depth,
                "final_depth": len(self._items),
            }
