"""Frame differencing, scene-change scoring and novelty against a rolling reference.

These are the signals the Phase 4 scheduler decides on. Phase 2's job is to produce them inside a
~10 ms per-frame budget and to make their behaviour measurable.

The distinction that matters for this project:

* **motion** — how much the pixels moved. Cheap, and fooled by lighting.
* **novelty** — how far the embedding is from a rolling reference of recent scene state. This is
  what should stay flat under a lighting change and spike on a semantic one.

Phase 4's headline failure mode is a false trigger on lighting drift with no semantic event, so
these two signals are deliberately kept separate rather than fused into one number here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class FrameScores:
    """Everything the fast tier knows about one frame."""

    frame_id: int
    motion: float           # mean absolute pixel difference vs previous frame, 0..1
    novelty: float          # cosine distance from the rolling reference, 0..2
    scene_change: float     # novelty against the immediately preceding frame, 0..2
    embedding: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class MotionScorer:
    """Mean absolute difference on a downsampled grayscale frame.

    Downsampled because full-resolution differencing costs more than it tells us: at 640x480 it is
    ~1 ms of pure memory traffic to learn something a 64x64 thumbnail already says.
    """

    def __init__(self, size: int = 64, blur: int = 3) -> None:
        self.size = int(size)
        self.blur = int(blur)
        self._prev: np.ndarray | None = None

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self.blur > 1:
            # Sensor noise at short exposure reads as motion. Blur first or every frame is "motion".
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        return gray

    def score(self, bgr: np.ndarray) -> float:
        cur = self._prep(bgr)
        if self._prev is None:
            self._prev = cur
            return 0.0
        diff = cv2.absdiff(cur, self._prev)
        self._prev = cur
        return float(diff.mean()) / 255.0

    def reset(self) -> None:
        self._prev = None


class RollingReference:
    """An exponentially weighted mean embedding of recent scene state.

    Novelty is measured against this rather than against the previous frame, so that a slow drift
    (a person gradually leaning, the light dimming) accumulates instead of being invisible one
    frame at a time — and so that a single noisy frame does not by itself look novel.

    `half_life_s` is the time for an old observation's weight to halve. It is expressed in seconds
    rather than frames so the behaviour does not silently change with capture rate.
    """

    def __init__(self, half_life_s: float = 2.0, min_updates: int = 5) -> None:
        self.half_life_s = float(half_life_s)
        self.min_updates = int(min_updates)
        self._ref: np.ndarray | None = None
        self._t_last: float | None = None
        self.n_updates = 0

    def update(self, embedding: np.ndarray, t: float) -> None:
        if self._ref is None:
            self._ref = embedding.astype(np.float32).copy()
            self._t_last = t
            self.n_updates = 1
            return
        dt = max(0.0, t - (self._t_last or t))
        alpha = 1.0 - 0.5 ** (dt / self.half_life_s) if self.half_life_s > 0 else 1.0
        self._ref = (1.0 - alpha) * self._ref + alpha * embedding
        n = np.linalg.norm(self._ref)
        if n > 0:
            self._ref /= n
        self._t_last = t
        self.n_updates += 1

    def distance(self, embedding: np.ndarray) -> float:
        """Cosine distance in [0, 2]. Returns 0.0 until the reference has settled."""
        if self._ref is None or self.n_updates < self.min_updates:
            return 0.0
        return float(1.0 - np.dot(self._ref, embedding))

    @property
    def ready(self) -> bool:
        return self._ref is not None and self.n_updates >= self.min_updates

    def reset(self) -> None:
        self._ref = None
        self._t_last = None
        self.n_updates = 0


class FastTier:
    """Motion + embedding + novelty + scene change, per frame, inside the budget."""

    def __init__(
        self,
        encoder: Any,
        motion_size: int = 64,
        half_life_s: float = 2.0,
        keep_embedding: bool = False,
        history: int = 0,
    ) -> None:
        self.encoder = encoder
        self.motion = MotionScorer(size=motion_size)
        self.reference = RollingReference(half_life_s=half_life_s)
        self.keep_embedding = keep_embedding
        self._prev_emb: np.ndarray | None = None
        self.history: deque[FrameScores] = deque(maxlen=history) if history else deque(maxlen=0)

    def process(self, frame: Any) -> FrameScores:
        motion = self.motion.score(frame.image)
        emb = self.encoder.encode(frame.image)

        scene_change = (
            float(1.0 - np.dot(self._prev_emb, emb)) if self._prev_emb is not None else 0.0
        )
        novelty = self.reference.distance(emb)

        # STREAM time, not capture time. The reference half-life is in seconds, so using the moment
        # we happened to read the frame makes novelty depend on how fast the consumer runs — during
        # Phase 4 trace building the VLM took ~500 ms/frame and novelty came out ~4x too small.
        # See Frame.t_presentation.
        self.reference.update(emb, frame.t_stream)
        self._prev_emb = emb

        scores = FrameScores(
            frame_id=frame.frame_id,
            motion=motion,
            novelty=novelty,
            scene_change=scene_change,
            embedding=emb if self.keep_embedding else None,
        )
        if self.history.maxlen:
            self.history.append(scores)
        return scores

    def reset(self) -> None:
        self.motion.reset()
        self.reference.reset()
        self._prev_emb = None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two unit vectors, clamped to [0, 2] for float error."""
    return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))
