"""Trigger policies — the research contribution.

One interface, six implementations, all deciding the same question on every frame: **is spending a
VLM invocation worth it right now?**

    fixed_interval     every frame / 1 Hz / 0.5 Hz / 0.2 Hz — the baselines to beat
    motion_threshold   fire when pixels moved. The thing a scheduler must beat to be a scheduler.
    embedding_novelty  fire when the embedding leaves a rolling reference of recent scene state
    information_gain   novelty discounted by how stale the current answer is and how recently we
                       last called — spend a call when it buys the most, not merely when something
                       moved
    learned            a small logistic model over the fast-tier features, fit on training clips

Every policy sees only the past: `decide()` receives the current frame's scores and whatever state
the policy has accumulated. There is no lookahead by construction, not by convention.

Each policy exposes `operating_points()` so a sweep can walk its range without the sweep needing to
know what its knob means.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class Decision:
    fire: bool
    score: float
    reason: str


class TriggerPolicy(ABC):
    """Decide, per frame, whether to spend a VLM call."""

    name: str = "policy"

    def reset(self) -> None:
        """Clear per-clip state. Called before each clip so clips cannot leak into each other."""

    @abstractmethod
    def decide(self, ctx: "FrameContext") -> Decision:
        ...

    def observe_call(self, ctx: "FrameContext") -> None:
        """Called after a decision that fired, so the policy can update its own bookkeeping."""

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name}


@dataclass
class FrameContext:
    """Everything a policy is allowed to see. Strictly the present and the past."""

    frame_idx: int
    t: float                       # seconds since clip start
    motion: float                  # fast-tier pixel-difference score
    novelty: float                 # fast-tier embedding distance from rolling reference
    scene_change: float            # fast-tier embedding distance from the previous frame
    embedding: np.ndarray | None
    t_last_call: float | None      # when we last spent a call
    n_calls: int

    @property
    def since_last_call(self) -> float:
        return float("inf") if self.t_last_call is None else self.t - self.t_last_call


class FixedIntervalPolicy(TriggerPolicy):
    """Call every `period_s` seconds regardless of content. The baseline the scheduler must beat.

    `period_s = 0` is the per-frame oracle.
    """

    def __init__(self, period_s: float) -> None:
        self.period_s = float(period_s)
        self.name = "oracle" if period_s <= 0 else f"fixed_{_hz(period_s)}"

    def decide(self, ctx: FrameContext) -> Decision:
        if self.period_s <= 0:
            return Decision(True, 1.0, "per-frame oracle")
        fire = ctx.since_last_call >= self.period_s
        return Decision(fire, 1.0 if fire else 0.0, f"interval {self.period_s:g}s")

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name, "period_s": self.period_s}

    @staticmethod
    def operating_points() -> list[float]:
        # 0 is the oracle; the rest span every-frame to one call per 10 s.
        return [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


class MotionThresholdPolicy(TriggerPolicy):
    """Fire when pixel motion exceeds a threshold.

    Included precisely because it is the thing the project claims to beat. If it wins, that is the
    result and it goes in the paper.
    """

    name = "motion_threshold"

    def __init__(self, threshold: float, min_gap_s: float = 0.3) -> None:
        self.threshold = float(threshold)
        self.min_gap_s = float(min_gap_s)

    def decide(self, ctx: FrameContext) -> Decision:
        if ctx.since_last_call < self.min_gap_s:
            return Decision(False, ctx.motion, "rate limited")
        fire = ctx.motion >= self.threshold or ctx.t_last_call is None
        return Decision(fire, ctx.motion, f"motion {ctx.motion:.4f} vs {self.threshold:.4f}")

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name, "threshold": self.threshold, "min_gap_s": self.min_gap_s}

    @staticmethod
    def operating_points() -> list[float]:
        return [0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06, 0.12]


class EmbeddingNoveltyPolicy(TriggerPolicy):
    """Fire when the embedding has left the rolling reference of recent scene state.

    The simple, obvious scheduler. `PROMPT.md` is explicit that if this beats the learned policy,
    that is a stronger finding than the reverse.
    """

    name = "embedding_novelty"

    def __init__(self, threshold: float, min_gap_s: float = 0.3) -> None:
        self.threshold = float(threshold)
        self.min_gap_s = float(min_gap_s)

    def decide(self, ctx: FrameContext) -> Decision:
        if ctx.t_last_call is None:
            return Decision(True, ctx.novelty, "first frame")
        if ctx.since_last_call < self.min_gap_s:
            return Decision(False, ctx.novelty, "rate limited")
        fire = ctx.novelty >= self.threshold
        return Decision(fire, ctx.novelty, f"novelty {ctx.novelty:.4f} vs {self.threshold:.4f}")

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name, "threshold": self.threshold, "min_gap_s": self.min_gap_s}

    @staticmethod
    def operating_points() -> list[float]:
        return [0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35, 0.6]


class InformationGainPolicy(TriggerPolicy):
    """Fire when a call buys the most, not merely when something moved.

    Novelty says *how different the world looks now*. Staleness says *how out of date our answer
    is*. Their product is a crude expected-information-gain: a scene that changed long ago and has
    not been re-queried is worth a call even if this instant is quiet, and a scene that changed one
    frame after we looked is not worth a second call yet.

    The saturating staleness term is what stops a static scene from eventually triggering purely
    because time passed.
    """

    name = "information_gain"

    def __init__(
        self, threshold: float, staleness_half_life_s: float = 3.0, min_gap_s: float = 0.3
    ) -> None:
        self.threshold = float(threshold)
        self.staleness_half_life_s = float(staleness_half_life_s)
        self.min_gap_s = float(min_gap_s)
        self._novelty_at_last_call = 0.0

    def reset(self) -> None:
        self._novelty_at_last_call = 0.0

    def gain(self, ctx: FrameContext) -> float:
        if ctx.t_last_call is None:
            return 1.0
        # Saturates at 1.0: staleness raises urgency but can never manufacture a trigger alone.
        stale = 1.0 - 0.5 ** (ctx.since_last_call / self.staleness_half_life_s)
        # Novelty accrued since the last call, not absolute novelty.
        delta = max(0.0, ctx.novelty - self._novelty_at_last_call)
        return float(delta * (0.5 + 0.5 * stale))

    def decide(self, ctx: FrameContext) -> Decision:
        if ctx.t_last_call is None:
            return Decision(True, 1.0, "first frame")
        if ctx.since_last_call < self.min_gap_s:
            return Decision(False, 0.0, "rate limited")
        g = self.gain(ctx)
        return Decision(g >= self.threshold, g, f"gain {g:.4f} vs {self.threshold:.4f}")

    def observe_call(self, ctx: FrameContext) -> None:
        self._novelty_at_last_call = ctx.novelty

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "threshold": self.threshold,
            "staleness_half_life_s": self.staleness_half_life_s,
            "min_gap_s": self.min_gap_s,
        }

    @staticmethod
    def operating_points() -> list[float]:
        return [0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.35]


class LearnedPolicy(TriggerPolicy):
    """A small logistic model over fast-tier features, fit on training clips.

    Deliberately tiny: 5 features and a bias. The project's claim is about the constraint and the
    measurement, not about model capacity, and a large policy would cost more per frame than the
    encoder it consumes. If this does not beat `embedding_novelty`, that is the honest result.

    Features are all causal — nothing here can see the future.
    """

    name = "learned"

    FEATURES = ("novelty", "scene_change", "motion", "since_last_call", "novelty_delta")

    def __init__(self, weights: Sequence[float], bias: float, threshold: float = 0.5,
                 min_gap_s: float = 0.3) -> None:
        self.w = np.asarray(weights, dtype=np.float64)
        self.b = float(bias)
        self.threshold = float(threshold)
        self.min_gap_s = float(min_gap_s)
        self._novelty_at_last_call = 0.0

    def reset(self) -> None:
        self._novelty_at_last_call = 0.0

    def features(self, ctx: FrameContext) -> np.ndarray:
        since = min(ctx.since_last_call, 30.0) if np.isfinite(ctx.since_last_call) else 30.0
        return np.array([
            ctx.novelty,
            ctx.scene_change,
            ctx.motion,
            since / 10.0,
            max(0.0, ctx.novelty - self._novelty_at_last_call),
        ], dtype=np.float64)

    def probability(self, ctx: FrameContext) -> float:
        z = float(np.dot(self.w, self.features(ctx)) + self.b)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def decide(self, ctx: FrameContext) -> Decision:
        if ctx.t_last_call is None:
            return Decision(True, 1.0, "first frame")
        if ctx.since_last_call < self.min_gap_s:
            return Decision(False, 0.0, "rate limited")
        p = self.probability(ctx)
        return Decision(p >= self.threshold, p, f"p={p:.3f} vs {self.threshold:.2f}")

    def observe_call(self, ctx: FrameContext) -> None:
        self._novelty_at_last_call = ctx.novelty

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "weights": {k: round(float(v), 4) for k, v in zip(self.FEATURES, self.w)},
            "bias": round(self.b, 4),
            "threshold": self.threshold,
            "min_gap_s": self.min_gap_s,
        }

    @staticmethod
    def operating_points() -> list[float]:
        return [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.88, 0.95]


def _hz(period_s: float) -> str:
    hz = 1.0 / period_s
    return f"{hz:g}hz" if hz >= 1 else f"{period_s:g}s"


def build_policy(kind: str, value: float, **kwargs: Any) -> TriggerPolicy:
    """Construct a policy at one operating point. `value` is that policy's knob."""
    if kind == "fixed_interval":
        return FixedIntervalPolicy(value)
    if kind == "motion_threshold":
        return MotionThresholdPolicy(value, **kwargs)
    if kind == "embedding_novelty":
        return EmbeddingNoveltyPolicy(value, **kwargs)
    if kind == "information_gain":
        return InformationGainPolicy(value, **kwargs)
    if kind == "learned":
        weights = kwargs.pop("weights")
        bias = kwargs.pop("bias")
        return LearnedPolicy(weights, bias, threshold=value, **kwargs)
    raise ValueError(f"unknown policy {kind!r}")


POLICY_KINDS = (
    "fixed_interval",
    "motion_threshold",
    "embedding_novelty",
    "information_gain",
    "learned",
)
