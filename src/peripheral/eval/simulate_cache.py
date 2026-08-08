"""Replay a policy **with a semantic cache** and measure what the cache actually bought.

The scheduler decides when a call is warranted. The cache then gets one chance to satisfy that
decision for free: if the current embedding matches a stored entry closely enough and recently
enough, the stored answer is served and **no VLM call is made**.

Three numbers decide whether this feature earns its place:

* **calls avoided** — the only benefit. Measured against the same policy with no cache.
* **false-hit rate** — hits that served an answer from a *different* scene state. This is the
  cost, and it is worse than it looks: a false hit is a confidently wrong answer produced at zero
  cost, which the system has no way to notice.
* **answer validity** — the same primary metric as Phase 4, so the accuracy cost of caching is
  directly comparable to the accuracy the scheduler achieved without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cache.semantic_cache import SceneStateSummary, SemanticCache
from ..scheduler.policies import FrameContext, TriggerPolicy
from ..vlm.quality import content_f1
from .traces import ClipTrace


@dataclass
class CacheSimResult:
    clip: str
    policy: str
    operating_point: float
    cache_threshold: float
    n_frames: int
    duration_s: float
    n_triggers: int          # times the policy wanted an answer
    n_vlm_calls: int         # times we actually paid
    n_cache_hits: int
    calls_avoided: int
    calls_avoided_pct: float
    cache_hit_rate: float | None
    false_hits: int
    false_hit_rate: float | None
    answer_validity: float
    accuracy: float
    staleness_ms_mean: float
    staleness_ms_p95: float
    calls_per_min: float
    call_fraction: float

    def as_row(self) -> dict[str, Any]:
        return dict(self.__dict__)


def simulate_with_cache(
    trace: ClipTrace,
    policy: TriggerPolicy,
    cache: SemanticCache | None,
    embeddings: np.ndarray,
    warmup_frames: int = 15,
) -> CacheSimResult:
    """Replay `policy` over `trace`, consulting `cache` before each would-be VLM call."""
    policy.reset()
    if cache is not None:
        cache.reset()
    summary = SceneStateSummary()

    fps = trace.fps
    n = trace.n_frames

    t_last_call: float | None = None
    current_answer: str | None = None
    current_evidence_frame: int | None = None

    n_triggers = n_vlm = n_hits = false_hits = 0
    per_valid: list[float] = []
    per_f1: list[float] = []
    staleness_ms: list[float] = []

    for i in range(n):
        t = i / fps
        ctx = FrameContext(
            frame_idx=i, t=t,
            motion=float(trace.motion[i]),
            novelty=float(trace.novelty[i]),
            scene_change=float(trace.scene_change[i]),
            embedding=None,
            t_last_call=t_last_call,
            n_calls=n_vlm,
        )
        decision = policy.decide(ctx)

        if decision.fire:
            n_triggers += 1
            emb = embeddings[i]
            served_from_cache = False

            if cache is not None:
                look = cache.lookup(emb, t)
                if look.hit and look.entry is not None:
                    n_hits += 1
                    served_from_cache = True
                    current_answer = look.answer
                    # The evidence is the frame the CACHED answer came from, not this frame.
                    # Recording it as `i` would make staleness read as zero for exactly the
                    # answers most in need of auditing.
                    current_evidence_frame = look.entry.frame_idx
                    if int(trace.state_ids[look.entry.frame_idx]) != int(trace.state_ids[i]):
                        false_hits += 1

            if not served_from_cache:
                n_vlm += 1
                current_answer = trace.answers[i]
                current_evidence_frame = i
                if cache is not None:
                    cache.store(emb, trace.answers[i], i, t)

            changed = current_answer != summary.answer
            summary.update(current_answer or "", current_evidence_frame or i, t, changed)
            t_last_call = t
            policy.observe_call(ctx)

        if i >= warmup_frames:
            per_valid.append(
                1.0 if (current_evidence_frame is not None
                        and int(trace.state_ids[current_evidence_frame])
                        == int(trace.state_ids[i]))
                else 0.0
            )
            per_f1.append(
                content_f1(current_answer, trace.answers[i]) if current_answer is not None else 0.0
            )
            if current_evidence_frame is not None:
                staleness_ms.append((i - current_evidence_frame) / fps * 1000.0)

    duration_s = n / fps
    return CacheSimResult(
        clip=trace.clip_name,
        policy=policy.name,
        operating_point=float(getattr(policy, "threshold", getattr(policy, "period_s", 0.0))),
        cache_threshold=cache.threshold if cache else float("nan"),
        n_frames=n,
        duration_s=round(duration_s, 3),
        n_triggers=n_triggers,
        n_vlm_calls=n_vlm,
        n_cache_hits=n_hits,
        calls_avoided=n_triggers - n_vlm,
        calls_avoided_pct=round(100.0 * (n_triggers - n_vlm) / n_triggers, 3) if n_triggers else 0.0,
        cache_hit_rate=round(n_hits / n_triggers, 5) if n_triggers else None,
        false_hits=false_hits,
        false_hit_rate=round(false_hits / n_hits, 5) if n_hits else None,
        answer_validity=round(float(np.mean(per_valid)), 5) if per_valid else 0.0,
        accuracy=round(float(np.mean(per_f1)), 5) if per_f1 else 0.0,
        staleness_ms_mean=round(float(np.mean(staleness_ms)), 2) if staleness_ms else 0.0,
        staleness_ms_p95=round(float(np.percentile(staleness_ms, 95)), 2) if staleness_ms else 0.0,
        calls_per_min=round(n_vlm / (duration_s / 60.0), 3),
        call_fraction=round(n_vlm / n, 5),
    )
