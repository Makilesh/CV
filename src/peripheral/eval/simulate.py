"""Replay a policy over a trace and score it.

Three metrics, in the order they matter for the headline claim:

* **accuracy vs the per-frame oracle** — at every frame, how well does the answer the system is
  *currently holding* match what a VLM asked about *this* frame would have said. A policy that
  calls rarely is penalised for every frame its answer is out of date, which is the honest cost of
  not calling.
* **VLM calls per minute** — the cost axis of the headline figure.
* **false-trigger rate** — the fraction of calls fired when the scene had not changed since the
  previous call. First-class, because it is what separates a scheduler from a motion detector.

No lookahead is possible here by construction: the policy is handed one `FrameContext` at a time,
built only from the current frame and the policy's own history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..scheduler.policies import FrameContext, TriggerPolicy
from ..vlm.quality import content_f1
from .traces import ClipTrace


@dataclass
class SimResult:
    clip: str
    policy: str
    operating_point: float
    n_frames: int
    duration_s: float
    n_calls: int
    calls_per_min: float
    call_fraction: float            # calls / frames = fraction of the per-frame oracle's budget
    # PRIMARY. Fraction of frames on which the answer we are holding describes the scene state the
    # camera is actually in. Immune to paraphrase, and 1.0 for the oracle by construction.
    answer_validity: float
    accuracy: float                 # SECONDARY: mean content-F1 against this frame's oracle answer
    accuracy_vs_oracle_pct: float   # same number as a percentage of the oracle's own 1.0
    staleness_ms_mean: float
    staleness_ms_p95: float
    false_triggers: int
    justified_calls: int
    false_trigger_rate: float | None
    missed_events: int
    total_events: int
    event_recall: float | None
    detection_delay_ms_mean: float | None
    call_frames: list[int] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop("call_frames", None)
        return d


def simulate(trace: ClipTrace, policy: TriggerPolicy, warmup_frames: int = 15) -> SimResult:
    """Replay `policy` over `trace`.

    `warmup_frames` excludes the opening frames from *accuracy* scoring only: the rolling reference
    inside the fast tier needs a few frames to settle, and every policy would otherwise be scored
    on a period where novelty is defined to be 0. Calls made during warmup still count as calls —
    the budget is not forgiven, only the grading.
    """
    policy.reset()
    fps = trace.fps
    n = trace.n_frames

    t_last_call: float | None = None
    last_state_at_call: int | None = None
    current_answer: str | None = None
    current_evidence_frame: int | None = None

    n_calls = 0
    false_triggers = 0
    justified = 0
    call_frames: list[int] = []
    per_frame_f1: list[float] = []
    per_frame_valid: list[float] = []
    staleness_ms: list[float] = []

    # Event bookkeeping: a state transition is "covered" once a call is made while in the new state.
    transitions = [i for i in range(1, n) if trace.state_ids[i] != trace.state_ids[i - 1]]
    covered: dict[int, int] = {}

    for i in range(n):
        t = i / fps
        ctx = FrameContext(
            frame_idx=i,
            t=t,
            motion=float(trace.motion[i]),
            novelty=float(trace.novelty[i]),
            scene_change=float(trace.scene_change[i]),
            embedding=None,
            t_last_call=t_last_call,
            n_calls=n_calls,
        )
        decision = policy.decide(ctx)

        if decision.fire:
            state_now = int(trace.state_ids[i])
            if last_state_at_call is not None:
                if state_now == last_state_at_call:
                    false_triggers += 1
                else:
                    justified += 1
            # The answer the oracle gave for this exact frame — same model, frame and prompt.
            current_answer = trace.answers[i]
            current_evidence_frame = i
            t_last_call = t
            last_state_at_call = state_now
            n_calls += 1
            call_frames.append(i)
            policy.observe_call(ctx)

            for tr in transitions:
                if tr <= i and tr not in covered and int(trace.state_ids[tr]) == state_now:
                    covered[tr] = i

        if i >= warmup_frames:
            # PRIMARY: is the answer we hold about the scene the camera is actually in?
            per_frame_valid.append(
                1.0 if (current_evidence_frame is not None
                        and int(trace.state_ids[current_evidence_frame]) == int(trace.state_ids[i]))
                else 0.0
            )
            # SECONDARY: text agreement with this frame's oracle answer.
            per_frame_f1.append(
                content_f1(current_answer, trace.answers[i]) if current_answer is not None else 0.0
            )
            if current_evidence_frame is not None:
                staleness_ms.append((i - current_evidence_frame) / fps * 1000.0)

    duration_s = n / fps
    acc = float(np.mean(per_frame_f1)) if per_frame_f1 else 0.0
    validity = float(np.mean(per_frame_valid)) if per_frame_valid else 0.0
    labelled_calls = false_triggers + justified
    delays = [(covered[tr] - tr) / fps * 1000.0 for tr in covered]

    return SimResult(
        clip=trace.clip_name,
        policy=policy.name,
        operating_point=float(getattr(policy, "threshold", getattr(policy, "period_s", 0.0))),
        n_frames=n,
        duration_s=round(duration_s, 3),
        n_calls=n_calls,
        calls_per_min=round(n_calls / (duration_s / 60.0), 2),
        call_fraction=round(n_calls / n, 5),
        answer_validity=round(validity, 5),
        accuracy=round(acc, 5),
        accuracy_vs_oracle_pct=round(100.0 * acc, 3),
        staleness_ms_mean=round(float(np.mean(staleness_ms)), 2) if staleness_ms else 0.0,
        staleness_ms_p95=round(float(np.percentile(staleness_ms, 95)), 2) if staleness_ms else 0.0,
        false_triggers=false_triggers,
        justified_calls=justified,
        false_trigger_rate=round(false_triggers / labelled_calls, 5) if labelled_calls else None,
        missed_events=len(transitions) - len(covered),
        total_events=len(transitions),
        event_recall=round(len(covered) / len(transitions), 4) if transitions else None,
        detection_delay_ms_mean=round(float(np.mean(delays)), 1) if delays else None,
        call_frames=call_frames,
    )


def aggregate(results: list[SimResult]) -> dict[str, Any]:
    """Mean and spread across clips for one policy at one operating point.

    Spread across clips is what the headline figure's error bars show. It is the honest source of
    variation here: the policies are deterministic given a trace, so repeating a run changes
    nothing — what changes is which scene you point them at.
    """
    if not results:
        return {}
    val = np.array([r.answer_validity for r in results])
    acc = np.array([r.accuracy for r in results])
    cpm = np.array([r.calls_per_min for r in results])
    ftr = np.array([r.false_trigger_rate for r in results if r.false_trigger_rate is not None])
    rec = np.array([r.event_recall for r in results if r.event_recall is not None])
    return {
        "policy": results[0].policy,
        "operating_point": results[0].operating_point,
        "n_clips": len(results),
        "answer_validity_mean": round(float(val.mean()), 5),
        "answer_validity_std": round(float(val.std()), 5),
        "answer_validity_min": round(float(val.min()), 5),
        "accuracy_mean": round(float(acc.mean()), 5),
        "accuracy_std": round(float(acc.std()), 5),
        "accuracy_min": round(float(acc.min()), 5),
        "calls_per_min_mean": round(float(cpm.mean()), 3),
        "calls_per_min_std": round(float(cpm.std()), 3),
        "call_fraction_mean": round(float(np.mean([r.call_fraction for r in results])), 5),
        "false_trigger_rate_mean": round(float(ftr.mean()), 5) if ftr.size else None,
        "event_recall_mean": round(float(rec.mean()), 4) if rec.size else None,
        "staleness_ms_mean": round(float(np.mean([r.staleness_ms_mean for r in results])), 1),
        "per_clip": {r.clip: {"answer_validity": r.answer_validity, "accuracy": r.accuracy,
                              "calls_per_min": r.calls_per_min,
                              "false_trigger_rate": r.false_trigger_rate} for r in results},
    }


def metric_ceiling(trace: ClipTrace, lag: int = 1) -> float:
    """The highest `accuracy` (text agreement) any non-oracle policy could reach on this trace.

    The oracle scores 1.0 only because it is compared against its own cached string. Any other
    policy is compared *across calls*, and llama-server is not deterministic at temperature 0
    (Phase 3: same frame twice → a different string 67% of the time). So the real ceiling is the
    agreement between the oracle's answers on two adjacent frames **in the same scene state**,
    where nothing semantic changed and every difference is serving noise.

    Measured at 0.783 on these clips. Reporting text agreement against 1.0 without saying this
    would understate every policy by ~22 points.
    """
    st = trace.state_ids
    vals = [
        content_f1(trace.answers[i - lag], trace.answers[i])
        for i in range(lag, trace.n_frames)
        if st[i] == st[i - lag]
    ]
    return float(np.mean(vals)) if vals else float("nan")
