"""Fit the small learned policy on training clips.

Logistic regression in numpy — no sklearn. Five features and a bias, because the project's claim is
about the constraint and the measurement, not model capacity, and anything larger would cost more
per frame than the encoder feeding it.

**How labels are made.** A teacher walks the clip calling exactly when the scene state changes,
which is the best any policy could do. At each frame the features are computed against *that*
teacher's call history, and the label is "would a call right now be justified" — i.e. has the state
changed since the teacher last called. This keeps features and labels self-consistent; labelling
against a policy's own history would make the target depend on the model being fit.

Positives are rare (a handful of transitions in 720 frames), so the loss is class-balanced.
Without that the model learns to never fire, which scores well on accuracy and is useless.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..eval.traces import ClipTrace
from .policies import FrameContext, LearnedPolicy


def _teacher_examples(trace: ClipTrace, min_gap_s: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels from a state-change teacher."""
    fps = trace.fps
    n = trace.n_frames
    feats, labels = [], []

    t_last_call: float | None = None
    state_at_last_call: int | None = None
    novelty_at_last_call = 0.0

    probe = LearnedPolicy(weights=np.zeros(5), bias=0.0)

    for i in range(n):
        t = i / fps
        ctx = FrameContext(
            frame_idx=i, t=t,
            motion=float(trace.motion[i]),
            novelty=float(trace.novelty[i]),
            scene_change=float(trace.scene_change[i]),
            embedding=None,
            t_last_call=t_last_call,
            n_calls=0,
        )
        probe._novelty_at_last_call = novelty_at_last_call
        state_now = int(trace.state_ids[i])

        # Justified iff the scene changed since the teacher last looked.
        justified = state_at_last_call is None or state_now != state_at_last_call
        rate_ok = t_last_call is None or (t - t_last_call) >= min_gap_s

        if t_last_call is not None:  # skip the forced first call, which teaches nothing
            feats.append(probe.features(ctx))
            labels.append(1.0 if (justified and rate_ok) else 0.0)

        if justified and rate_ok:
            t_last_call = t
            state_at_last_call = state_now
            novelty_at_last_call = float(trace.novelty[i])

    return np.asarray(feats, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def fit(
    traces: Sequence[ClipTrace],
    l2: float = 1e-3,
    epochs: int = 4000,
    lr: float = 0.5,
    min_gap_s: float = 0.3,
    seed: int = 1337,
) -> dict[str, Any]:
    """Return weights, bias and training diagnostics."""
    Xs, ys = [], []
    for tr in traces:
        x, y = _teacher_examples(tr, min_gap_s=min_gap_s)
        if len(x):
            Xs.append(x)
            ys.append(y)
    if not Xs:
        raise ValueError("no training examples")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)

    # Standardise so one feature's units cannot dominate the others.
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xn = (X - mu) / sigma

    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    if n_pos == 0:
        raise ValueError("no positive examples: the teacher never fired")
    # Class balance: positives are ~1% of frames. Unweighted, "never fire" is a great local optimum.
    w_pos, w_neg = n_neg / max(n_pos, 1.0), 1.0
    sample_w = np.where(y > 0, w_pos, w_neg)
    sample_w = sample_w / sample_w.mean()

    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=Xn.shape[1])
    b = 0.0

    for _ in range(epochs):
        z = Xn @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        err = (p - y) * sample_w
        gw = Xn.T @ err / len(y) + l2 * w
        gb = err.mean()
        w -= lr * gw
        b -= lr * gb

    # Fold standardisation back into the weights so the policy needs no scaler at inference.
    w_raw = w / sigma
    b_raw = float(b - np.dot(w / sigma, mu))

    z = X @ w_raw + b_raw
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    pred = (p >= 0.5).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())

    return {
        "weights": [float(v) for v in w_raw],
        "bias": b_raw,
        "feature_names": list(LearnedPolicy.FEATURES),
        "n_examples": int(len(y)),
        "n_positive": int(n_pos),
        "positive_rate": round(float(n_pos / len(y)), 5),
        "train_precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "train_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "train_clips": [t.clip_name for t in traces],
    }
