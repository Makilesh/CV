"""Answer-quality scoring for the Phase 3 sweep.

**What this measures, and what it does not.** There are no ground-truth annotations for these
clips yet (Phase 4 builds them), so nothing here scores *correctness*. What it scores is
**fidelity**: how far a quantized model's answer drifts from the same model's highest-precision
variant on the same frame. That is exactly the question a quantization sweep should answer —
"what did Q4_K_M cost me relative to Q8_0" — and it is honest about being nothing more.

Deliberately dependency-free. A sentence-embedding model would give smoother numbers, but it would
put a second neural network's opinion inside a measurement whose whole point is to isolate the
first one.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Sequence

# Words that carry no scene content — dropped before computing content overlap so that shared
# boilerplate ("the image shows a ...") does not inflate agreement.
_STOP = frozenset(
    """a an the this that these those is are was were be been being am it its it's of in on at to
    for with by from as and or but if then there here what which who whom whose how when where why
    i you he she they we me him her them us my your his their our not no nor so than too very can
    will just don't should now image picture photo shows showing appears seems likely possibly""".split()
)

_WORD = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall((text or "").lower()))


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2]


def similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1]. Sensitive to phrasing as well as content."""
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def content_f1(a: str, b: str) -> float:
    """Bag-of-content-words F1 in [0, 1]. Insensitive to word order and phrasing."""
    ca, cb = Counter(content_words(a)), Counter(content_words(b))
    if not ca or not cb:
        return 1.0 if not ca and not cb else 0.0
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / sum(ca.values())
    r = overlap / sum(cb.values())
    return 2 * p * r / (p + r)


def is_degenerate(text: str, min_words: int = 3, repeat_ratio: float = 0.5) -> bool:
    """Empty, truncated to nothing, or looping on one token — a failure mode of hard quantization.

    A degenerate answer can still score well on similarity against another degenerate answer, so
    it is counted separately rather than folded into the agreement score.
    """
    words = _WORD.findall((text or "").lower())
    if len(words) < min_words:
        return True
    most_common = Counter(words).most_common(1)[0][1]
    return most_common / len(words) > repeat_ratio


def score_against_reference(
    answers: Sequence[str], reference: Sequence[str]
) -> dict[str, Any]:
    """Fidelity of `answers` to `reference`, pairwise by frame."""
    n = min(len(answers), len(reference))
    if n == 0:
        return {
            "n": 0, "similarity_mean": None, "content_f1_mean": None,
            "exact_match_rate": None, "degenerate_rate": None, "mean_words": None,
        }
    sims = [similarity(answers[i], reference[i]) for i in range(n)]
    f1s = [content_f1(answers[i], reference[i]) for i in range(n)]
    exact = [normalize(answers[i]) == normalize(reference[i]) for i in range(n)]
    degen = [is_degenerate(a) for a in answers[:n]]
    words = [len(_WORD.findall(a or "")) for a in answers[:n]]
    return {
        "n": n,
        "similarity_mean": round(sum(sims) / n, 4),
        "content_f1_mean": round(sum(f1s) / n, 4),
        "exact_match_rate": round(sum(exact) / n, 4),
        "degenerate_rate": round(sum(degen) / n, 4),
        "mean_words": round(sum(words) / n, 2),
    }


def self_consistency(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Exact-match rate between two passes over the same frames at temperature 0."""
    n = min(len(first), len(second))
    if n == 0:
        return None
    return round(sum(normalize(first[i]) == normalize(second[i]) for i in range(n)) / n, 4)


def noise_floor(first: Sequence[str], second: Sequence[str]) -> dict[str, Any]:
    """**The number that makes every other quality number readable.**

    llama-server is not deterministic at temperature 0 — measured 2026-08-08, a model asked the
    same frame twice back to back, with identical cache state and the prompt cache disabled,
    returned a different string 67% of the time. The differences are paraphrase
    ("looking thoughtfully toward" vs "at"), which is what floating-point noise flipping a near-tie
    in greedy argmax looks like.

    So a config scoring 0.85 content-F1 against its family reference has NOT necessarily lost 0.15
    to quantization. It has lost 0.15 to quantization *and* serving noise, and the two are
    inseparable unless you know how much a model disagrees with **itself**. That is this number:
    ask the same config the same frames twice and score it against its own answers.

    Read the table as: fidelity ≈ noise floor means no measurable quantization damage.
    """
    n = min(len(first), len(second))
    if n == 0:
        return {"n": 0, "self_content_f1": None, "self_exact_match": None}
    f1s = [content_f1(first[i], second[i]) for i in range(n)]
    return {
        "n": n,
        "self_content_f1": round(sum(f1s) / n, 4),
        "self_exact_match": self_consistency(first, second),
    }
