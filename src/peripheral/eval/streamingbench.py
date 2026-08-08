"""StreamingBench Real-Time Visual Understanding adapter.

StreamingBench asks multiple-choice questions **at timestamps** inside a video, which is exactly the
shape our replay harness already enforces: a question at t=00:04:12 may only be answered from frames
that have already arrived by then.

Two honesty constraints shape this adapter:

**Real time is expensive, and we do not cheat around it.** Covering all 250 questions in samples
1–50 requires 377 minutes of wall-clock replay, because the clips are minutes long and replay runs
at 1.0×. We therefore evaluate a **documented subset chosen by clip length**, and state the
selection rule and its bias rather than quietly sampling.

**Scoring is by letter, not by text similarity.** The benchmark supplies four options and a correct
letter, so the model's answer is parsed to a letter and compared exactly. That avoids importing the
paraphrase problems from Phase 3 into an external benchmark, and it makes the number comparable to
published StreamingBench results — with the caveat that our model is prompted zero-shot on a single
frame's worth of held context, not fed the whole clip.
"""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

QID_RE = re.compile(r"sample_(\d+)_(\d+)")
LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_timestamp(ts: str) -> float:
    """`HH:MM:SS` or `MM:SS` to seconds."""
    parts = [int(p) for p in ts.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


@dataclass
class SBQuestion:
    question_id: str
    sample: int
    index: int
    task_type: str
    question: str
    t: float
    answer: str                 # correct option letter
    options: list[str] = field(default_factory=list)
    frames_required: str = ""
    temporal_clue_type: str = ""

    def prompt(self) -> str:
        """Zero-shot multiple choice. Asking for a bare letter keeps parsing unambiguous."""
        opts = "\n".join(self.options)
        return (
            f"{self.question}\n{opts}\n"
            "Answer with the single letter of the correct option (A, B, C or D) and nothing else."
        )


@dataclass
class SBSample:
    sample: int
    questions: list[SBQuestion]

    @property
    def last_t(self) -> float:
        return max(q.t for q in self.questions)


def load_questions(csv_path: str | Path) -> list[SBQuestion]:
    out: list[SBQuestion] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = QID_RE.search(row["question_id"])
            if not m:
                continue
            try:
                options = list(ast.literal_eval(row["options"]))
            except Exception:  # noqa: BLE001 - a malformed row is skipped, not guessed at
                continue
            out.append(SBQuestion(
                question_id=row["question_id"],
                sample=int(m.group(1)),
                index=int(m.group(2)),
                task_type=row["task_type"],
                question=row["question"],
                t=parse_timestamp(row["time_stamp"]),
                answer=row["answer"].strip().upper()[:1],
                options=[str(o) for o in options],
                frames_required=row.get("frames_required", ""),
                temporal_clue_type=row.get("temporal_clue_type", ""),
            ))
    return out


def group_samples(questions: list[SBQuestion]) -> dict[int, SBSample]:
    by: dict[int, list[SBQuestion]] = {}
    for q in questions:
        by.setdefault(q.sample, []).append(q)
    return {s: SBSample(s, sorted(qs, key=lambda q: q.t)) for s, qs in by.items()}


def select_subset(
    samples: dict[int, SBSample],
    available: set[int],
    budget_s: float,
) -> tuple[list[SBSample], dict[str, Any]]:
    """Choose samples that fit a wall-clock budget, shortest first.

    **The selection rule and its bias, stated up front:** clips are taken in ascending order of
    last-question timestamp, so the subset is biased towards *shorter* clips. Shorter clips give a
    scheduler less opportunity to drift out of date, so this subset is, if anything, **favourable**
    to us. It is reported as a subset, never as a StreamingBench score.
    """
    candidates = sorted((s for i, s in samples.items() if i in available), key=lambda s: s.last_t)
    chosen: list[SBSample] = []
    total = 0.0
    for s in candidates:
        if total + s.last_t > budget_s:
            continue
        chosen.append(s)
        total += s.last_t
    return chosen, {
        "selection_rule": "ascending last-question timestamp until the wall-clock budget is spent",
        "budget_s": budget_s,
        "selected_replay_s": round(total, 1),
        "n_samples_selected": len(chosen),
        "n_samples_available": len(candidates),
        "n_questions_selected": sum(len(s.questions) for s in chosen),
        "bias": (
            "Biased towards SHORTER clips, which give a scheduler less time to go stale — so this "
            "subset is favourable to us. Not a StreamingBench score; a subset result."
        ),
    }


def extract_letter(text: str) -> str | None:
    """Pull the chosen option letter out of a free-form answer.

    Tolerant of "B." / "(B)" / "The answer is B", and of a leading restatement — but it takes the
    FIRST standalone letter, so a model that hedges by listing several is scored on its first
    commitment rather than being credited for mentioning the right one somewhere.
    """
    if not text:
        return None
    head = text.strip()
    m = re.match(r"^[^A-Za-z]*([ABCD])\b", head)
    if m:
        return m.group(1)
    m = LETTER_RE.search(head)
    return m.group(1) if m else None


def score(predictions: list[tuple[SBQuestion, str | None]]) -> dict[str, Any]:
    """Exact-letter accuracy overall and by task type."""
    n = len(predictions)
    if n == 0:
        return {"n": 0, "accuracy": None, "by_task": {}, "unparsed": 0}
    correct = sum(1 for q, p in predictions if p == q.answer)
    unparsed = sum(1 for _, p in predictions if p is None)

    by_task: dict[str, dict[str, Any]] = {}
    for q, p in predictions:
        d = by_task.setdefault(q.task_type, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(p == q.answer)
    for d in by_task.values():
        d["accuracy"] = round(d["correct"] / d["n"], 4)

    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "correct": correct,
        "unparsed": unparsed,
        "random_baseline": 0.25,
        "by_task": dict(sorted(by_task.items())),
    }
