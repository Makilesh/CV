"""Embedding-keyed answer cache with staleness tracking.

**What this is for, precisely.** The scheduler already decides *when* to call. A cache can only add
value in one situation the scheduler cannot exploit: the scene **returns to a state it was in
before**. An object is put down and later picked up; a person leaves and comes back; the lights dim
and recover. The scheduler correctly sees novelty and wants to call — but we have already paid for
an answer to exactly this scene, and can serve it for free.

That is the entire hypothesis, and it is narrow. Phase 4 got the call rate down to 0.42% of the
per-frame oracle, so whatever a cache saves comes out of an already-tiny budget. `PROMPT.md` marks
this phase droppable; the measurement is designed to be able to say "cut it".

**The risk it introduces is a false hit**: serving a stored answer for a scene that only *looks*
similar. That is strictly worse than calling, because it produces a confidently wrong answer at
zero cost, and it is the number that decides whether this feature earns its place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    embedding: np.ndarray      # L2-normalised fast-tier embedding, the key
    answer: str
    frame_idx: int             # frame the answer was actually derived from — the evidence
    t: float                   # when that evidence was captured
    hits: int = 0

    def similarity(self, other: np.ndarray) -> float:
        return float(np.dot(self.embedding, other))


@dataclass
class CacheLookup:
    hit: bool
    answer: str | None = None
    entry: CacheEntry | None = None
    similarity: float = 0.0
    staleness_s: float = 0.0
    reason: str = ""


class SemanticCache:
    """Bounded, embedding-keyed store of past answers.

    A lookup hits when some entry's embedding is within `threshold` cosine similarity **and** the
    entry is younger than `max_staleness_s`. The staleness bound matters: an embedding match says
    the scene *looks* the same, not that a stale description is still true — a room can look
    identical while the thing you were asked about has changed.
    """

    def __init__(
        self,
        threshold: float = 0.98,
        max_entries: int = 64,
        max_staleness_s: float = 30.0,
    ) -> None:
        self.threshold = float(threshold)
        self.max_entries = int(max_entries)
        self.max_staleness_s = float(max_staleness_s)
        self.entries: list[CacheEntry] = []

        self.n_lookups = 0
        self.n_hits = 0
        self.n_evictions = 0

    def reset(self) -> None:
        self.entries.clear()
        self.n_lookups = self.n_hits = self.n_evictions = 0

    def lookup(self, embedding: np.ndarray, t: float) -> CacheLookup:
        self.n_lookups += 1
        if not self.entries:
            return CacheLookup(False, reason="empty")

        sims = np.array([e.similarity(embedding) for e in self.entries])
        order = np.argsort(-sims)

        best_sim = float(sims[order[0]])
        for i in order:
            entry = self.entries[i]
            sim = float(sims[i])
            if sim < self.threshold:
                break
            age = t - entry.t
            if age > self.max_staleness_s:
                continue  # looks the same, but too old to trust
            entry.hits += 1
            self.n_hits += 1
            return CacheLookup(True, entry.answer, entry, sim, age, "hit")

        return CacheLookup(False, similarity=best_sim, reason="below threshold or too stale")

    def store(self, embedding: np.ndarray, answer: str, frame_idx: int, t: float) -> None:
        """Insert an answer we just paid for.

        Least-recently-useful eviction: drop the entry with the fewest hits, breaking ties by age.
        A pure LRU would evict a rarely-but-reliably-reused scene state — which is exactly the
        thing worth keeping.
        """
        self.entries.append(CacheEntry(embedding.astype(np.float32).copy(), answer, frame_idx, t))
        while len(self.entries) > self.max_entries:
            victim = min(range(len(self.entries)), key=lambda i: (self.entries[i].hits,
                                                                 self.entries[i].t))
            self.entries.pop(victim)
            self.n_evictions += 1

    @property
    def hit_rate(self) -> float | None:
        return self.n_hits / self.n_lookups if self.n_lookups else None

    def describe(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "max_entries": self.max_entries,
            "max_staleness_s": self.max_staleness_s,
            "n_entries": len(self.entries),
            "n_lookups": self.n_lookups,
            "n_hits": self.n_hits,
            "hit_rate": round(self.hit_rate, 5) if self.hit_rate is not None else None,
            "n_evictions": self.n_evictions,
        }


class SceneStateSummary:
    """A rolling textual summary of scene state, answerable with no VLM call.

    Deliberately extractive rather than generative: it holds the most recent answer plus how long
    the scene has looked this way. Generating a summary would need a language model, which is the
    cost this component exists to avoid — a summariser in the "free" path would be self-defeating.
    """

    def __init__(self) -> None:
        self.answer: str | None = None
        self.since_t: float | None = None
        self.evidence_frame: int | None = None
        self.n_confirmations = 0

    def reset(self) -> None:
        self.answer = None
        self.since_t = None
        self.evidence_frame = None
        self.n_confirmations = 0

    def update(self, answer: str, frame_idx: int, t: float, changed: bool) -> None:
        if changed or self.answer is None:
            self.answer = answer
            self.since_t = t
            self.evidence_frame = frame_idx
            self.n_confirmations = 1
        else:
            self.n_confirmations += 1

    def query(self, t: float) -> dict[str, Any]:
        """Answer a query with no VLM call."""
        if self.answer is None:
            return {"answer": None, "held_for_s": None, "evidence_frame": None}
        return {
            "answer": self.answer,
            "held_for_s": round(t - (self.since_t or t), 3),
            "evidence_frame": self.evidence_frame,
            "confirmations": self.n_confirmations,
        }
