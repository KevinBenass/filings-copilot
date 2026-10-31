"""Reciprocal rank fusion (Cormack et al., 2009).

RRF combines ranked lists using ranks only, so BM25 scores (unbounded) and cosine scores
(bounded) never need calibrating against each other.
"""

from __future__ import annotations

from collections import defaultdict


def rrf(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
