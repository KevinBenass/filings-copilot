"""Retrieval metrics with span-based relevance.

Gold evidence is a character span in a filing, not a chunk id. A retrieved chunk "covers"
an evidence span if they are in the same filing and overlap substantially. This keeps the
gold set valid across chunking strategies, which is what makes the chunking ablation fair.

For multi-hop questions with several evidence spans, a chunk only earns gain for evidence
not already covered by a higher-ranked chunk, so retrieving the same fact twice is not
rewarded.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from filings_copilot.schemas import Chunk, Evidence

MIN_OVERLAP = 0.5


def covers(chunk: Chunk, ev: Evidence, min_overlap: float = MIN_OVERLAP) -> bool:
    if chunk.doc_id != ev.doc_id:
        return False
    overlap = min(chunk.end, ev.end) - max(chunk.start, ev.start)
    if overlap <= 0:
        return False
    ev_len = max(1, ev.end - ev.start)
    chunk_len = max(1, chunk.end - chunk.start)
    # Either most of the evidence is in the chunk, or the chunk is mostly evidence
    # (the latter handles long evidence spans split across several small chunks).
    return overlap / ev_len >= min_overlap or overlap / chunk_len >= min_overlap


def gains(retrieved: Sequence[Chunk], evidence: Sequence[Evidence]) -> list[int]:
    """Binary gain per rank: 1 if the chunk covers a not-yet-covered evidence span."""
    covered: set[int] = set()
    out: list[int] = []
    for chunk in retrieved:
        new = {j for j, ev in enumerate(evidence) if j not in covered and covers(chunk, ev)}
        covered |= new
        out.append(1 if new else 0)
    return out


def recall_at_k(g: list[int], n_evidence: int, k: int) -> float:
    return sum(g[:k]) / n_evidence if n_evidence else 0.0


def hit_at_k(g: list[int], k: int) -> float:
    return 1.0 if any(g[:k]) else 0.0


def mrr(g: list[int]) -> float:
    for rank, gain in enumerate(g, 1):
        if gain:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(g: list[int], n_evidence: int, k: int) -> float:
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(g[:k], 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(n_evidence, k) + 1))
    return dcg / ideal if ideal else 0.0


def score_question(
    retrieved: Sequence[Chunk], evidence: Sequence[Evidence], ks: Sequence[int]
) -> dict[str, float]:
    g = gains(retrieved, evidence)
    n = len(evidence)
    row: dict[str, float] = {"mrr": mrr(g)}
    for k in ks:
        row[f"hit@{k}"] = hit_at_k(g, k)
        row[f"recall@{k}"] = recall_at_k(g, n, k)
        row[f"ndcg@{k}"] = ndcg_at_k(g, n, k)
    return row
