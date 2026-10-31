"""Lexical retrieval. Strong on exact terms that matter in filings: tickers, line items
("diluted EPS", "free cash flow"), product names and numbers."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "will",
        "with",
        "how",
        "did",
        "does",
        "do",
    ]
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._bm25 = BM25Okapi([tokenize(t) or ["<empty>"] for t in texts], k1=k1, b=b)
        self.size = len(texts)

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
