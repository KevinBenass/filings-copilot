"""The retriever: query plan -> filtered candidates -> BM25 + dense -> RRF -> rerank."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from filings_copilot.config import Settings
from filings_copilot.index.bm25 import BM25Index
from filings_copilot.index.dense import DenseIndex, Embedder
from filings_copilot.llm import LLM
from filings_copilot.retrieve.fusion import rrf
from filings_copilot.retrieve.query import QueryPlan, extract_tickers, extract_years, plan_query
from filings_copilot.retrieve.rerank import Reranker
from filings_copilot.schemas import Chunk, RetrievedChunk


def index_path(settings: Settings) -> Path:
    slug = settings.embedding.model.replace("/", "__")
    return (
        settings.resolve(settings.embedding.index_dir) / settings.chunking.variant / f"{slug}.npy"
    )


class Retriever:
    def __init__(
        self,
        settings: Settings,
        chunks: list[Chunk],
        dense: DenseIndex | None,
        embedder: Embedder | None,
        reranker: Reranker | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings
        self.cfg = settings.retrieval
        self.chunks = chunks
        self.bm25 = BM25Index([c.index_text for c in chunks])
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm
        self.known_tickers = {c.ticker for c in chunks}
        self._tickers = np.array([c.ticker for c in chunks])
        self._years = np.array([c.fiscal_year for c in chunks])
        self.last_timings: dict[str, float] = {}

    # ---------------------------------------------------------------- filters
    def build_mask(self, tickers: list[str], years: list[int]) -> np.ndarray | None:
        """Ticker filter is hard; year filter is soft.

        A question about FY2023 is often answered by the FY2024 report, which restates the
        prior year for comparison, so filings for year Y and Y+1 are both kept. If the filter
        leaves too few candidates (bad extraction), it is relaxed rather than starving recall.
        """
        if not self.cfg.use_filters or (not tickers and not years):
            return None
        min_size = self.cfg.top_k * 3
        mask = np.isin(self._tickers, tickers) if tickers else np.ones(len(self.chunks), bool)
        if years:
            allowed = sorted({y for yr in years for y in (yr, yr + 1)})
            with_years = mask & np.isin(self._years, allowed)
            if with_years.sum() >= min_size:
                mask = with_years
        return mask if mask.sum() >= min_size else None

    # ---------------------------------------------------------------- stages
    def _candidates(
        self, query: str, mask: np.ndarray | None
    ) -> tuple[list[int], dict[int, dict[str, int]]]:
        n = self.cfg.candidates_per_retriever
        lists: list[list[int]] = []
        ranks: dict[int, dict[str, int]] = {}
        t0 = time.perf_counter()
        if self.cfg.mode in ("bm25", "hybrid"):
            bm = [i for i, _ in self.bm25.search(query, n, mask)]
            lists.append(bm)
            for r, i in enumerate(bm, 1):
                ranks.setdefault(i, {})["bm25"] = r
        t1 = time.perf_counter()
        if self.cfg.mode in ("dense", "hybrid"):
            if self.dense is None or self.embedder is None:
                raise RuntimeError("Dense retrieval requested but no dense index is loaded")
            dn = [i for i, _ in self.dense.search(self.embedder.embed_query(query), n, mask)]
            lists.append(dn)
            for r, i in enumerate(dn, 1):
                ranks.setdefault(i, {})["dense"] = r
        t2 = time.perf_counter()
        fused = [i for i, _ in rrf(lists, self.cfg.rrf_k)] if len(lists) > 1 else lists[0]
        for r, i in enumerate(fused, 1):
            ranks.setdefault(i, {})["fused"] = r
        self.last_timings["bm25_ms"] = self.last_timings.get("bm25_ms", 0) + (t1 - t0) * 1e3
        self.last_timings["dense_ms"] = self.last_timings.get("dense_ms", 0) + (t2 - t1) * 1e3
        return fused, ranks

    def _rerank(
        self, query: str, ids: list[int], ranks: dict[int, dict[str, int]]
    ) -> list[tuple[int, float]]:
        if not (self.cfg.rerank and self.reranker):
            return [(i, 1.0 / r) for r, i in enumerate(ids, 1)]
        t0 = time.perf_counter()
        pool = ids[: self.cfg.rerank_top_n]
        scores = self.reranker.score(query, [self.chunks[i].index_text for i in pool])
        order = sorted(zip(pool, scores, strict=True), key=lambda x: -x[1])
        for r, (i, _) in enumerate(order, 1):
            ranks[i]["rerank"] = r
        self.last_timings["rerank_ms"] = (
            self.last_timings.get("rerank_ms", 0) + (time.perf_counter() - t0) * 1e3
        )
        return order

    # ---------------------------------------------------------------- public
    def plan(self, question: str) -> QueryPlan:
        return plan_query(question, self.known_tickers, self.llm, self.cfg.decompose)

    def retrieve(self, question: str, plan: QueryPlan | None = None) -> list[RetrievedChunk]:
        self.last_timings = {}
        t0 = time.perf_counter()
        plan = plan or self.plan(question)
        k = self.cfg.top_k
        per_query: list[list[tuple[int, float]]] = []
        ranks: dict[int, dict[str, int]] = {}
        for q in plan.queries:
            # Sub-queries carry their own entities; the original question's apply otherwise.
            tickers = extract_tickers(q, self.known_tickers) or plan.tickers
            years = extract_years(q) or plan.fiscal_years
            ids, r = self._candidates(q, self.build_mask(tickers, years))
            for i, d in r.items():
                ranks.setdefault(i, {}).update(d)
            per_query.append(self._rerank(q, ids, ranks))

        # Round-robin across sub-queries so every needed fact gets coverage.
        merged: list[tuple[int, float]] = []
        seen: set[int] = set()
        for depth in range(max(len(p) for p in per_query) if per_query else 0):
            for p in per_query:
                if depth < len(p) and p[depth][0] not in seen:
                    seen.add(p[depth][0])
                    merged.append(p[depth])
            if len(merged) >= k:
                break
        self.last_timings["total_ms"] = (time.perf_counter() - t0) * 1e3
        return [
            RetrievedChunk(chunk=self.chunks[i], score=s, ranks=ranks.get(i, {}))
            for i, s in merged[:k]
        ]
