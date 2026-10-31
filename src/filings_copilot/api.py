"""FastAPI service. Run: `uvicorn filings_copilot.api:app` (or `make api`).

Also the integration point for the equity-research-agents project, which calls
POST /retrieve and POST /ask as tools.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from filings_copilot.cli import BEST_CONFIG, settings_for
from filings_copilot.generate.answer import Copilot
from filings_copilot.llm import LLM
from filings_copilot.retrieve.factory import build_retriever

app = FastAPI(title="Filings Copilot", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, examples=["How did NVIDIA's data center revenue change in fiscal 2025?"]
    )
    top_k: int | None = Field(None, ge=1, le=20)


class Passage(BaseModel):
    n: int
    chunk_id: str
    ticker: str
    company: str
    fiscal_year: int
    section: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    passages: list[Passage]
    timings_ms: dict[str, float]


class AskResponse(RetrieveResponse):
    answer: str
    answerable: bool
    citations: list[int]
    latency_ms: float
    cost_usd: float


@lru_cache(maxsize=1)
def copilot() -> Copilot:
    s = settings_for(os.environ.get("FCOPILOT_CONFIG", BEST_CONFIG))
    llm = LLM(s.llm, s.resolve(s.llm.cache_path))
    return Copilot(build_retriever(s, llm), llm)


def _passages(hits) -> list[Passage]:
    return [
        Passage(
            n=n,
            chunk_id=h.chunk.chunk_id,
            ticker=h.chunk.ticker,
            company=h.chunk.company,
            fiscal_year=h.chunk.fiscal_year,
            section=h.chunk.section_title,
            text=h.chunk.text,
            score=h.score,
        )
        for n, h in enumerate(hits, 1)
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: AskRequest) -> RetrieveResponse:
    r = copilot().retriever
    hits = r.retrieve(req.question)[: req.top_k or r.cfg.top_k]
    return RetrieveResponse(passages=_passages(hits), timings_ms=r.last_timings)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    c = copilot()
    ans = c.ask(req.question)
    return AskResponse(
        answer=ans.answer,
        answerable=ans.answerable,
        citations=ans.citations,
        passages=_passages(ans.passages),
        timings_ms=c.retriever.last_timings,
        latency_ms=ans.latency_ms,
        cost_usd=ans.cost_usd,
    )
