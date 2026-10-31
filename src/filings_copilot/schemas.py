"""Core data contracts shared by ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FilingMeta(BaseModel):
    ticker: str
    company: str
    cik: str
    form: str
    fiscal_year: int
    period_of_report: str
    filing_date: str
    accession: str
    url: str

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}-{self.form}-FY{self.fiscal_year}"


class Section(BaseModel):
    """A top-level 10-K item (e.g. 'Item 7. MD&A') with character offsets into the doc text."""

    item: str
    title: str
    start: int
    end: int


class ParsedFiling(BaseModel):
    meta: FilingMeta
    text: str
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    ticker: str
    company: str
    fiscal_year: int
    item: str
    section_title: str
    start: int  # character offsets into ParsedFiling.text (chunking-independent evidence)
    end: int
    text: str
    is_table: bool = False
    context: str = ""  # prepended for indexing only (metadata or LLM-written situating context)

    @property
    def index_text(self) -> str:
        return f"{self.context}\n\n{self.text}" if self.context else self.text


class Evidence(BaseModel):
    """Gold evidence expressed as a character span, so relevance survives re-chunking."""

    doc_id: str
    start: int
    end: int
    quote: str = ""


QuestionType = Literal["single_hop", "table", "multi_hop", "unanswerable"]


class GoldQuestion(BaseModel):
    qid: str
    question: str
    answer: str
    qtype: QuestionType
    evidence: list[Evidence] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    fiscal_years: list[int] = Field(default_factory=list)
    verified: bool = False  # set True after human review


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    ranks: dict[str, int] = Field(default_factory=dict)  # per-stage rank, for debugging


class Answer(BaseModel):
    question: str
    answer: str
    answerable: bool
    citations: list[int] = Field(default_factory=list)  # 1-based indexes into passages
    passages: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: float = 0.0
    cost_usd: float = 0.0
