"""Query understanding: entity/period extraction (rules) and multi-hop decomposition (LLM).

Rules handle the frequent, cheap case: "What was Microsoft's FY2024 operating margin?" gives
ticker=MSFT, year=2024 and lets us filter the candidate set before ranking, which removes
the most common failure in financial RAG: retrieving the right passage from the wrong
company or year. The LLM is only called to split comparative questions into sub-queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from filings_copilot.llm import LLM

ALIASES: dict[str, list[str]] = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon", "aws"],
    "GOOGL": ["alphabet", "google", "youtube"],
    "META": ["meta platforms", "facebook", "instagram"],
    "JPM": ["jpmorgan", "jp morgan", "j.p. morgan", "chase"],
    "JNJ": ["johnson & johnson", "johnson and johnson", "j&j"],
    "XOM": ["exxon", "exxonmobil", "exxon mobil"],
    "TSLA": ["tesla"],
}

YEAR_RE = re.compile(
    r"\b(?:fy|fiscal(?:\s+year)?)\s*'?(\d{2}|\d{4})\b|\b(20\d{2})\b", re.IGNORECASE
)


@dataclass
class QueryPlan:
    original: str
    tickers: list[str] = field(default_factory=list)
    fiscal_years: list[int] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)

    @property
    def queries(self) -> list[str]:
        return self.sub_queries or [self.original]


def extract_tickers(question: str, known: set[str]) -> list[str]:
    q = question.lower()
    found: list[str] = []
    for ticker in sorted(known):
        names = [ticker.lower(), *ALIASES.get(ticker, [])]
        for name in names:
            pattern = rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])"
            if re.search(pattern, q):
                found.append(ticker)
                break
    return found


def extract_years(question: str) -> list[int]:
    years: set[int] = set()
    for m in YEAR_RE.finditer(question):
        raw = m.group(1) or m.group(2)
        year = int(raw) + 2000 if len(raw) == 2 else int(raw)
        if 2000 <= year <= 2100:
            years.add(year)
    return sorted(years)


DECOMPOSE_PROMPT = """You split questions about SEC annual reports into retrieval sub-queries.

Question: {question}

If answering needs facts from more than one company, fiscal year or section, return one
self-contained search query per fact needed (name the company and year in each). If one
search suffices, return the question unchanged as the only item.
Respond with JSON only: {{"sub_queries": ["...", "..."]}}"""


def plan_query(
    question: str, known_tickers: set[str], llm: LLM | None = None, decompose: bool = False
) -> QueryPlan:
    plan = QueryPlan(
        original=question,
        tickers=extract_tickers(question, known_tickers),
        fiscal_years=extract_years(question),
    )
    if decompose and llm is not None:
        try:
            out = llm.complete_json(
                DECOMPOSE_PROMPT.format(question=question), model=llm.cfg.fast_model, max_tokens=300
            )
            subs = [s for s in out.get("sub_queries", []) if isinstance(s, str) and s.strip()]
            if len(subs) > 1:
                plan.sub_queries = subs[:4]
        except (ValueError, AttributeError):
            pass  # decomposition is an optimization; fall back to the original query
    return plan
