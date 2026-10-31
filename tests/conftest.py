"""Shared fixtures: a miniature 10-K and model stand-ins, so tests need no network or weights."""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from filings_copilot.config import load_settings
from filings_copilot.ingest.parse import parse_filing
from filings_copilot.schemas import FilingMeta

PROSE = (
    "The Company designs, manufactures and markets accelerated computing platforms. "
    "Data Center revenue grew substantially as cloud providers expanded capacity for "
    "large language model training. Gaming revenue was broadly flat year over year. "
)


def mini_10k_html(company: str = "Acme Compute", revenue: str = "60,922") -> str:
    risk = PROSE * 4 + "Export controls on advanced chips could reduce our revenue in China. " * 3
    mdna = PROSE * 3 + "Total revenue for fiscal year 2025 increased primarily due to Data Center. "
    return f"""<html><head><title>10-K</title><style>.x{{}}</style></head><body>
<div style="display:none"><ix:header>hidden xbrl</ix:header></div>
<p>{company} Annual Report on Form 10-K</p>
<table>
  <tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
  <tr><td>Item 1A.</td><td>Risk Factors</td><td>10</td></tr>
  <tr><td>Item 7.</td><td>MD&amp;A</td><td>30</td></tr>
</table>
<p>Item 1. Business</p>
<p>{PROSE * 4}</p>
<p>Item 1A. Risk Factors</p>
<p>{risk}</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>{mdna}</p>
<p>The following table summarizes revenue by segment (in millions):</p>
<table>
  <tr><td></td><td>Fiscal 2025</td><td></td><td>Fiscal 2024</td></tr>
  <tr><td>Data Center</td><td>$</td><td>{revenue}</td><td>$</td><td>47,525</td></tr>
  <tr><td>Gaming</td><td>$</td><td>10,447</td><td>$</td><td>(2,100</td><td>)</td></tr>
  <tr><td>Total</td><td>$</td><td>71,369</td><td>$</td><td>45,425</td></tr>
</table>
<p>Item 8. Financial Statements and Supplementary Data</p>
<p>{PROSE * 3}</p>
</body></html>"""


def meta(ticker: str = "ACME", fy: int = 2025, company: str = "Acme Compute") -> FilingMeta:
    return FilingMeta(
        ticker=ticker,
        company=company,
        cik="0000000001",
        form="10-K",
        fiscal_year=fy,
        period_of_report=f"{fy}-01-26",
        filing_date=f"{fy}-02-26",
        accession="0000000001-25-000001",
        url="https://example.com/x.htm",
    )


@pytest.fixture
def parsed():
    return parse_filing(mini_10k_html(), meta())


@pytest.fixture
def settings():
    return load_settings()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder (a stand-in for a real model)."""

    name = "hash"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class OverlapReranker:
    name = "overlap"

    def score(self, query, passages):
        q = set(re.findall(r"[a-z]+", query.lower()))
        return [len(q & set(re.findall(r"[a-z]+", p.lower()))) for p in passages]


class FakeAnthropic:
    """Mimics anthropic.Anthropic().messages.create with a scripted reply function."""

    def __init__(self, reply):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._reply = reply

    def _create(self, **request):
        self.calls.append(request)
        text = self._reply(request)
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=100,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=usage)
