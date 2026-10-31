"""Download annual reports from SEC EDGAR.

Uses the public submissions API (no key needed). The SEC requires a descriptive
User-Agent and asks clients to stay under 10 requests/second.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from filings_copilot.schemas import FilingMeta

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}"


class EdgarClient:
    def __init__(self, user_agent: str, min_interval_s: float = 0.15) -> None:
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires a User-Agent containing a contact email. "
                "Set SEC_USER_AGENT='Your Name you@example.com'."
            )
        self._client = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=60.0,
            follow_redirects=True,
        )
        self._min_interval = min_interval_s
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=20))
    def get(self, url: str) -> httpx.Response:
        self._throttle()
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp

    def ticker_map(self) -> dict[str, tuple[str, str]]:
        """Map ticker -> (zero-padded CIK, company title)."""
        data = self.get(TICKERS_URL).json()
        return {
            row["ticker"].upper(): (str(row["cik_str"]).zfill(10), row["title"])
            for row in data.values()
        }

    def annual_filings(
        self, ticker: str, cik: str, company: str, form: str, fiscal_years: list[int]
    ) -> Iterator[FilingMeta]:
        recent = self.get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        seen: set[int] = set()
        for i, f in enumerate(recent["form"]):
            if f != form:
                continue
            period = recent["reportDate"][i]
            fy = fiscal_year_from_period(period)
            if fy not in fiscal_years or fy in seen:
                continue
            seen.add(fy)
            acc = recent["accessionNumber"][i]
            yield FilingMeta(
                ticker=ticker,
                company=company,
                cik=cik,
                form=form,
                fiscal_year=fy,
                period_of_report=period,
                filing_date=recent["filingDate"][i],
                accession=acc,
                url=ARCHIVE_URL.format(
                    cik_int=int(cik), acc=acc.replace("-", ""), doc=recent["primaryDocument"][i]
                ),
            )


def fiscal_year_from_period(period_of_report: str) -> int:
    """Fiscal year label from the period end date.

    The calendar year in which the fiscal year ends matches how most filers label it
    (Microsoft's FY2025 ends June 2025, NVIDIA's FY2025 ends January 2025).
    """
    return int(period_of_report[:4])


def download_corpus(
    tickers: list[str], form: str, fiscal_years: list[int], user_agent: str, raw_dir: Path
) -> list[Path]:
    """Download filings as raw HTML with a sidecar metadata JSON. Idempotent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = EdgarClient(user_agent)
    tmap = client.ticker_map()
    written: list[Path] = []
    for ticker in tickers:
        if ticker not in tmap:
            raise KeyError(f"Ticker {ticker} not found in SEC ticker map")
        cik, company = tmap[ticker]
        for meta in client.annual_filings(ticker, cik, company, form, fiscal_years):
            html_path = raw_dir / f"{meta.doc_id}.html"
            meta_path = raw_dir / f"{meta.doc_id}.json"
            if not html_path.exists():
                html_path.write_bytes(client.get(meta.url).content)
            meta_path.write_text(json.dumps(meta.model_dump(), indent=2))
            written.append(html_path)
    return written
