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
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
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

    def _submission_pages(self, cik: str, min_year: int) -> Iterator[dict]:
        """Yield columnar filing tables: the recent page, then older pages as needed.

        The "recent" page holds at most ~1,000 filings. Prolific filers (banks issuing
        structured notes) push their older 10-Ks onto the paginated history files.
        """
        data = self.get(SUBMISSIONS_URL.format(cik=cik)).json()
        self.last_entity_name = data.get("name", "")
        yield data["filings"]["recent"]
        for page in data["filings"].get("files", []):
            if page.get("filingTo", "9999")[:4] < str(min_year):
                break
            yield self.get(SUBMISSIONS_PAGE_URL.format(name=page["name"])).json()

    def annual_filings(
        self, ticker: str, cik: str, company: str, form: str, fiscal_years: list[int]
    ) -> Iterator[FilingMeta]:
        seen: set[int] = set()
        for table in self._submission_pages(cik, min(fiscal_years)):
            for i, f in enumerate(table["form"]):
                if f != form:
                    continue
                period = table["reportDate"][i]
                if not period:
                    continue
                fy = fiscal_year_from_period(period)
                if fy not in fiscal_years or fy in seen:
                    continue
                seen.add(fy)
                acc = table["accessionNumber"][i]
                yield FilingMeta(
                    ticker=ticker,
                    company=company,
                    cik=cik,
                    form=form,
                    fiscal_year=fy,
                    period_of_report=period,
                    filing_date=table["filingDate"][i],
                    accession=acc,
                    url=ARCHIVE_URL.format(
                        cik_int=int(cik), acc=acc.replace("-", ""), doc=table["primaryDocument"][i]
                    ),
                )
            if seen >= set(fiscal_years):
                return


def fiscal_year_from_period(period_of_report: str) -> int:
    """Fiscal year label from the period end date.

    The calendar year in which the fiscal year ends matches how most filers label it
    (Microsoft's FY2025 ends June 2025, NVIDIA's FY2025 ends January 2025).
    """
    return int(period_of_report[:4])


def download_corpus(
    tickers: list[str],
    form: str,
    fiscal_years: list[int],
    user_agent: str,
    raw_dir: Path,
    ciks: dict[str, str] | None = None,
) -> tuple[list[Path], dict[str, str]]:
    """Download filings as raw HTML with a sidecar metadata JSON. Idempotent.

    Returns the written paths and, per ticker, why some fiscal years are missing.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = EdgarClient(user_agent)
    ciks = {t.upper(): str(c).zfill(10) for t, c in (ciks or {}).items()}
    tmap = client.ticker_map() if any(t not in ciks for t in tickers) else {}
    written: list[Path] = []
    missing: dict[str, str] = {}
    for ticker in tickers:
        if ticker in ciks:
            cik, company = ciks[ticker], ""
        elif ticker in tmap:
            cik, company = tmap[ticker]
        else:
            missing[ticker] = "ticker not in the SEC ticker list; add its CIK under corpus.ciks"
            continue
        found: set[int] = set()
        for meta in client.annual_filings(ticker, cik, company, form, fiscal_years):
            if not meta.company:
                meta.company = client.last_entity_name or ticker
            html_path = raw_dir / f"{meta.doc_id}.html"
            meta_path = raw_dir / f"{meta.doc_id}.json"
            if not html_path.exists():
                html_path.write_bytes(client.get(meta.url).content)
            meta_path.write_text(json.dumps(meta.model_dump(), indent=2))
            written.append(html_path)
            found.add(meta.fiscal_year)
        if gaps := sorted(set(fiscal_years) - found):
            missing[ticker] = (
                f"no {form} on EDGAR for fiscal years {', '.join(map(str, gaps))} (CIK {cik})"
            )
    return written, missing
