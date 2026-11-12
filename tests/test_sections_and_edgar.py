"""Section detection on the layouts real filers use, and EDGAR pagination."""

from conftest import meta

from filings_copilot.eval.report import CORPUS_END, CORPUS_START, write_corpus_table
from filings_copilot.ingest.edgar import EdgarClient
from filings_copilot.ingest.parse import find_sections, parse_filing

FILLER = "The Company discusses its results and operations in detail here. " * 12


def _items(text: str) -> list[str]:
    return [s.item for s in find_sections(text)]


def test_headings_in_layout_tables_are_found():
    """Headings rendered as two-cell table rows ('Item 7.' | 'Management's ...')."""
    toc = "\n".join(
        f"| Item {i}. | Title {i} | {n} |" for i, n in [("1", 3), ("1A", 9), ("7", 30), ("8", 50)]
    )
    body = "\n\n".join(
        f"| Item {i}. | Title {i} |\n|---|---|\n\n{FILLER}" for i in ["1", "1A", "7", "8"]
    )
    text = f"Cover page\n\n{toc}\n\n{FILLER}\n\n{body}"
    sections = find_sections(text)
    assert [s.item for s in sections] == ["0", "1", "1A", "7", "8"]
    # The body heading wins over the table-of-contents row.
    item7 = next(s for s in sections if s.item == "7")
    assert item7.start > text.index(FILLER)


def test_headings_without_space_and_invisible_characters():
    html = (
        "<html><body>"
        + "".join(
            f"<div><span>Item</span><span>​{i}.</span><span>Title</span></div><p>{FILLER}</p>"
            for i in ["1", "1A", "2", "7", "8"]
        )
        + "</body></html>"
    )
    doc = parse_filing(html, meta())
    assert [s.item for s in doc.sections] == ["1", "1A", "2", "7", "8"]  # no cover page here


def test_title_fallback_for_annual_report_style_10k():
    """Bank-style 10-K: no 'Item 7.' heading; MD&A located by title, page headers ignored."""
    text = "\n\n".join(
        [
            "Form 10-K cross-reference index",
            "| Item 7. | Management's discussion and analysis | 50 |",
            "Item 1. Business",
            FILLER,
            "Item 1A. Risk Factors",
            FILLER,
            "Management's discussion and analysis",
            FILLER,
            "Management's discussion and analysis",  # running page header on the next page
            FILLER,
            "Market Risk Management",
            FILLER,
            "Report of Independent Registered Public Accounting Firm",
            FILLER,
        ]
    )
    sections = find_sections(text)
    assert [s.item for s in sections] == ["0", "1", "1A", "7", "7A", "8"]
    mdna = next(s for s in sections if s.item == "7")
    assert text[mdna.start :].startswith("Management's discussion and analysis")
    assert mdna.end == text.index("Market Risk Management")


def test_regular_filing_unaffected_by_fallbacks(parsed):
    assert [s.item for s in parsed.sections] == ["0", "1", "1A", "7", "8"]


class FakeEdgar(EdgarClient):
    """Recent page holds only FY2025; older 10-Ks sit on a paginated history file."""

    def __init__(self):
        super().__init__("Test test@example.com", min_interval_s=0)
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        recent = {
            "form": ["424B2"] * 3 + ["10-K"],
            "reportDate": ["", "", "", "2025-12-31"],
            "filingDate": ["2026-03-01"] * 3 + ["2026-02-14"],
            "accessionNumber": ["a", "b", "c", "0000019617-26-000010"],
            "primaryDocument": ["x"] * 3 + ["jpm-20251231.htm"],
        }
        older = {
            "form": ["10-K", "10-K", "10-K"],
            "reportDate": ["2024-12-31", "2023-12-31", "2022-12-31"],
            "filingDate": ["2025-02-14", "2024-02-16", "2023-02-21"],
            "accessionNumber": [
                "0000019617-25-000010",
                "0000019617-24-000010",
                "0000019617-23-000010",
            ],
            "primaryDocument": ["jpm-20241231.htm", "jpm-20231231.htm", "jpm-20221231.htm"],
        }
        payload = (
            {
                "filings": {
                    "recent": recent,
                    "files": [
                        {"name": "CIK0000019617-submissions-001.json", "filingTo": "2025-12-31"}
                    ],
                }
            }
            if url.endswith("CIK0000019617.json")
            else older
        )
        return type("R", (), {"json": lambda self: payload})()


def test_annual_filings_follow_pagination():
    client = FakeEdgar()
    years = sorted(
        m.fiscal_year
        for m in client.annual_filings("JPM", "0000019617", "JPMorgan", "10-K", [2023, 2024, 2025])
    )
    assert years == [2023, 2024, 2025]
    assert any("submissions-001" in u for u in client.urls)


def test_corpus_table_written_to_readme(parsed, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"x\n{CORPUS_START}\nold\n{CORPUS_END}\ny")
    write_corpus_table([parsed], tmp_path / "results", readme)
    text = readme.read_text()
    assert (
        "old" not in text
        and "[FY2025](https://example.com/x.htm)" in text
        and "1 annual reports" in text
    )


def test_financial_section_appended_at_end():
    """Exxon-style: Items 7, 7A, 8 are pointers; the real content follows Part IV."""
    pointer = "Reference is made to the Financial Section of this report. " * 8  # > 400 chars
    text = "\n\n".join(
        [
            "Item 1. Business",
            FILLER,
            "Item 1A. Risk Factors",
            FILLER,
            "Item 7. Management's Discussion and Analysis",
            "See the Financial Section.",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            pointer,
            "Item 8. Financial Statements and Supplementary Data",
            pointer,
            "Item 9A. Controls and Procedures",
            FILLER,
            "Item 16. Form 10-K Summary",
            "None.\n" + FILLER,
            "FINANCIAL SECTION",
            "MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
            FILLER * 3,
            "MARKET RISKS, INFLATION AND OTHER UNCERTAINTIES",
            FILLER,
            "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM",
            FILLER * 4,
            "CONSOLIDATED STATEMENT OF INCOME",
            FILLER,
        ]
    )
    sections = find_sections(text)
    assert [s.item for s in sections] == ["1", "1A", "9A", "16", "7", "7A", "8"]
    by_item = {s.item: s for s in sections}
    assert text[by_item["7"].start :].startswith("MANAGEMENT’S DISCUSSION")
    assert text[by_item["8"].start :].startswith("REPORT OF INDEPENDENT")
    assert by_item["8"].end == len(text)


def test_running_page_headers_do_not_move_sections():
    """Microsoft-style: 'PART II Item 7' repeats at the top of every MD&A page."""
    page = FILLER  # one page of text, > 400 chars
    text = "\n".join(
        [
            "Item 1. Business",
            page,
            "Item 7. Management's Discussion and Analysis",
            page,
            "Item 7",
            page,  # running header, page 2
            "Item 7",
            page + page,  # running header, page 3 (the longest span)
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
            page,
            "Item 8. Financial Statements and Supplementary Data",
            page,
            "Item 8",
            page + page + page,
        ]
    )
    by_item = {s.item: s for s in find_sections(text)}
    assert text[by_item["7"].start :].startswith("Item 7. Management's")
    assert by_item["7"].end == text.index("Item 7A.")
    assert text[by_item["8"].start :].startswith("Item 8. Financial")
