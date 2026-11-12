"""Turn 10-K inline-XBRL HTML into clean text with Markdown tables and item-level sections.

Design choices
- Tables are rendered as Markdown so both the LLM and BM25 see row/column structure, and
  so the chunker can keep a table together instead of slicing through it.
- EDGAR splits currency symbols and parentheses into their own cells ("$" | "1,234" | ")");
  these are merged back so numbers stay intact ("$1,234", "(56)").
- Section boundaries are found from "Item N." headings. Table-of-contents entries are
  filtered out because they are table rows or because their "section" is implausibly short.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from filings_copilot.schemas import FilingMeta, ParsedFiling, Section

BLOCK_TAGS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}

ITEM_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_RE = re.compile(
    r"^\s*item\s*(\d{1,2}[a-cA-C]?)(?![0-9])\s*[\.:\-–—]?\s*(.{0,150})$", re.IGNORECASE
)
MIN_SECTION_CHARS = 400
MIN_ITEMS_BEFORE_FALLBACK = 5
INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)

# Some filers (large banks especially) publish the 10-K as their annual report and map
# items to it through a cross-reference index, so "Item 7." never appears as a heading.
# For those, sections are located by their conventional titles instead.
TITLE_FALLBACK: dict[str, list[str]] = {
    "7": ["management's discussion and analysis", "management’s discussion and analysis"],
    "7A": ["market risk management", "quantitative and qualitative disclosures about market risk"],
    "8": [
        "report of independent registered public accounting firm",
        "consolidated statements of income",
        "consolidated financial statements",
    ],
}


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _merge_split_cells(cells: list[str]) -> list[str]:
    """Merge EDGAR's split currency / parenthesis / percent cells into their numbers."""
    out: list[str] = [""] if cells and not cells[0] else []  # keep header-row label column
    pending_prefix = ""
    for cell in cells:
        if not cell:
            continue
        if cell in {"$", "€", "£", "(", "$(", "($"}:
            pending_prefix += cell
            continue
        if cell in {")", "%", ")%", "%)"} and out:
            out[-1] += cell
            continue
        out.append(pending_prefix + cell)
        pending_prefix = ""
    return out


def table_to_markdown(table: Tag) -> str | None:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_clean_cell(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
        merged = _merge_split_cells(cells)
        if merged:
            rows.append(merged)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    if width < 2 or len(rows) < 2:
        # Layout table (a single column of prose). Return its text unformatted.
        return "\n".join(" ".join(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    esc = [[c.replace("|", "/") for c in r] for r in rows]
    lines = ["| " + " | ".join(esc[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in esc[1:]]
    return "\n".join(lines)


def _is_hidden(tag: Tag) -> bool:
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "display:none" in style


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head", "title"]):
        t.decompose()
    for t in soup.find_all(
        lambda x: isinstance(x, Tag) and x.name and x.name.startswith("ix:header")
    ):
        t.decompose()
    for t in soup.find_all(_is_hidden):
        t.decompose()

    for table in soup.find_all("table"):
        if table.find_parent("table") is not None:
            continue  # handled by the outer table
        md = table_to_markdown(table)
        table.replace_with(NavigableString(f"\n\n{md}\n\n" if md else "\n"))

    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after(NavigableString("\n"))

    raw = soup.get_text("").translate(INVISIBLE)
    lines: list[str] = []
    for line in raw.replace("\xa0", " ").splitlines():
        line = (
            line.strip() if line.lstrip().startswith("|") else re.sub(r"[ \t]+", " ", line).strip()
        )
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Drop bare page numbers and running "Table of Contents" links between pages.
    text = re.sub(r"\n(?:\d{1,3}|Table of Contents)\n", "\n", text, flags=re.IGNORECASE)
    return text.strip()


def _table_row_heading(line: str) -> str | None:
    """'| Item 7. | Management's Discussion ... |' -> 'Item 7. Management's Discussion ...'."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    cells = [c for c in cells if c and not re.fullmatch(r"\d{1,3}", c)]  # drop page numbers
    return " ".join(cells) if cells else None


def _candidates(text: str, include_tables: bool) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        if body.lstrip().startswith("|"):
            body = _table_row_heading(body) if include_tables else None
        if body:
            m = ITEM_RE.match(body)
            if m:
                out.append((offset, m.group(1).upper(), m.group(2).strip()))
        offset += len(line)
    return out


def _pick(
    candidates: list[tuple[int, str, str]], text_len: int, prefer_last: bool
) -> dict[str, tuple[int, str]]:
    """Choose one heading per item.

    A candidate's span runs to the next candidate; table-of-contents lines and inline
    cross-references have short spans and are discarded. Normally the longest span wins.
    When table rows are included, the table of contents itself can produce long spans,
    so the last qualifying occurrence (the body heading, after the TOC) wins instead.
    """
    best: dict[str, tuple[int, str, int]] = {}
    for i, (start, item, title) in enumerate(candidates):
        length = (candidates[i + 1][0] if i + 1 < len(candidates) else text_len) - start
        if length < MIN_SECTION_CHARS:
            continue
        if item not in best or prefer_last or length > best[item][2]:
            best[item] = (start, title, length)
    return {item: (start, title) for item, (start, title, _) in best.items()}


def _title_fallback(text: str, chosen: dict[str, tuple[int, str]]) -> dict[str, tuple[int, str]]:
    """Fill missing MD&A / market risk / financial statements by their usual titles.

    Only short, heading-like lines outside tables count, and only the first occurrence
    (later ones are running page headers). Order is enforced: 7 < 7A < 8.
    """
    out = dict(chosen)
    line_starts: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and not stripped.startswith("|") and len(stripped) <= 90:
            line_starts.append((offset, stripped.lower().rstrip(".:")))
        offset += len(line)
    floor = 0
    for item in ("7", "7A", "8"):
        if item in out:
            floor = out[item][0]
            continue
        titles = TITLE_FALLBACK[item]
        hit = next(
            (
                pos
                for pos, low in line_starts
                if pos > floor and any(low.startswith(t) for t in titles)
            ),
            None,
        )
        if hit is not None:
            out[item] = (hit, "")
            floor = hit
    return out


def find_sections(text: str) -> list[Section]:
    chosen = _pick(_candidates(text, include_tables=False), len(text), prefer_last=False)
    if len(chosen) < MIN_ITEMS_BEFORE_FALLBACK:
        with_tables = _pick(_candidates(text, include_tables=True), len(text), prefer_last=True)
        if len(with_tables) > len(chosen):
            chosen = with_tables
    if not {"7", "8"} <= set(chosen):
        chosen = _title_fallback(text, chosen)

    if not chosen:
        return [Section(item="0", title="Full document", start=0, end=len(text))]

    ordered = sorted((start, item, title) for item, (start, title) in chosen.items())
    sections: list[Section] = []
    if ordered[0][0] > 0:
        sections.append(
            Section(item="0", title="Cover and Part I preamble", start=0, end=ordered[0][0])
        )
    for i, (start, item, title) in enumerate(ordered):
        end = ordered[i + 1][0] if i + 1 < len(ordered) else len(text)
        canonical = ITEM_TITLES.get(item, title or f"Item {item}")
        sections.append(Section(item=item, title=f"Item {item}. {canonical}", start=start, end=end))
    return sections


def section_report(text: str, pattern: str, context: int = 80, limit: int = 15) -> list[str]:
    """Lines matching ``pattern`` with surrounding text, to see how a filer formats headings."""
    out = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        lo, hi = max(0, m.start() - context), min(len(text), m.end() + context)
        out.append(f"@{m.start():>8,}  " + repr(text[lo:hi]))
        if len(out) >= limit:
            break
    return out


def parse_filing(html: str, meta: FilingMeta) -> ParsedFiling:
    text = html_to_text(html)
    return ParsedFiling(meta=meta, text=text, sections=find_sections(text))


def parse_corpus(raw_dir: Path, parsed_dir: Path) -> list[Path]:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for meta_path in sorted(raw_dir.glob("*.json")):
        meta = FilingMeta.model_validate(json.loads(meta_path.read_text()))
        html = meta_path.with_suffix(".html").read_text(errors="ignore")
        parsed = parse_filing(html, meta)
        dest = parsed_dir / f"{meta.doc_id}.json"
        dest.write_text(parsed.model_dump_json())
        out.append(dest)
    return out


def load_parsed(parsed_dir: Path) -> list[ParsedFiling]:
    return [
        ParsedFiling.model_validate_json(p.read_text()) for p in sorted(parsed_dir.glob("*.json"))
    ]
