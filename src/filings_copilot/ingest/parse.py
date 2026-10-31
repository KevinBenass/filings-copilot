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

ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[\.:\-–—]?\s*(.{0,150})$", re.IGNORECASE)
MIN_SECTION_CHARS = 400


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

    raw = soup.get_text("")
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


def find_sections(text: str) -> list[Section]:
    candidates: list[tuple[int, str, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if not line.lstrip().startswith("|"):
            m = ITEM_RE.match(line.rstrip("\n"))
            if m:
                candidates.append((offset, m.group(1).upper(), m.group(2).strip()))
        offset += len(line)

    if not candidates:
        return [Section(item="0", title="Full document", start=0, end=len(text))]

    # Span of each candidate = distance to the next candidate.
    spans = [
        (
            start,
            item,
            title,
            (candidates[i + 1][0] if i + 1 < len(candidates) else len(text)) - start,
        )
        for i, (start, item, title) in enumerate(candidates)
    ]
    best: dict[str, tuple[int, str, int]] = {}
    for start, item, title, length in spans:
        if length < MIN_SECTION_CHARS:
            continue  # table-of-contents line or a cross-reference
        if item not in best or length > best[item][2]:
            best[item] = (start, title, length)

    chosen = sorted((start, item, title) for item, (start, title, _) in best.items())
    sections: list[Section] = []
    if chosen and chosen[0][0] > 0:
        sections.append(
            Section(item="0", title="Cover and Part I preamble", start=0, end=chosen[0][0])
        )
    for i, (start, item, title) in enumerate(chosen):
        end = chosen[i + 1][0] if i + 1 < len(chosen) else len(text)
        canonical = ITEM_TITLES.get(item, title or f"Item {item}")
        sections.append(Section(item=item, title=f"Item {item}. {canonical}", start=start, end=end))
    return sections


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
