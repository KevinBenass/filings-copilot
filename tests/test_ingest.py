from filings_copilot.config import ChunkingConfig
from filings_copilot.ingest.chunking import chunk_document, n_tokens
from filings_copilot.ingest.edgar import fiscal_year_from_period
from filings_copilot.ingest.parse import _merge_split_cells


def test_hidden_xbrl_and_styles_removed(parsed):
    assert "hidden xbrl" not in parsed.text
    assert ".x{" not in parsed.text


def test_table_cells_merged_and_rendered_as_markdown(parsed):
    assert "|  | Fiscal 2025 | Fiscal 2024 |" in parsed.text
    assert "| Data Center | $60,922 | $47,525 |" in parsed.text
    assert "$(2,100)" in parsed.text


def test_merge_split_cells():
    assert _merge_split_cells(["Revenue", "$", "1,234", "", "(", "56", ")", "12", "%"]) == [
        "Revenue",
        "$1,234",
        "(56)",
        "12%",
    ]


def test_sections_skip_table_of_contents(parsed):
    items = [s.item for s in parsed.sections]
    assert items == ["0", "1", "1A", "7", "8"]
    mdna = next(s for s in parsed.sections if s.item == "7")
    assert parsed.text[mdna.start : mdna.end].startswith("Item 7.")
    assert "Data Center | $60,922" in parsed.text[mdna.start : mdna.end]
    # Sections tile the document without gaps or overlaps.
    for a, b in zip(parsed.sections, parsed.sections[1:], strict=False):
        assert a.end == b.start


def test_fiscal_year_label():
    assert fiscal_year_from_period("2025-01-26") == 2025
    assert fiscal_year_from_period("2024-09-28") == 2024


def test_structure_chunks_keep_tables_whole_with_caption(parsed):
    chunks = chunk_document(
        parsed, ChunkingConfig(strategy="structure", max_tokens=120, context="metadata")
    )
    tables = [c for c in chunks if c.is_table and c.item == "7"]
    assert len(tables) == 1
    assert tables[0].text.startswith("The following table summarizes revenue")
    assert "| Total | $71,369 | $45,425 |" in tables[0].text
    assert tables[0].item == "7"


def test_chunk_offsets_point_into_document(parsed):
    for strategy in ("structure", "fixed"):
        chunks = chunk_document(
            parsed,
            ChunkingConfig(strategy=strategy, max_tokens=80, overlap_tokens=10, context="none"),
        )
        assert chunks
        for c in chunks:
            span = parsed.text[c.start : c.end]
            # The chunk's first and last words come from its span.
            assert c.text.split()[0] in span and c.text.split()[-1] in span
            assert c.context == ""


def test_structure_chunks_respect_budget(parsed):
    cfg = ChunkingConfig(strategy="structure", max_tokens=80, context="none")
    for c in chunk_document(parsed, cfg):
        assert n_tokens(c.text) <= cfg.max_tokens * 1.6


def test_metadata_context(parsed):
    c = next(c for c in chunk_document(parsed, ChunkingConfig(context="metadata")) if c.item == "7")
    assert c.context.startswith(
        "Acme Compute (ACME) · Form 10-K · fiscal year 2025 · Item 7. Management"
    )
    assert c.index_text.startswith(c.context)
