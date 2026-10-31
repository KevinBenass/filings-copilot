"""Two chunking strategies, compared in the ablation.

fixed      Sliding window of ~N tokens with overlap over the whole document. The common
           baseline: ignores sections, cuts tables in half.
structure  Packs paragraphs within a 10-K item up to N tokens. Tables stay whole (or are
           split by rows with the header repeated), and a short caption paragraph like
           "The following table presents revenue by segment" is attached to its table.

Every chunk keeps character offsets into the parsed document, which is what lets the eval
judge relevance by span overlap instead of by chunk id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from filings_copilot.config import ChunkingConfig
from filings_copilot.schemas import Chunk, ParsedFiling, Section

TOKENS_PER_WORD = 1.3
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def n_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD) + 1


@dataclass
class Block:
    start: int
    end: int
    text: str
    is_table: bool


def split_blocks(text: str, base: int) -> list[Block]:
    """Split a section into paragraph and table blocks with absolute offsets."""
    blocks: list[Block] = []
    for m in re.finditer(r"(?:[^\n]+\n?)+", text):
        raw = m.group(0).rstrip("\n")
        if not raw.strip():
            continue
        lines = raw.split("\n")
        is_table = sum(ln.lstrip().startswith("|") for ln in lines) >= max(2, len(lines) // 2)
        blocks.append(Block(base + m.start(), base + m.start() + len(raw), raw, is_table))
    return blocks


def _split_long_paragraph(block: Block, max_tokens: int) -> list[Block]:
    pieces: list[Block] = []
    cursor = 0
    current_start = 0
    sentences = SENTENCE_RE.split(block.text)
    buf = ""
    for sent in sentences:
        idx = block.text.find(sent, cursor)
        cursor = idx + len(sent)
        if buf and n_tokens(buf + " " + sent) > max_tokens:
            pieces.append(Block(block.start + current_start, block.start + idx, buf.strip(), False))
            current_start, buf = idx, sent
        else:
            buf = f"{buf} {sent}" if buf else sent
    if buf:
        pieces.append(Block(block.start + current_start, block.end, buf.strip(), False))
    return pieces


def _split_table(block: Block, max_tokens: int) -> list[Block]:
    lines = block.text.split("\n")
    header, rows = lines[:2], lines[2:]
    header_text = "\n".join(header)
    pieces: list[Block] = []
    buf: list[str] = []
    offset = block.start + len(header_text) + 1
    piece_start = offset
    for row in rows:
        if buf and n_tokens(header_text + "\n" + "\n".join([*buf, row])) > max_tokens:
            pieces.append(Block(piece_start, offset - 1, header_text + "\n" + "\n".join(buf), True))
            buf, piece_start = [], offset
        buf.append(row)
        offset += len(row) + 1
    if buf:
        pieces.append(Block(piece_start, block.end, header_text + "\n" + "\n".join(buf), True))
    return pieces


def _normalize(blocks: list[Block], max_tokens: int) -> list[Block]:
    out: list[Block] = []
    for b in blocks:
        if n_tokens(b.text) <= max_tokens:
            out.append(b)
        elif b.is_table:
            out.extend(_split_table(b, int(max_tokens * 1.5)))  # tables tolerate a bit more
        else:
            out.extend(_split_long_paragraph(b, max_tokens))
    return out


def structure_chunks(doc: ParsedFiling, cfg: ChunkingConfig) -> list[tuple[Section, Block]]:
    result: list[tuple[Section, Block]] = []
    for section in doc.sections:
        blocks = _normalize(
            split_blocks(doc.text[section.start : section.end], section.start), cfg.max_tokens
        )
        packed: list[Block] = []
        for b in blocks:
            prev = packed[-1] if packed else None
            if b.is_table:
                # Attach a short caption paragraph ("The following table ...") to the table.
                if (
                    prev
                    and not prev.is_table
                    and n_tokens(prev.text) < 80
                    and prev.text.rstrip().endswith(":")
                ):
                    packed[-1] = Block(prev.start, b.end, f"{prev.text}\n\n{b.text}", True)
                else:
                    packed.append(b)
                continue
            is_caption = b.text.rstrip().endswith(":") and n_tokens(b.text) < 80
            fits = prev is not None and n_tokens(prev.text) + n_tokens(b.text) <= cfg.max_tokens
            # A heading or other tiny block always joins the block that follows it.
            prev_tiny = prev is not None and n_tokens(prev.text) < cfg.min_tokens
            if prev and not prev.is_table and not is_caption and (fits or prev_tiny):
                packed[-1] = Block(prev.start, b.end, f"{prev.text}\n\n{b.text}", False)
            else:
                packed.append(b)
        # Merge a tiny trailing prose chunk into its predecessor.
        if (
            len(packed) >= 2
            and not packed[-1].is_table
            and not packed[-2].is_table
            and n_tokens(packed[-1].text) < cfg.min_tokens
        ):
            a, b = packed[-2], packed[-1]
            packed[-2:] = [Block(a.start, b.end, f"{a.text}\n\n{b.text}", False)]
        result.extend((section, b) for b in packed)
    return result


def fixed_chunks(doc: ParsedFiling, cfg: ChunkingConfig) -> list[tuple[Section, Block]]:
    words = list(re.finditer(r"\S+", doc.text))
    size = max(1, int(cfg.max_tokens / TOKENS_PER_WORD))
    step = max(1, size - int(cfg.overlap_tokens / TOKENS_PER_WORD))
    result: list[tuple[Section, Block]] = []
    for i in range(0, len(words), step):
        window = words[i : i + size]
        if not window:
            break
        start, end = window[0].start(), window[-1].end()
        text = doc.text[start:end]
        section = next((s for s in doc.sections if s.start <= start < s.end), doc.sections[-1])
        result.append((section, Block(start, end, text, text.count("\n|") >= 2)))
        if i + size >= len(words):
            break
    return result


def metadata_context(doc: ParsedFiling, section: Section) -> str:
    m = doc.meta
    return (
        f"{m.company} ({m.ticker}) · Form {m.form} · fiscal year {m.fiscal_year} · {section.title}"
    )


def chunk_document(doc: ParsedFiling, cfg: ChunkingConfig) -> list[Chunk]:
    pairs = structure_chunks(doc, cfg) if cfg.strategy == "structure" else fixed_chunks(doc, cfg)
    chunks: list[Chunk] = []
    for i, (section, block) in enumerate(pairs):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.meta.doc_id}::{cfg.strategy}::{i:04d}",
                doc_id=doc.meta.doc_id,
                ticker=doc.meta.ticker,
                company=doc.meta.company,
                fiscal_year=doc.meta.fiscal_year,
                item=section.item,
                section_title=section.title,
                start=block.start,
                end=block.end,
                text=block.text,
                is_table=block.is_table,
                context=metadata_context(doc, section) if cfg.context != "none" else "",
            )
        )
    return chunks
