"""LLM-written situating context for each chunk ("contextual retrieval").

A chunk like "Revenue increased 12% driven by higher volumes" is ambiguous on its own. We
ask a small model to write one or two sentences that situate it (which company, segment,
period, what the surrounding discussion is about) and prepend that to the indexed text.

Cost control: the surrounding section text is sent as a cached prompt prefix and shared by
every chunk in the same window, so each extra chunk pays ~10% of the window's input price.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from rich.progress import track

from filings_copilot.ingest.chunking import n_tokens
from filings_copilot.llm import LLM
from filings_copilot.schemas import Chunk, ParsedFiling

WINDOW_CHARS = 24_000

PROMPT = """<chunk>
{chunk}
</chunk>

The chunk above comes from the document excerpt given earlier. Write one or two short
sentences that situate this chunk within the filing to improve search retrieval: name the
company, the fiscal period, the business segment or topic, and what a reader would need to
know to interpret any numbers (units, comparison period). Answer with the sentences only."""


def _prefix(doc: ParsedFiling, section_start: int, section_end: int, window: int) -> str:
    lo = section_start + window * WINDOW_CHARS
    hi = min(section_end, lo + WINDOW_CHARS)
    m = doc.meta
    return (
        f'<document company="{m.company}" ticker="{m.ticker}" form="{m.form}" '
        f'fiscal_year="{m.fiscal_year}">\n{doc.text[lo:hi]}\n</document>'
    )


def _windows(doc: ParsedFiling, chunks: list[Chunk]) -> dict[tuple[int, int, int], list[Chunk]]:
    groups: dict[tuple[int, int, int], list[Chunk]] = defaultdict(list)
    for c in chunks:
        sec = next(s for s in doc.sections if s.start <= c.start < s.end)
        groups[(sec.start, sec.end, (c.start - sec.start) // WINDOW_CHARS)].append(c)
    return groups


def estimate_cost(
    docs: list[ParsedFiling], chunks_by_doc: dict[str, list[Chunk]], llm: LLM
) -> float:
    model = llm.cfg.fast_model
    price = llm.cfg.prices[model]
    total = 0.0
    for doc in docs:
        for (s, e, w), group in _windows(doc, chunks_by_doc[doc.meta.doc_id]).items():
            window_tokens = n_tokens(_prefix(doc, s, e, w))
            first = window_tokens * 1.25 * price.input
            rest = (len(group) - 1) * window_tokens * 0.10 * price.input
            chunk_in = sum(n_tokens(c.text) + 120 for c in group) * price.input
            out = len(group) * 70 * price.output
            total += (first + rest + chunk_in + out) / 1e6
    return total


def contextualize(
    docs: list[ParsedFiling], chunks_by_doc: dict[str, list[Chunk]], llm: LLM, workers: int = 8
) -> None:
    """Prepend LLM context to every chunk in place (after the metadata header)."""
    for doc in track(docs, description="Contextualizing"):
        for (s, e, w), group in _windows(doc, chunks_by_doc[doc.meta.doc_id]).items():
            prefix = _prefix(doc, s, e, w)

            def run(chunk: Chunk, prefix: str = prefix) -> None:
                resp = llm.complete(
                    PROMPT.format(chunk=chunk.text),
                    model=llm.cfg.fast_model,
                    cached_prefix=prefix,
                    max_tokens=150,
                )
                header = chunk.context
                chunk.context = f"{header}\n{resp.text.strip()}" if header else resp.text.strip()

            run(group[0])  # first call writes the prompt cache
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(run, group[1:]))
