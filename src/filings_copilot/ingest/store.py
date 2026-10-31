"""Build and persist chunk sets, one JSONL file per chunking variant."""

from __future__ import annotations

from pathlib import Path

from filings_copilot.config import ChunkingConfig, Settings
from filings_copilot.ingest.chunking import chunk_document
from filings_copilot.ingest.contextualize import contextualize, estimate_cost
from filings_copilot.ingest.parse import load_parsed
from filings_copilot.llm import LLM
from filings_copilot.schemas import Chunk


def chunks_path(settings: Settings, chunking: ChunkingConfig | None = None) -> Path:
    c = chunking or settings.chunking
    return settings.resolve(c.chunks_dir) / f"{c.variant}.jsonl"


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open() as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def build_chunks(
    settings: Settings, llm: LLM | None = None, dry_run: bool = False
) -> tuple[Path, float]:
    """Chunk the parsed corpus for the configured variant. Returns (path, cost_estimate)."""
    docs = load_parsed(settings.resolve(settings.corpus.parsed_dir))
    if not docs:
        raise FileNotFoundError(
            "No parsed filings found. Run `fcopilot download` and `fcopilot parse`."
        )
    by_doc = {d.meta.doc_id: chunk_document(d, settings.chunking) for d in docs}
    estimate = 0.0
    if settings.chunking.context == "llm":
        llm = llm or LLM(settings.llm, settings.resolve(settings.llm.cache_path))
        estimate = estimate_cost(docs, by_doc, llm)
        if dry_run:
            return chunks_path(settings), estimate
        contextualize(docs, by_doc, llm)
    path = chunks_path(settings)
    save_chunks([c for d in docs for c in by_doc[d.meta.doc_id]], path)
    return path, estimate
