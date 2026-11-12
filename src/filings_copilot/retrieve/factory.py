"""Wire up a Retriever for a given configuration, building the dense index if needed."""

from __future__ import annotations

from functools import lru_cache

from rich.console import Console

from filings_copilot.config import Settings
from filings_copilot.index.dense import DenseIndex, SentenceTransformerEmbedder
from filings_copilot.ingest.store import chunks_path, load_chunks
from filings_copilot.llm import LLM
from filings_copilot.retrieve.pipeline import Retriever, index_path
from filings_copilot.retrieve.rerank import CrossEncoderReranker

console = Console()


@lru_cache(maxsize=4)
def _embedder(
    name: str, instruction: str, batch_size: int, device: str = "auto"
) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(name, instruction, batch_size, device)


@lru_cache(maxsize=2)
def _reranker(name: str, device: str = "auto") -> CrossEncoderReranker:
    return CrossEncoderReranker(name, device=device)


def build_dense_index(settings: Settings, force: bool = False) -> DenseIndex:
    path = index_path(settings)
    if path.exists() and not force:
        return DenseIndex.load(path)
    chunks = load_chunks(chunks_path(settings))
    e = settings.embedding
    console.print(f"Embedding {len(chunks)} chunks with {e.model} -> {path}")
    index = DenseIndex.build(
        _embedder(e.model, e.query_instruction, e.batch_size, e.device),
        [c.index_text for c in chunks],
    )
    index.save(path)
    return index


def build_retriever(settings: Settings, llm: LLM | None = None) -> Retriever:
    cpath = chunks_path(settings)
    if not cpath.exists():
        raise FileNotFoundError(
            f"Missing chunk set {cpath}. Run `fcopilot chunk` for this variant."
        )
    chunks = load_chunks(cpath)
    e, r = settings.embedding, settings.retrieval
    dense = embedder = None
    if r.mode in ("dense", "hybrid"):
        dense = build_dense_index(settings)
        embedder = _embedder(e.model, e.query_instruction, e.batch_size, e.device)
    reranker = _reranker(r.reranker_model, e.device) if r.rerank else None
    if llm is None and r.decompose:
        llm = LLM(settings.llm, settings.resolve(settings.llm.cache_path))
    return Retriever(settings, chunks, dense, embedder, reranker, llm)
