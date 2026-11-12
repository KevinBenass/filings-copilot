import pytest
from conftest import HashEmbedder, OverlapReranker, meta, mini_10k_html

from filings_copilot.config import ChunkingConfig, load_settings
from filings_copilot.index.dense import DenseIndex
from filings_copilot.ingest.chunking import chunk_document
from filings_copilot.ingest.parse import parse_filing
from filings_copilot.retrieve.pipeline import Retriever
from filings_copilot.retrieve.query import extract_tickers, extract_years


@pytest.fixture
def corpus_chunks():
    docs = [
        parse_filing(mini_10k_html("Acme Compute", "60,922"), meta("ACME", 2025, "Acme Compute")),
        parse_filing(mini_10k_html("Acme Compute", "41,000"), meta("ACME", 2024, "Acme Compute")),
        parse_filing(
            mini_10k_html("Borealis Chips", "12,345"), meta("BORE", 2025, "Borealis Chips")
        ),
    ]
    cfg = ChunkingConfig(strategy="structure", max_tokens=120, context="metadata")
    return [c for d in docs for c in chunk_document(d, cfg)]


def make_retriever(chunks, **overrides):
    s = load_settings(overrides={"retrieval.top_k": 3, **overrides})
    emb = HashEmbedder()
    return Retriever(
        s, chunks, DenseIndex.build(emb, [c.index_text for c in chunks]), emb, OverlapReranker()
    )


def test_extract_entities():
    known = {"AAPL", "MSFT", "JPM"}
    assert extract_tickers("Compare Apple and Microsoft revenue", known) == ["AAPL", "MSFT"]
    assert extract_tickers("Share repurchases at Apple", known) == ["AAPL"]  # "chase" not matched
    assert extract_years("What changed between FY23 and fiscal year 2025?") == [2023, 2025]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_modes_return_top_k(corpus_chunks, mode):
    hits = make_retriever(corpus_chunks, **{"retrieval.mode": mode}).retrieve(
        "Data Center revenue by segment"
    )
    assert len(hits) == 3
    assert len({h.chunk.chunk_id for h in hits}) == 3


def test_filters_restrict_company_and_year(corpus_chunks):
    r = make_retriever(corpus_chunks, **{"retrieval.use_filters": True})
    hits = r.retrieve("Borealis Chips data center revenue fiscal 2025")
    assert {h.chunk.ticker for h in hits} == {"BORE"}


def test_soft_year_filter_keeps_next_year_filing(corpus_chunks):
    r = make_retriever(corpus_chunks, **{"retrieval.use_filters": True})
    mask = r.build_mask(["ACME"], [2024])
    years = {c.fiscal_year for c, keep in zip(corpus_chunks, mask, strict=True) if keep}
    assert years == {2024, 2025}


def test_filter_relaxes_when_too_narrow(corpus_chunks):
    r = make_retriever(corpus_chunks, **{"retrieval.use_filters": True, "retrieval.top_k": 50})
    assert r.build_mask(["BORE"], [2025]) is None


def test_rerank_ranks_recorded(corpus_chunks):
    hits = make_retriever(corpus_chunks).retrieve("export controls China revenue risk")
    assert all("rerank" in h.ranks and "fused" in h.ranks for h in hits)
    assert "Export controls" in hits[0].chunk.text


def test_pick_device_avoids_mps_on_intel_macs(monkeypatch):
    import sys
    import types

    from filings_copilot.index import device

    fake = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(device.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(device.platform, "machine", lambda: "x86_64")
    assert (
        device.pick_device() == "cpu"
    )  # Intel Mac: MPS exists but is slower and runs out of memory
    monkeypatch.setattr(device.platform, "machine", lambda: "arm64")
    assert device.pick_device() == "mps"  # Apple Silicon
    assert device.pick_device("cpu") == "cpu"  # explicit setting wins
