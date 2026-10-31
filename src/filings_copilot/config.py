"""Typed configuration loaded from YAML, with dotted-key overrides for ablations."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


class CorpusConfig(BaseModel):
    tickers: list[str]
    form: str = "10-K"
    fiscal_years: list[int]
    user_agent: str = "filings-copilot research"
    raw_dir: Path = Path("data/raw")
    parsed_dir: Path = Path("data/parsed")


class ChunkingConfig(BaseModel):
    strategy: Literal["structure", "fixed"] = "structure"
    max_tokens: int = 400
    overlap_tokens: int = 60
    min_tokens: int = 40
    context: Literal["none", "metadata", "llm"] = "metadata"
    chunks_dir: Path = Path("data/chunks")

    @property
    def variant(self) -> str:
        """Identifier of the chunk set; indexes are built per variant."""
        return f"{self.strategy}-{self.max_tokens}-{self.context}"


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-base-en-v1.5"
    query_instruction: str = ""
    batch_size: int = 32
    index_dir: Path = Path("data/index")


class RetrievalConfig(BaseModel):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    candidates_per_retriever: int = 50
    rrf_k: int = 60
    use_filters: bool = True
    decompose: bool = False
    rerank: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 40
    top_k: int = 8


class Price(BaseModel):
    input: float
    output: float


class LLMConfig(BaseModel):
    fast_model: str = "claude-haiku-5-5"
    answer_model: str = "claude-sonnet-5-5"
    judge_model: str = "claude-haiku-5-5"
    max_tokens: int = 1024
    temperature: float = 0.0
    cache_path: Path = Path("data/llm_cache.sqlite")
    prices: dict[str, Price] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    gold_path: Path = Path("data/eval/gold.jsonl")
    n_single_hop: int = 180
    n_table: int = 60
    n_multi_hop: int = 40
    n_unanswerable: int = 30
    seed: int = 7
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    results_dir: Path = Path("results")


class Settings(BaseModel):
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    eval: EvalConfig

    @property
    def sec_user_agent(self) -> str:
        return os.environ.get("SEC_USER_AGENT", self.corpus.user_agent)

    def resolve(self, path: Path) -> Path:
        """Resolve a config path relative to the repository root."""
        return path if path.is_absolute() else ROOT / path


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``raw`` with dotted-key overrides applied (e.g. ``retrieval.mode``)."""
    out = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"Unknown config section in override: {dotted}")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"Unknown config key in override: {dotted}")
        node[leaf] = value
    return out


def load_settings(
    path: Path | str | None = None, overrides: dict[str, Any] | None = None
) -> Settings:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    if overrides:
        raw = apply_overrides(raw, overrides)
    return Settings.model_validate(raw)


def load_ablations(path: Path | str | None = None) -> list[dict[str, Any]]:
    return yaml.safe_load(Path(path or ROOT / "configs" / "ablations.yaml").read_text())
