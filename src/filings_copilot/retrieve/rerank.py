"""Cross-encoder reranking: reads query and passage together, so it catches relevance that
bi-encoder similarity misses (e.g. the passage that answers "why", not just mentions)."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder  # optional heavy dependency

        self.name = model_name
        self._model = CrossEncoder(model_name, max_length=max_length)
        self._batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, p) for p in passages], batch_size=self._batch_size, show_progress_bar=False
        )
        return [float(s) for s in scores]
