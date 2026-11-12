"""Dense retrieval with an exact inner-product search over normalized embeddings.

A corpus of ~25k chunks fits comfortably in memory, so exact search (one matrix-vector
product, a few ms) beats an approximate index on both recall and simplicity. The Embedder
protocol keeps the model swappable and lets tests run without downloading weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        query_instruction: str = "",
        batch_size: int = 32,
        device: str = "auto",
    ) -> None:
        from sentence_transformers import SentenceTransformer  # optional heavy dependency

        self.name = model_name
        from filings_copilot.index.device import pick_device

        self._model = SentenceTransformer(model_name, device=pick_device(device))
        self._instruction = query_instruction
        self._batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            self._instruction + text, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)


class DenseIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors.astype(np.float32)
        self.size = len(vectors)

    @classmethod
    def build(cls, embedder: Embedder, texts: list[str]) -> DenseIndex:
        return cls(embedder.embed_documents(texts))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        return cls(np.load(path))

    def search(
        self, query_vec: np.ndarray, k: int, mask: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        scores = self.vectors @ query_vec
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]
