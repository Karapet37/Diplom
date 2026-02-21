"""Embedding layer with deterministic offline vectors and SQL persistence."""

from __future__ import annotations

import hashlib
from typing import Iterable

from src.living_system.knowledge_sql import KnowledgeSQLStore


class HashEmbeddingService:
    """Deterministic hash embeddings for local/offline semantics."""

    def __init__(self, store: KnowledgeSQLStore, *, model_name: str = "hash-embed-v1", dimensions: int = 16):
        self.store = store
        self.model_name = model_name
        self.dimensions = max(4, int(dimensions))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = [token.strip().lower() for token in text.replace("\n", " ").split(" ")]
        return [token for token in tokens if token]

    def _vector_for_tokens(self, tokens: Iterable[str]) -> list[float]:
        vec = [0.0 for _ in range(self.dimensions)]
        count = 0
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for idx in range(self.dimensions):
                byte = digest[idx % len(digest)]
                vec[idx] += (float(byte) / 255.0) - 0.5
            count += 1
        if count <= 0:
            return vec
        return [round(value / float(count), 6) for value in vec]

    def embed_text(self, text: str) -> list[float]:
        return self._vector_for_tokens(self._tokenize(text))

    def embed_and_store(self, *, owner_type: str, owner_id: str, text: str, version: int = 1) -> list[float]:
        vector = self.embed_text(text)
        self.store.store_embedding(
            owner_type=owner_type,
            owner_id=owner_id,
            vector=vector,
            model_name=self.model_name,
            version=version,
        )
        return vector
