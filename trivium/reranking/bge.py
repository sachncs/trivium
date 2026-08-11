"""BGE reranker (bge-reranker-large)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.reranking.base import Reranker


class Bge(Reranker):
    """bge-reranker-large / -base.

    Default: BAAI/bge-reranker-large (2.3 GB). Input format is
    'query\\npassage' (sentence-transformers CrossEncoder compatible).
    """

    def __init__(
        self,
        slug: str = "bge-reranker-large",
        model_id: str = "BAAI/bge-reranker-large",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        self._slug = slug
        self._model_id = model_id
        self._max_length = max_length
        self._batch_size = batch_size
        self._model = None

    @property
    def slug(self) -> str:
        return self._slug

    @property
    def model_id(self) -> str:
        return self._model_id

    def rerank(
        self,
        query: str,
        candidates: Sequence[Document],
        top_k: int,
    ) -> SearchResult:
        if not candidates:
            return SearchResult.empty()
        self._ensure_model()
        pairs = [(query, c.body) for c in candidates]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            activation_fct=np.sigmoid,
        )
        scores = np.asarray(scores, dtype=np.float64)
        order = np.argsort(-scores)[:top_k]
        return SearchResult(
            hits=[
                (c, float(s))
                for c, s in zip(
                    [candidates[i] for i in order],
                    [scores[i] for i in order],
                    strict=False,
                )
            ]
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self._model_id, max_length=self._max_length)
