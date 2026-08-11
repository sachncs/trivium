"""Cross-encoder reranker (sentence-transformers CrossEncoder backend)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.reranking.base import Reranker


class Encoder(Reranker):
    """Cross-encoder reranker over a sentence-transformers CrossEncoder model.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (90 MB, +180 ms p95).
    Replaces trivium.reranking.encoder.Encoder formerly known as
    `CrossEncoderReranker` (the legacy triple-encoder-redundant name).
    """

    def __init__(
        self,
        slug: str = "encoder",
        model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256,
        batch_size: int = 32,
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
