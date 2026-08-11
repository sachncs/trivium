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
        self.slug_value = slug
        self.model_id_value = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None

    @property
    def slug(self) -> str:
        return self.slug_value

    @property
    def model_id(self) -> str:
        return self.model_id_value

    def rerank(
        self,
        query: str,
        candidates: Sequence[Document],
        top_k: int,
    ) -> SearchResult:
        if not candidates:
            return SearchResult.empty()
        self.ensure_model()
        pairs = [(query, c.body) for c in candidates]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
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

    def ensure_model(self) -> None:
        """Lazy-load the CrossEncoder model."""
        if self.model is not None:
            return
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(self.model_id_value, max_length=self.max_length)
