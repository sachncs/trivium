"""Reranker ABC: the contract every cross-attention reranker implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from trivium.domain.document import Document
from trivium.domain.result import SearchResult


class Reranker(ABC):
    """Reranks a candidate set against a query.

    Implementations:
        - trivium.reranking.encoder.Encoder     (cross-encoder MiniLM-L-6)
        - trivium.reranking.bge.Bge             (bge-reranker-large)
        - trivium.reranking.monot5.Monot5       (MonoT5-3B seq2seq)
    """

    @property
    @abstractmethod
    def slug(self) -> str:
        """Short identifier used in CSV rows and registries."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[Document],
        top_k: int,
    ) -> SearchResult:
        """Score (query, candidate) pairs and return top_k by score.

        Args:
            query: The query text.
            candidates: Documents to score. Their `body` property
                provides the text fragment for the cross-attention
                pair.
            top_k: Number of candidates to keep in the output.
        """
