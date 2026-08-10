"""Retriever ABC: the contract every search engine implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from trivium.domain.document import Document
from trivium.domain.result import SearchResult


class Retriever(ABC):
    """Returns ranked SearchResults for query vectors.

    Implementations:
        - trivium.retrieval.bm25.Bm25
        - trivium.retrieval.faiss.Faiss.Flat
        - trivium.retrieval.faiss.Faiss.Ivpq
        - trivium.retrieval.diskbbq.Bbq
    """

    @property
    @abstractmethod
    def slug(self) -> str:
        """Short identifier used in CSV rows and registries."""

    @abstractmethod
    def add_documents(self, documents: Sequence[Document], vectors: np.ndarray | None = None) -> None:
        """Index documents.

        Args:
            documents: The corpus slice this retriever will search.
            vectors: Optional pre-computed embeddings (float32). For
                BM25 this is ignored; for vector retrievers it is
                mandatory. Shape must be (N, dim).
        """

    @abstractmethod
    def search(self, query_vectors: np.ndarray, k: int) -> SearchResult:
        """Search N queries. Returns a list of SearchResult, one per query.

        Args:
            query_vectors: (N, dim) float32.
            k: Top-k to return per query.
        """

    @abstractmethod
    def set_search_params(self, **params) -> None:
        """Set search-time parameters (e.g. nprobe for IVF indexes).

        Implementations validate the keys they recognise and ignore
        the rest.
        """

    @abstractmethod
    def size_bytes(self) -> int:
        """Estimated in-memory (or on-disk) footprint in bytes."""
