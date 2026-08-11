"""Embedder ABC: the contract every text vectorizer implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    """Encodes text into float32 vectors.

    Implementations:
        - trivium.embeddings.sentence.Sentence  (sentence-transformers backend)
        - trivium.embeddings.e5.E5              (E5-Mistral instruct template)
    """

    @property
    @abstractmethod
    def slug(self) -> str:
        """Short identifier used in CSV rows and registries."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension. Constant per model."""

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode corpus/document text. Returns (N, dim) float32."""

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode query text. Returns (N, dim) float32.

        Implementations may use a different prompt template than
        encode_documents (E5-Mistral does; sentence-transformers
        mostly doesn't).
        """

    @abstractmethod
    def warmup(self, sample_texts: Sequence[str]) -> None:
        """Load the model lazily on first use and warm any caches.
        Called once at pipeline start so first-query cost is bounded.
        """
