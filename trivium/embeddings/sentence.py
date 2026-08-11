"""Sentence-transformers-based embedder.

One concrete Embedder implementation handles every model loadable
via the sentence-transformers library:
- all-MiniLM-L6-v2 (90 MB, 384d) - baseline
- BAAI/bge-large-en-v1.5 (1.34 GB, 1024d)
- BAAI/bge-base-en-v1.5 (0.4 GB, 768d)
- BAAI/bge-small-en-v1.5 (0.13 GB, 384d)
- all-mpnet-base-v2 (0.4 GB, 768d)
- intfloat/e5-large-v2 (1.3 GB, 1024d) — uses BGE-style prefixes here

The model is loaded lazily on first call to encode_*. The heavy
import is deferred to keep CLI startup fast.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.embeddings.base import Embedder


class Sentence(Embedder):
    """One embedder over any sentence-transformers model.

    Args:
        slug: Short identifier (e.g. 'bge-large').
        model_id: HuggingFace model id (e.g. 'BAAI/bge-large-en-v1.5').
        dimension: Embedding dimension (must match the model).
        prompt_prefix_doc: Optional passage-side prefix.
        prompt_prefix_query: Optional query-side prefix.
        batch_size: Encoding batch size.
        max_seq_length: Tokeniser max length.
        normalize: L2 normalise vectors.
        device: 'cpu' / 'cuda' / 'mps'.
    """

    def __init__(
        self,
        slug: str,
        model_id: str,
        dimension: int,
        prompt_prefix_doc: str = "",
        prompt_prefix_query: str = "",
        batch_size: int = 128,
        max_seq_length: int = 256,
        normalize: bool = True,
        device: str = "cpu",
    ) -> None:
        self._slug = slug
        self._model_id = model_id
        self._dimension = dimension
        self._prompt_prefix_doc = prompt_prefix_doc
        self._prompt_prefix_query = prompt_prefix_query
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._normalize = normalize
        self._device = device
        self._model = None

    @property
    def slug(self) -> str:
        return self._slug

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    def warmup(self, sample_texts: Sequence[str] = ("warmup",)) -> None:
        if self._model is None:
            self._ensure_model()
        _ = self._model.encode(
            list(sample_texts),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        prefixed = [self._prompt_prefix_doc + t for t in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        prefixed = [self._prompt_prefix_query + t for t in texts]
        vectors = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_id, device=self._device)
        self._model.max_seq_length = self._max_seq_length
