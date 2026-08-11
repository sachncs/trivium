"""ExactSearch: dense exact ground-truth over IndexFlatIP."""

from __future__ import annotations

import numpy as np


def _as_float32(arr: np.ndarray) -> np.ndarray:
    return arr if arr.dtype == np.float32 else arr.astype(np.float32)


class ExactSearch:
    """Lazy exact-search wrapper around faiss.IndexFlatIP.

    Single source of truth for ground-truth construction
    (replaces the duplicated bench/ground_truth.py:build_flat
    and search/vector_index.py:build_flat in the legacy code).

    For large corpora the build is lazy — IndexFlatIP is built
    on first search() call and cached. Vectors are stored as
    float32 (faiss requirement).
    """

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = _as_float32(vectors)
        self.dimension = int(self.vectors.shape[1])
        self.index = None  # type: ignore[assignment]

    def build(self) -> ExactSearch:
        import faiss

        index = faiss.IndexFlatIP(self.dimension)
        index.add(self.vectors)
        self.index = index
        return self

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (k-NN ids, scores). Single-source-of-truth signature."""
        if self.index is None:
            self.build()
        queries = _as_float32(queries)
        scores, ids = self.index.search(queries, k)
        return ids, scores

    def size_bytes(self) -> int:
        return int(self.vectors.nbytes)
