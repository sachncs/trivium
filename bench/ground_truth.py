"""Exact dense retrieval via IndexFlatIP. Used for ANN ground truth + ceiling measurement."""
from __future__ import annotations

import numpy as np
import faiss


def build_flat(vectors: np.ndarray) -> faiss.Index:
    d = vectors.shape[1]
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index


def exact_topk(vectors: np.ndarray, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (k-NN ids, scores). Same signature as the approximate index."""
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32)
    index = build_flat(vectors)
    scores, ids = index.search(queries, k)
    return ids, scores
