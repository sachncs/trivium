"""FAISS OPQ-IVF-PQ FastScan + exact refinement. RAM-resident algorithmic analog of DiskBBQ."""
from __future__ import annotations

import math

import faiss
import numpy as np


def ivfpq_factory(nlist: int, m: int, nbits: int, use_opq: bool = True, use_rflat: bool = True) -> str:
    opq = "OPQ" if use_opq else ""
    rflat = ",RFlat" if use_rflat else ""
    return f"{opq}{m},IVF{nlist},PQ{m}x{nbits}fs{rflat}"


def build_index(vectors: np.ndarray, nlist: int, m: int, nbits: int = 4, use_opq: bool = True, use_rflat: bool = True, seed: int = 42) -> faiss.Index:
    """Build OPQ-IVF-PQ FastScan + RFlat. vectors must be l2-normalized; metric = IP."""
    d = vectors.shape[1]
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    if m > d:
        m = d
    train_size = min(len(vectors), max(50 * nlist, 30_000))
    factory = ivfpq_factory(nlist, m, nbits, use_opq, use_rflat)
    index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
    rng_state = np.random.get_state()
    np.random.seed(seed)
    train_idx = np.random.choice(len(vectors), size=train_size, replace=False)
    np.random.set_state(rng_state)
    index.train(vectors[train_idx])
    index.add(vectors)
    return index


def set_search_params(index: faiss.Index, nprobe: int, k_factor: int = 4) -> None:
    """Walk the composite index to find the IVF coarse quantizer and set nprobe.

    IndexRefineFlat exposes `base_index` (the OPQ-IVFPQ tree) and `refine_index` (the flat).
    We recursively descend into composite indexes until we find one with `nprobe`.
    """
    if hasattr(index, "k_factor"):
        index.k_factor = k_factor

    def find_ivf(idx):
        if hasattr(idx, "nprobe"):
            return idx
        for attr in ("base_index", "index"):
            if hasattr(idx, attr):
                sub = getattr(idx, attr)
                if sub is not None:
                    found = find_ivf(sub)
                    if found is not None:
                        return found
        return None

    ivf = find_ivf(index)
    if ivf is not None:
        ivf.nprobe = nprobe


def search(index: faiss.Index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (k-NN ids, scores). IndexFlatIP-compatible signature."""
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32)
    scores, ids = index.search(queries, k)
    return ids, scores


def build_flat(vectors: np.ndarray) -> faiss.Index:
    """Exact baseline over same vectors. Used for ANN ground truth + recall measurements."""
    d = vectors.shape[1]
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index
