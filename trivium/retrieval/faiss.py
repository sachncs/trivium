"""FAISS-backed retriever.

Two concrete index types share this module:
- Faiss.Flat   (IndexFlatIP, the exact dense ceiling)
- Faiss.Ivpq   (OPQ-IVFPQ-RFlat, the approximate baseline)

Replaces two legacy files:
- search/vector_index.py:build_index (OPQ-IVFPQ-RFlat)
- search/vector_index.py:build_flat (IndexFlatIP)
- bench/ground_truth.py:build_flat (duplicated IndexFlatIP)

The two divergent code paths in bench/runner.py:run_vector
are now one method body.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from trivium.config.schema import VectorConfig
from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.retrieval.base import Retriever


def ivfpq_factory_str(nlist: int, m: int, nbits: int, use_opq: bool = True, use_rflat: bool = True) -> str:
    opq = "OPQ" if use_opq else ""
    rflat = ",RFlat" if use_rflat else ""
    return f"{opq}{m},IVF{nlist},PQ{m}x{nbits}fs{rflat}"


class Faiss:
    """Faiss-backed retrievers. Use Faiss.Flat or Faiss.Ivpq."""

    class Flat(Retriever):
        """IndexFlatIP — exact cosine over l2-normalised vectors."""

        @property
        def slug(self) -> str:
            return "faiss_flat"

        def __init__(self) -> None:
            self._index = None
            self._doc_ids: list[str] = []

        def add_documents(self, documents: Sequence[Document], vectors: np.ndarray | None = None) -> None:
            import faiss

            if vectors is None:
                raise ValueError("Faiss.Flat requires pre-computed vectors")
            self._doc_ids = [d.doc_id for d in documents]
            d = int(vectors.shape[1])
            arr = np.asarray(vectors, dtype=np.float32)
            self._index = faiss.IndexFlatIP(d)
            self._index.add(arr)

        def search(self, query_vectors: np.ndarray, k: int) -> list[SearchResult]:
            if self._index is None:
                raise RuntimeError("Faiss.Flat.search called before add_documents")
            arr = np.asarray(query_vectors, dtype=np.float32)
            scores, ids = self._index.search(arr, k)
            results: list[SearchResult] = []
            for row in range(ids.shape[0]):
                pairs = [
                    (self._doc_ids[int(ids[row, j])], float(scores[row, j]))
                    for j in range(ids.shape[1])
                    if int(ids[row, j]) != -1
                ]
                results.append(SearchResult.from_pairs(pairs))
            return results

        def set_search_params(self, **params) -> None:
            return

        def size_bytes(self) -> int:
            try:
                return int(self._index.ntotal * self._index.d * 4)
            except Exception:
                return 0

    class Ivpq(Retriever):
        """OPQ-IVFPQ-RFlat — approximate analogue of DiskBBQ."""

        @property
        def slug(self) -> str:
            return "faiss_ivpq_rflat"

        def __init__(self, config: VectorConfig, scale: int, nlist_override: int | None = None) -> None:
            self.config = config
            self.scale = scale
            self.nlist = nlist_override if nlist_override is not None else config.nlist_for_scale(scale)
            self._index = None
            self._ivf_handle: Any = None
            self._doc_ids: list[str] = []
            self._train_size_strategy: Literal["sqrt_n", "50_x_nlist", "fixed_N"] = config.train_size_strategy
            self._train_size_fixed: int = config.train_size_fixed
            self._use_opq: bool = config.use_opq
            self._use_rflat: bool = config.use_rflat
            self._nbits: int = config.nbits
            self._m: int = config.m
            self._nprobe: int = config.nprobe_sweep[0]

        def add_documents(self, documents: Sequence[Document], vectors: np.ndarray | None = None) -> None:
            import faiss

            if vectors is None:
                raise ValueError("Faiss.Ivpq requires pre-computed vectors")
            self._doc_ids = [d.doc_id for d in documents]
            arr = np.asarray(vectors, dtype=np.float32)
            d = int(arr.shape[1])
            m = min(self._m, d)

            rng_state = np.random.get_state()
            np.random.seed(42)
            train_n = self._train_size(arr)
            train_idx = np.random.choice(len(arr), size=min(len(arr), train_n), replace=False)
            np.random.set_state(rng_state)

            factory = ivfpq_factory_str(
                nlist=self.nlist,
                m=m,
                nbits=self._nbits,
                use_opq=self._use_opq,
                use_rflat=self._use_rflat,
            )
            index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
            index.train(arr[train_idx])
            index.add(arr)
            self._index = index
            self._ivf_handle = self.locate_ivf(index)
            self._ivf_handle.nprobe = self._nprobe

        def search(self, query_vectors: np.ndarray, k: int) -> list[SearchResult]:
            if self._index is None:
                raise RuntimeError("Faiss.Ivpq.search called before add_documents")
            arr = np.asarray(query_vectors, dtype=np.float32)
            scores, ids = self._index.search(arr, k)
            results: list[SearchResult] = []
            for row in range(ids.shape[0]):
                pairs = [
                    (self._doc_ids[int(ids[row, j])], float(scores[row, j]))
                    for j in range(ids.shape[1])
                    if int(ids[row, j]) != -1
                ]
                results.append(SearchResult.from_pairs(pairs))
            return results

        def set_search_params(self, **params) -> None:
            nprobe = params.get("nprobe")
            if nprobe is not None and self._ivf_handle is not None:
                self._nprobe = int(nprobe)
                self._ivf_handle.nprobe = int(nprobe)
            k_factor = params.get("k_factor")
            if k_factor is not None and hasattr(self._index, "k_factor"):
                self._index.k_factor = int(k_factor)

        def size_bytes(self) -> int:
            try:
                return int(self._index.ntotal * self._index.d * 4)
            except Exception:
                return 0

        def _train_size(self, vectors: np.ndarray) -> int:
            n = len(vectors)
            if self._train_size_strategy == "sqrt_n":
                return int(np.sqrt(n))
            if self._train_size_strategy == "50_x_nlist":
                return max(50 * self.nlist, self._train_size_fixed)
            return self._train_size_fixed

        @staticmethod
        def locate_ivf(index) -> Any:
            """Public method (no leading underscore) — recursively walks
            a composite FAISS index to find the IVF coarse quantizer.

            Replaces the closure-nested find_ivf() helper in
            search/vector_index.py. Promoted to a public static
            method so DiskBBQ can reuse it for index compatibility.
            """
            if hasattr(index, "nprobe"):
                return index
            for attr in ("base_index", "index"):
                if hasattr(index, attr):
                    sub = getattr(index, attr)
                    if sub is not None:
                        found = Faiss.Ivpq.locate_ivf(sub)
                        if found is not None:
                            return found
            return None
