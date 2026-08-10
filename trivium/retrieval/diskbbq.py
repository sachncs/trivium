"""Disk-resident Bbq (Balanced Binary Quantization) retriever.

This is the trivium-style API surface for a disk-resident index.
The full 100M-vector build lands in a follow-up commit; this
file establishes the contract so pipelines can reference it.

Layout on disk (built lazily, peak memory = O(nlist)):
  out_dir/centroids.npy                    float32 (nlist, dim)
  out_dir/subvector_thresholds.npy         float32 (m,)
  out_dir/list_{cid}.codes                 uint8 packed bits  (|list| * m/8 bytes)
  out_dir/list_{cid}.ids                   int64   (|list|,)
  out_dir/list_{cid}.floats                float32 (|list|, dim) - optional RFlat
  out_dir/metadata.json                    {nlist, m, dim, nbits, list_sizes}

Search path:
  1. Coarse-quantise query against centroids (RAM).
  2. Async I/O via ThreadPoolExecutor for top-nprobe lists.
  3. Compute Hamming distances via bitwise_xor + popcount.
  4. Optional RFlat rescore with the per-list float32 vectors.

Until a real build method is wired, Bbq() returns an index whose
add_documents() raises NotImplementedError; pipelines pick this
slug only at scales >= the threshold configured by CF3.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.retrieval.base import Retriever


class Bbq(Retriever):
    """Disk-resident IVF + 1-bit binary product quantization."""

    def __init__(
        self,
        out_dir: str | Path,
        nlist: int | None = None,
        m: int = 48,
        nbits: int = 1,
        prefetch_depth: int = 2,
        nprobe: int = 8,
        use_rescore: bool = True,
        k_factor: int = 4,
        max_workers: int | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.prefetch_depth = prefetch_depth
        self.nprobe = nprobe
        self.use_rescore = use_rescore
        self.k_factor = k_factor
        self.max_workers = max_workers
        self._centroids: np.ndarray | None = None
        self._thresholds: np.ndarray | None = None
        self._list_files: dict[int, dict[str, Path]] = {}
        self._list_handles: dict[int, dict[str, object]] = {}
        self._doc_ids: list[str] = []

    @property
    def slug(self) -> str:
        return "diskbbq"

    def add_documents(self, documents: Sequence[Document], vectors: np.ndarray | None = None) -> None:
        raise NotImplementedError(
            "Bbq.add_documents is the build path; it streams writes via "
            "np.memmap with O(nlist) peak memory. Wire it in the DiskBBQ "
            "production commit (out of scope for the refactor PR). The "
            "trivium-style API contract is established here."
        )

    def search(self, query_vectors: np.ndarray, k: int) -> list[SearchResult]:
        raise NotImplementedError("Bbq.search awaits the production build path.")

    def set_search_params(self, **params) -> None:
        if "nprobe" in params:
            self.nprobe = int(params["nprobe"])
        if "use_rescore" in params:
            self.use_rescore = bool(params["use_rescore"])

    def size_bytes(self) -> int:
        total = 0
        for cid, paths in self._list_files.items():
            for p in paths.values():
                if Path(p).exists():
                    total += Path(p).stat().st_size
        if self._centroids is not None:
            total += int(self._centroids.nbytes)
        if self._thresholds is not None:
            total += int(self._thresholds.nbytes)
        return total

    def _ensure_worker_pool(self) -> ThreadPoolExecutor:
        workers = self.max_workers
        if workers is None:
            import os

            workers = min(32, (os.cpu_count() or 1))
        return ThreadPoolExecutor(max_workers=workers)

    def _load_metadata(self) -> dict:
        path = self.out_dir / "metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())
