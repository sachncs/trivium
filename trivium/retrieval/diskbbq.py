"""Disk-resident Bbq (Balanced Binary Quantization) retriever.

Implements the disk-resident IVF + 1-bit binary product quantisation
index:

  out_dir/
    centroids.npy              float32 (nlist, dim)         RAM at search
    subvector_thresholds.npy   float32 (m,)                 RAM at search
    list_{cid}.codes.npy       uint8 packed bits  (|list| * m/8 bytes)
    list_{cid}.ids.npy         int64   (|list|,)
    list_{cid}.floats.npy      float32 (|list|, dim)        optional RFlat
    metadata.json              {nlist, m, dim, nbits, list_sizes}

Build path: keep in-RAM arrays after build (fast path for ≤100K),
but flush every file to disk so the contract is honoured.

Search path: three modes.
  - 'ram'  : use the in-RAM arrays (fastest at small scale).
  - 'disk' : lazy-load only the top-nprobe lists via ThreadPoolExecutor.
  - 'auto' : 'disk' if the on-disk index footprint exceeds
             `auto_disk_threshold_bytes`; else 'ram'.

Hamming distance: `np.bitwise_xor` + `np.unpackbits(...).sum(axis=...)`
on packed uint8 codes (m/8 bytes per vector).

Optional RFlat: re-rank top `k_factor * k` candidates with the exact
inner product against `.floats`.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.retrieval.base import Retriever

SearchMode = str  # 'ram' | 'disk' | 'auto'


class Bbq(Retriever):
    """Disk-resident IVF + 1-bit binary product quantisation."""

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
        auto_disk_threshold_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if nbits != 1:
            raise ValueError(f"only nbits=1 is supported; got {nbits}")
        self.out_dir = Path(out_dir)
        self.nlist = nlist
        self.m = int(m)
        self.nbits = int(nbits)
        self.prefetch_depth = int(prefetch_depth)
        self.nprobe = int(nprobe)
        self.use_rescore = bool(use_rescore)
        self.k_factor = int(k_factor)
        self.max_workers = max_workers
        self.auto_disk_threshold_bytes = int(auto_disk_threshold_bytes)

        self.centroids: np.ndarray | None = None
        self.thresholds: np.ndarray | None = None
        self.dim: int = 0
        self.doc_ids: list[str] = []
        self.list_files: dict[int, dict[str, Path]] = {}
        self.list_handles: dict[int, dict[str, object]] = {}
        self._memmaps: list[object] = []

        self._ram_codes: dict[int, np.ndarray] | None = None
        self._ram_ids: dict[int, np.ndarray] | None = None
        self._ram_floats: dict[int, np.ndarray] | None = None

    @property
    def slug(self) -> str:
        return "diskbbq"

    def add_documents(
        self, documents: Sequence[Document], vectors: np.ndarray | None = None
    ) -> None:
        """Build the disk-resident index.

        Builds coarse centroids, packs 1-bit codes per list, and writes
        every file to disk. In-RAM copies are retained so 'ram' search
        mode is available without disk I/O.
        """
        if vectors is None:
            raise ValueError("Bbq requires pre-computed vectors")
        if not documents:
            raise RuntimeError("Bbq.add_documents requires at least one document")

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"vectors must be 2-D; got shape {arr.shape}")
        n, dim = arr.shape
        if dim % self.m != 0:
            raise ValueError(f"dim={dim} not divisible by m={self.m}")
        self.dim = int(dim)
        self.doc_ids = [d.doc_id for d in documents]

        nlist = self.nlist or self._default_nlist(n)
        self.nlist = int(nlist)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for old in self.out_dir.glob("list_*.codes.npy"):
            old.unlink()
        for old in self.out_dir.glob("list_*.ids.npy"):
            old.unlink()
        for old in self.out_dir.glob("list_*.floats.npy"):
            old.unlink()

        self.centroids = self._train_coarse(arr)
        assignments = self._assign_lists(arr, self.centroids)

        codes_ram: dict[int, np.ndarray] = {}
        ids_ram: dict[int, np.ndarray] = {}
        floats_ram: dict[int, np.ndarray] = {}
        list_sizes: list[int] = []

        for cid in range(self.nlist):
            idx = assignments[cid]
            if idx.size == 0:
                list_sizes.append(0)
                codes_ram[cid] = np.empty((0, self.m // 8), dtype=np.uint8)
                ids_ram[cid] = np.empty((0,), dtype=np.int64)
                floats_ram[cid] = np.empty((0, self.dim), dtype=np.float32)
                continue

            list_vecs = arr[idx]
            codes, thresholds_for_list = self._pack_binary_codes(list_vecs)
            self._write_list(
                cid,
                ids=np.arange(n, dtype=np.int64)[idx],
                codes=codes,
                floats=list_vecs if self.use_rescore else None,
            )
            codes_ram[cid] = codes
            ids_ram[cid] = np.arange(n, dtype=np.int64)[idx]
            floats_ram[cid] = list_vecs if self.use_rescore else np.empty((0, self.dim), dtype=np.float32)
            list_sizes.append(int(idx.size))

        self.thresholds = self._compute_thresholds(arr, assignments)
        np.save(self.out_dir / "centroids.npy", self.centroids.astype(np.float32))
        np.save(self.out_dir / "subvector_thresholds.npy", self.thresholds.astype(np.float32))
        self._write_metadata(list_sizes)

        self._ram_codes = codes_ram
        self._ram_ids = ids_ram
        self._ram_floats = floats_ram
        self.list_files = self._discover_list_files()
        self.list_handles = {}

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        mode: SearchMode = "auto",
    ) -> list[SearchResult]:
        """Search N queries; return a SearchResult per query.

        Args:
            query_vectors: (N, dim) float32.
            k: top-k to return.
            mode: 'ram' | 'disk' | 'auto'.
        """
        if self.centroids is None or self.thresholds is None:
            raise RuntimeError("Bbq.search called before add_documents")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        arr = np.asarray(query_vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        effective_mode = self._resolve_mode(mode)
        if effective_mode == "ram":
            return self._ram_search(arr, k)
        return self._disk_search(arr, k)

    def set_search_params(self, **params) -> None:
        if "nprobe" in params:
            self.nprobe = int(params["nprobe"])
        if "use_rescore" in params:
            self.use_rescore = bool(params["use_rescore"])
        if "k_factor" in params:
            self.k_factor = int(params["k_factor"])

    def size_bytes(self) -> int:
        total = 0
        for paths in self.list_files.values():
            for p in paths.values():
                if Path(p).exists():
                    total += Path(p).stat().st_size
        if self.centroids is not None:
            total += int(self.centroids.nbytes)
        if self.thresholds is not None:
            total += int(self.thresholds.nbytes)
        meta = self.out_dir / "metadata.json"
        if meta.exists():
            total += meta.stat().st_size
        return total

    def ensure_worker_pool(self) -> ThreadPoolExecutor:
        workers = self.max_workers
        if workers is None:
            workers = min(32, (os.cpu_count() or 1))
        return ThreadPoolExecutor(max_workers=workers)

    def load_metadata(self) -> dict:
        path = self.out_dir / "metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def close(self) -> None:
        for mm in self._memmaps:
            try:
                close = getattr(mm, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        self._memmaps = []
        self.list_handles = {}

    def _resolve_mode(self, mode: SearchMode) -> SearchMode:
        if mode == "auto":
            return "disk" if self.size_bytes() >= self.auto_disk_threshold_bytes else "ram"
        if mode in ("ram", "disk"):
            return mode
        raise ValueError(f"unknown search mode {mode!r}; expected 'ram', 'disk', or 'auto'")

    @staticmethod
    def _default_nlist(n: int) -> int:
        """4·sqrt(N) rounded up to the next power of 2 (FAISS wiki recipe)."""
        raw = max(1, int(4 * np.sqrt(n)))
        return 1 << (raw - 1).bit_length()

    def _train_coarse(self, vectors: np.ndarray) -> np.ndarray:
        """Train k-means centroids; seeded for reproducibility.

        Uses the numpy fallback path directly. FAISS Kmeans hangs on
        some platforms with real l2-normalised MiniLM vectors; the
        numpy path is correct and runs in O(n_iter * n * k * dim)
        which is plenty fast for our sizes.
        """
        n = vectors.shape[0]
        n_iter = 25
        rng = np.random.RandomState(42)
        nlist = int(self.nlist)
        train_n = min(n, max(50 * nlist, 30_000))
        init_idx = rng.choice(n, size=min(n, train_n), replace=False)
        init = vectors[init_idx[:nlist]].copy()
        centroids = init
        for _ in range(n_iter):
            diffs = vectors[:, None, :] - centroids[None, :, :]
            dists = (diffs * diffs).sum(axis=2)
            assign = np.argmin(dists, axis=1)
            new = np.zeros_like(centroids)
            counts = np.zeros(nlist, dtype=np.int64)
            np.add.at(new, assign, vectors)
            np.add.at(counts, assign, 1)
            mask = counts > 0
            new[mask] /= counts[mask, None]
            empty = ~mask
            if empty.any():
                rand_idx = rng.choice(n, size=int(empty.sum()))
                new[empty] = vectors[rand_idx]
            shift = np.linalg.norm(new - centroids, axis=1).sum()
            centroids = new
            if shift < 1e-4:
                break
        return centroids.astype(np.float32)

    @staticmethod
    def _assign_lists(vectors: np.ndarray, centroids: np.ndarray) -> list[np.ndarray]:
        """Argmin-distance partition; returns list of index arrays."""
        diffs = vectors[:, None, :] - centroids[None, :, :]
        dists = (diffs * diffs).sum(axis=2)
        assign = np.argmin(dists, axis=1)
        nlist = centroids.shape[0]
        return [np.where(assign == cid)[0] for cid in range(nlist)]

    def _pack_binary_codes(self, list_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split into m subvectors; per-subvector mean → 1-bit codes packed in uint8.

        Returns (codes (|list|, m*chunk/8) uint8, thresholds (m,) float32 for THIS list).
        The per-list thresholds are not used at search; the global
        `thresholds` (mean over all vectors) is.
        """
        if list_vectors.shape[0] == 0:
            return np.empty((0, (self.dim // self.m) * self.m // 8), dtype=np.uint8), np.zeros(self.m, dtype=np.float32)
        n, dim = list_vectors.shape
        chunk = dim // self.m
        reshaped = list_vectors.reshape(n, self.m, chunk)
        thresholds = reshaped.mean(axis=(0, 2)).astype(np.float32)
        bits = (reshaped > thresholds[None, :, None]).astype(np.uint8)
        bits = bits.reshape(n, self.m * chunk)
        codes = np.packbits(bits, axis=1).astype(np.uint8)
        return codes, thresholds

    def _compute_thresholds(self, vectors: np.ndarray, assignments: list[np.ndarray]) -> np.ndarray:
        """Global per-subvector mean over all vectors (1-bit quantisation thresholds)."""
        n, dim = vectors.shape
        chunk = dim // self.m
        if n == 0:
            return np.zeros(self.m, dtype=np.float32)
        reshaped = vectors.reshape(n, self.m, chunk)
        return reshaped.mean(axis=(0, 2)).astype(np.float32)

    def _write_list(self, cid: int, ids: np.ndarray, codes: np.ndarray, floats: np.ndarray | None) -> None:
        np.save(self.out_dir / f"list_{cid}.ids.npy", ids.astype(np.int64))
        np.save(self.out_dir / f"list_{cid}.codes.npy", codes.astype(np.uint8))
        if floats is not None and floats.size:
            np.save(self.out_dir / f"list_{cid}.floats.npy", floats.astype(np.float32))

    def _write_metadata(self, list_sizes: list[int]) -> None:
        meta = {
            "nlist": int(self.nlist),
            "m": int(self.m),
            "nbits": int(self.nbits),
            "dim": int(self.dim),
            "list_sizes": [int(s) for s in list_sizes],
            "use_rescore": bool(self.use_rescore),
            "doc_count": len(self.doc_ids),
        }
        (self.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    def _discover_list_files(self) -> dict[int, dict[str, Path]]:
        files: dict[int, dict[str, Path]] = {}
        for cid in range(self.nlist):
            entry: dict[str, Path] = {}
            ids = self.out_dir / f"list_{cid}.ids.npy"
            codes = self.out_dir / f"list_{cid}.codes.npy"
            if ids.exists() and codes.exists():
                entry["ids"] = ids
                entry["codes"] = codes
            fl = self.out_dir / f"list_{cid}.floats.npy"
            if fl.exists():
                entry["floats"] = fl
            files[cid] = entry
        return files

    def _coarse_quantize(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) to top-nprobe centroids for one query vector."""
        assert self.centroids is not None
        diffs = self.centroids - query[None, :]
        dists = (diffs * diffs).sum(axis=1)
        k = min(self.nprobe, self.centroids.shape[0])
        order = np.argpartition(dists, k - 1)[:k]
        order = order[np.argsort(dists[order])]
        return dists[order], order

    def _pack_query_bits(self, query: np.ndarray) -> np.ndarray:
        assert self.thresholds is not None
        chunk = self.dim // self.m
        reshaped = query.reshape(self.m, chunk)
        bits = (reshaped > self.thresholds[:, None]).astype(np.uint8).reshape(self.m * chunk)
        return np.packbits(bits).astype(np.uint8)

    def _hamming_to_codes(self, query_bits: np.ndarray, list_codes: np.ndarray) -> np.ndarray:
        """XOR query_bits against each packed code; popcount gives Hamming distance."""
        if list_codes.shape[0] == 0:
            return np.empty((0,), dtype=np.int32)
        xor = np.bitwise_xor(list_codes, query_bits[None, :])
        return np.unpackbits(xor, axis=1).sum(axis=1, dtype=np.int32)

    def _ram_search(self, queries: np.ndarray, k: int) -> list[SearchResult]:
        """RAM-resident search: uses in-memory codes + RFlat rescoring."""
        assert self._ram_codes is not None and self._ram_ids is not None
        out: list[SearchResult] = []
        for q in queries:
            _, probe_idx = self._coarse_quantize(q)
            qbits = self._pack_query_bits(q)
            cands: list[tuple[int, int, int]] = []
            for cid in probe_idx:
                codes = self._ram_codes[int(cid)]
                if codes.shape[0] == 0:
                    continue
                dists = self._hamming_to_codes(qbits, codes)
                for j in range(dists.shape[0]):
                    cands.append((int(dists[j]), int(cid), int(j)))
            cands.sort(key=lambda t: t[0])
            top = cands[: self.k_factor * k]
            if self.use_rescore and top:
                assert self._ram_floats is not None
                refined: list[tuple[float, int, int]] = []
                for _, cid, j in top:
                    vec = self._ram_floats[int(cid)][int(j)]
                    refined.append((float(vec @ q), int(cid), int(j)))
                refined.sort(key=lambda t: -t[0])
                top = [(s, c, j) for s, c, j in refined[:k]]
            else:
                top = top[:k]
            pairs = [
                (self.doc_ids[self._ram_ids[int(cid)][int(j)]], float(score))
                for score, cid, j in top
            ]
            out.append(SearchResult.from_pairs(pairs))
        return out

    def _disk_search(self, queries: np.ndarray, k: int) -> list[SearchResult]:
        """Disk-resident search: lazy-load top-nprobe lists via ThreadPoolExecutor."""
        out: list[SearchResult] = []
        with self.ensure_worker_pool() as pool:
            for q in queries:
                _, probe_idx = self._coarse_quantize(q)
                qbits = self._pack_query_bits(q)
                preload_count = min(len(probe_idx), self.prefetch_depth * self.nprobe)
                futures = {
                    int(cid): pool.submit(self._load_list_for_search, int(cid))
                    for cid in probe_idx[:preload_count]
                }
                cands: list[tuple[int, int, int, int]] = []
                for cid in probe_idx:
                    fut = futures.get(int(cid))
                    payload = fut.result() if fut is not None else self._load_list_for_search(int(cid))
                    if payload is None:
                        continue
                    codes, doc_indices, floats = payload
                    if codes.shape[0] == 0:
                        continue
                    dists = self._hamming_to_codes(qbits, codes)
                    for j in range(dists.shape[0]):
                        cands.append((int(dists[j]), int(cid), int(j), int(doc_indices[j])))
                cands.sort(key=lambda t: t[0])
                top = cands[: self.k_factor * k]
                if self.use_rescore and top:
                    refined: list[tuple[float, int]] = []
                    for _, cid, j, doc_idx in top:
                        fut = futures.get(int(cid))
                        payload = fut.result() if fut is not None else self._load_list_for_search(int(cid))
                        if payload is None:
                            continue
                        _codes, _doc_indices, floats = payload
                        if floats is None or floats.shape[0] == 0:
                            continue
                        vec = floats[j]
                        refined.append((float(vec @ q), int(doc_idx)))
                    refined.sort(key=lambda t: -t[0])
                    pairs = [(self.doc_ids[doc_idx], score) for score, doc_idx in refined[:k]]
                else:
                    pairs = [
                        (self.doc_ids[doc_idx], float(-dist))
                        for dist, _cid, _j, doc_idx in top[:k]
                    ]
                out.append(SearchResult.from_pairs(pairs))
        return out

    def _load_list_for_search(self, cid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Lazy-load list_{cid}.{ids,codes,floats} as memmaps; cache handle."""
        if cid in self.list_handles:
            cached = self.list_handles[cid]
            return cached["codes"], cached["ids"], cached.get("floats")
        paths = self.list_files.get(cid, {})
        ids_path = paths.get("ids")
        codes_path = paths.get("codes")
        if ids_path is None or codes_path is None:
            return None
        ids = np.load(ids_path, mmap_mode="r")
        codes = np.load(codes_path, mmap_mode="r")
        floats_path = paths.get("floats")
        floats = np.load(floats_path, mmap_mode="r") if floats_path is not None else None
        handles = {"ids": ids, "codes": codes, "floats": floats}
        self.list_handles[cid] = handles
        return codes, ids, floats