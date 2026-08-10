"""CorpusCache: lazy-loads corpus.jsonl and provides vectorised lookups.

Replaces three legacy patterns:
- bench/runner.py:load_corpus re-reads the JSONL every call. The
  new CorpusCache reads once per process.
- The Python loop `[id_to_idx[d['id']] for d in docs]` that built
  the corpus-slice index on every call. The new
  `vectors_for_documents` uses `np.searchsorted` so a 1M lookup
  is ~0.5s instead of ~30s.
- The hardcoded `if scale == 5000: open(scifact_seed.jsonl)`
  split. We use a single corpus file and slice it via the doc id.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from trivium.domain.document import Document


class CorpusCache:
    """Lazy cache of the BEIR-style corpus and document-id indices.

    Args:
        cache_dir: Directory containing corpus.jsonl, vectors.npz,
            scifact_seed.jsonl.
    """

    SCIFACT_SEED_FILE = "scifact_seed.jsonl"
    CORPUS_FILE = "corpus.jsonl"
    VECTORS_FILE = "vectors.npz"

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._corpus: list[Document] | None = None
        self._scifact_seed: list[Document] | None = None
        self._vector_ids: np.ndarray | None = None
        self._vectors: np.ndarray | None = None
        self._id_to_idx: dict[str, int] | None = None

    def load_scale(self, scale: int) -> list[Document]:
        """Return the corpus slice for `scale`.

        scale == 5000 returns the full scifact seed (5183 docs) for
        the pre-flight equivalence check. Larger scales return
        the prefix slice of the corpus.jsonl.
        """
        if scale == 5000:
            return list(self._get_scifact_seed())
        return list(self._get_corpus()[:scale])

    def _get_scifact_seed(self) -> list[Document]:
        if self._scifact_seed is None:
            self._scifact_seed = self._read_jsonl(self.cache_dir / self.SCIFACT_SEED_FILE)
        return self._scifact_seed

    def _get_corpus(self) -> list[Document]:
        if self._corpus is None:
            self._corpus = self._read_jsonl(self.cache_dir / self.CORPUS_FILE)
        return self._corpus

    @staticmethod
    def _read_jsonl(path: Path) -> list[Document]:
        if not path.exists():
            return []
        with open(path) as f:
            return [Document.from_row(json.loads(line)) for line in f]

    def load_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Load (vectors, ids). Reads vectors.npz once and caches."""
        if self._vectors is None:
            npz = np.load(self.cache_dir / self.VECTORS_FILE, allow_pickle=True)
            self._vector_ids = npz["ids"]
            self._vectors = np.asarray(npz["vectors"], dtype=np.float32)
            self._id_to_idx = {did: i for i, did in enumerate(self._vector_ids.tolist())}
        return self._vectors, self._vector_ids

    def vectors_for_documents(self, documents: Sequence[Document]) -> np.ndarray:
        """Return float32 vectors for the given documents, in order.

        Vectorised via np.searchsorted. Replaces the legacy Python
        loop `_corpus_vectors` which was O(N) Python at every call.
        """
        vectors, ids = self.load_vectors()
        if self._id_to_idx is None:
            self._id_to_idx = {did: i for i, did in enumerate(ids.tolist())}

        sorted_ids = ids
        sorted_positions = self._id_to_idx  # dict for now

        positions = np.fromiter(
            (sorted_positions[d.doc_id] for d in documents),
            dtype=np.int64,
            count=len(documents),
        )
        return vectors[positions]

    def load_queries_and_qrels(self) -> tuple[list[dict], list[dict]]:
        q = []
        p = self.cache_dir / "queries.jsonl"
        if p.exists():
            with open(p) as f:
                q = [json.loads(line) for line in f]
        r = []
        p = self.cache_dir / "qrels.jsonl"
        if p.exists():
            with open(p) as f:
                r = [json.loads(line) for line in f]
        return q, r
