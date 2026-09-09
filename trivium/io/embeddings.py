"""EmbeddingCache: persist embeddings per (encoder_slug, encoder_revision).

Replaces the legacy behaviour where embeddings lived in a single
data/cache/vectors.npz and adding a new encoder required either
a destructive re-embed or a renamed second file outside the cache.

Layout:
  data/cache/vectors_{slug}.npz          one file per encoder slug
  data/cache/embedding_index.json       maps (slug, revision) -> file + sha

get() returns the (vectors, ids) tuple from cache, or None if
not cached. put() writes a new npz keyed on (slug, revision).
The revision-hash includes model_id + max_seq_length + prompt
prefixes, so a prompt-template upgrade is a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def encoder_revision(config: dict) -> str:
    """Stable SHA-256 over the fields that change embedding outputs."""
    payload = json.dumps(
        {
            "model_id": config.get("model_id", ""),
            "max_seq_length": config.get("max_seq_length", 256),
            "normalize": config.get("normalize", True),
            "prompt_prefix_doc": config.get("prompt_prefix_doc", ""),
            "prompt_prefix_query": config.get("prompt_prefix_query", ""),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def safe_slug(slug: str) -> str:
    """Reduce an encoder slug to a filename-safe identifier."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", slug)


class EmbeddingCache:
    """Per-encoder-slug embedding cache keyed on revision hash."""

    INDEX_FILE = "embedding_index.json"

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_index: dict | None = None

    def index(self) -> dict:
        """Lazy-load the embedding index."""
        if self.cache_index is None:
            p = self.cache_dir / self.INDEX_FILE
            self.cache_index = json.loads(p.read_text()) if p.exists() else {}
        return self.cache_index

    def file_for(self, slug: str, revision: str) -> Path:
        return self.cache_dir / f"vectors_{safe_slug(slug)}__{revision}.npz"

    def get(self, slug: str, revision: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (vectors, ids) if cached for (slug, revision). Else None."""
        entries = self.index()
        record = entries.get(f"{slug}@{revision}")
        if record is None:
            return None
        path = self.cache_dir / record["file"]
        if not path.exists():
            return None
        npz = np.load(path, allow_pickle=True)
        vectors = np.asarray(npz["vectors"], dtype=np.float32)
        ids = np.asarray(npz["ids"])
        return vectors, ids

    def put(
        self, slug: str, revision: str, vectors: np.ndarray, ids: Sequence[str], doc_count: int
    ) -> None:
        """Persist (vectors, ids) and record the entry in the index."""
        path = self.file_for(slug, revision)
        np.savez_compressed(path, vectors=vectors, ids=np.array(list(ids)))
        idx = self.index()
        idx[f"{slug}@{revision}"] = {
            "file": path.name,
            "doc_count": int(doc_count),
            "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        }
        (self.cache_dir / self.INDEX_FILE).write_text(json.dumps(idx, indent=2, sort_keys=True))
        self.cache_index = idx
