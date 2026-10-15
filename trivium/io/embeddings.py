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
import os
import re
import tempfile
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
        try:
            with np.load(path, allow_pickle=False) as npz:
                vectors = np.asarray(npz["vectors"], dtype=np.float32)
                ids = np.asarray(npz["ids"])
        except (OSError, ValueError, KeyError, EOFError):
            return None
        if vectors.ndim != 2 or ids.ndim != 1 or len(vectors) != len(ids):
            return None
        return vectors, ids

    def put(
        self, slug: str, revision: str, vectors: np.ndarray, ids: Sequence[str], doc_count: int
    ) -> None:
        """Persist (vectors, ids) and record the entry in the index."""
        vectors = np.asarray(vectors, dtype=np.float32)
        id_array = np.asarray(list(ids), dtype=str)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D; got shape {vectors.shape}")
        if len(vectors) != len(id_array) or len(vectors) != int(doc_count):
            raise ValueError("vectors, ids, and doc_count must contain the same number of rows")
        if len(set(id_array.tolist())) != len(id_array):
            raise ValueError("embedding ids must be unique")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.file_for(slug, revision)
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as tmp:
            temp_path = Path(tmp.name)
        try:
            np.savez_compressed(temp_path, vectors=vectors, ids=id_array)
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        idx = self.index()
        idx[f"{slug}@{revision}"] = {
            "file": path.name,
            "doc_count": int(doc_count),
            "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        }
        index_path = self.cache_dir / self.INDEX_FILE
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.cache_dir, suffix=".json", encoding="utf-8", delete=False
        ) as tmp:
            json.dump(idx, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            temp_index = Path(tmp.name)
        try:
            os.replace(temp_index, index_path)
        finally:
            temp_index.unlink(missing_ok=True)
        self.cache_index = idx
