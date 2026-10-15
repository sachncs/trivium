from __future__ import annotations

import numpy as np
import pytest
from trivium.io.embeddings import EmbeddingCache


def test_embedding_cache_round_trip_is_numeric_and_atomic(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put("demo", "revision", np.eye(2), ["a", "b"], 2)

    vectors, ids = cache.get("demo", "revision")

    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 2)
    assert ids.tolist() == ["a", "b"]


def test_embedding_cache_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        EmbeddingCache(tmp_path).put("demo", "revision", np.ones((2, 2)), ["a", "a"], 2)


def test_embedding_cache_treats_corrupt_npz_as_cache_miss(tmp_path):
    cache = EmbeddingCache(tmp_path)
    path = cache.file_for("demo", "revision")
    path.write_bytes(b"not an npz file")
    cache.cache_index = {"demo@revision": {"file": path.name}}

    assert cache.get("demo", "revision") is None
