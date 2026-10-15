from __future__ import annotations

import json

import numpy as np
import pytest
from trivium.domain.document import Document
from trivium.io.corpus import CorpusCache


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_corpus_cache_loads_documents_and_vectors(tmp_path):
    rows = [{"id": "a", "title": "A", "text": "one"}, {"id": "b", "text": "two"}]
    write_jsonl(tmp_path / "corpus.jsonl", rows)
    write_jsonl(tmp_path / "scifact_seed.jsonl", rows[:1])
    np.savez(tmp_path / "vectors.npz", vectors=np.eye(2), ids=np.array(["a", "b"]))
    cache = CorpusCache(tmp_path)

    assert cache.load_scale(5000) == [Document.from_row(rows[0])]
    assert [d.doc_id for d in cache.load_scale(1)] == ["a"]
    vectors = cache.vectors_for_documents([Document("b")])
    assert vectors.tolist() == [[0.0, 1.0]]


def test_corpus_cache_rejects_duplicate_vector_ids(tmp_path):
    np.savez(tmp_path / "vectors.npz", vectors=np.ones((2, 2)), ids=np.array(["a", "a"]))

    with pytest.raises(ValueError, match="duplicate"):
        CorpusCache(tmp_path).load_vectors()


def test_corpus_cache_reports_missing_embedding(tmp_path):
    np.savez(tmp_path / "vectors.npz", vectors=np.ones((1, 2)), ids=np.array(["a"]))

    with pytest.raises(KeyError, match="no embedding"):
        CorpusCache(tmp_path).vectors_for_documents([Document("missing")])
