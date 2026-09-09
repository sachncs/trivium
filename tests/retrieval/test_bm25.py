"""Tests for trivium.retrieval.bm25.Bm25 save/load round-trip.

The bm25s on-disk format has no place for document IDs, so save()
must serialise doc_ids.npy alongside the index and load() must
restore them. Without this round-trip, search() after load()
returns SearchResult objects with no doc_id values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trivium.domain.document import Document
from trivium.retrieval.bm25 import Bm25


def _toy_corpus(n: int = 100, seed: int = 42) -> list[Document]:
    rng = np.random.RandomState(seed)
    return [
        Document(
            doc_id=f"doc-{i:04d}",
            title=f"title {i}",
            text=f"body {i} with words {' '.join(f'w{j}' for j in range(10))}",
        )
        for i in range(n)
    ]


class TestBm25RoundTrip:
    def test_save_persists_doc_ids(self, tmp_path: Path) -> None:
        docs = _toy_corpus()
        bm = Bm25()
        bm.add_documents(docs)
        bm.save(tmp_path)
        assert (tmp_path / "doc_ids.npy").exists()

    def test_load_restores_doc_ids(self, tmp_path: Path) -> None:
        docs = _toy_corpus()
        bm = Bm25()
        bm.add_documents(docs)
        bm.save(tmp_path)

        bm2 = Bm25()
        bm2.load(tmp_path)

        assert bm2.doc_ids == [d.doc_id for d in docs]
        assert bm2.id_to_pos == {d.doc_id: i for i, d in enumerate(docs)}

    def test_search_after_load_returns_correct_doc_ids(self, tmp_path: Path) -> None:
        """End-to-end: save, load, search -> hits carry the original doc_id."""
        docs = _toy_corpus()
        bm = Bm25()
        bm.add_documents(docs)
        bm.save(tmp_path)

        bm2 = Bm25()
        bm2.load(tmp_path)
        results = bm2.search(np.array(["body 5"]), k=5)
        assert len(results) == 1
        for hit in results[0]:
            assert hit.doc_id.startswith("doc-")
            assert hit.doc_id in {d.doc_id for d in docs}
            assert hit.score > 0.0

    def test_load_without_doc_ids_file_yields_empty(self, tmp_path: Path) -> None:
        """Back-compat: a pre-fix saved index has no doc_ids.npy and loads cleanly."""
        docs = _toy_corpus()

        bm = Bm25()
        bm.add_documents(docs)
        bm.save(tmp_path)

        # Delete the doc_ids file to simulate a pre-fix save.
        (tmp_path / "doc_ids.npy").unlink()

        bm2 = Bm25()
        bm2.load(tmp_path)
        assert bm2.doc_ids == []
        assert bm2.id_to_pos == {}

    def test_round_trip_preserves_scores(self, tmp_path: Path) -> None:
        """Top-1 hit's score must match across save/load."""
        docs = _toy_corpus()
        bm = Bm25()
        bm.add_documents(docs)
        query = np.array(["body 42"])
        before = bm.search(query, k=1)

        bm.save(tmp_path)
        bm2 = Bm25()
        bm2.load(tmp_path)
        after = bm2.search(query, k=1)

        assert before[0][0].doc_id == after[0][0].doc_id
        assert before[0][0].score == pytest.approx(after[0][0].score, rel=1e-3)
