"""Unit tests for trivium.domain dataclasses."""
from __future__ import annotations

import pytest

from trivium.domain.document import Document
from trivium.domain.hit import Hit
from trivium.domain.pipeline_result import FORMAT_VERSION, PipelineResult
from trivium.domain.query import Query
from trivium.domain.qrels import Qrels
from trivium.domain.result import SearchResult


class TestDocument:
    def test_body_concatenates_title_and_text(self):
        d = Document("d1", "Title", "Body text")
        assert d.body == "Title\nBody text"

    def test_body_handles_empty_title(self):
        d = Document("d2", "", "only body")
        assert d.body == "only body"

    def test_body_handles_empty_text(self):
        d = Document("d3", "only title", "")
        assert d.body == "only title"

    def test_body_handles_both_empty(self):
        d = Document("d4", "", "")
        assert d.body == ""

    def test_from_row(self):
        d = Document.from_row({"id": "x", "title": "T", "text": "B"})
        assert d.doc_id == "x"
        assert d.body == "T\nB"

    def test_frozen(self):
        d = Document("d", "T", "B")
        with pytest.raises(Exception):
            d.doc_id = "other"  # type: ignore[misc]


class TestQuery:
    def test_from_row(self):
        q = Query.from_row({"id": "q1", "text": "hello"})
        assert q.query_id == "q1"
        assert q.text == "hello"


class TestQrels:
    def test_from_rows_nested(self):
        rows = [
            {"qid": "q1", "did": "d1", "rel": 1},
            {"qid": "q1", "did": "d2", "rel": 0},
            {"qid": "q2", "did": "d1", "rel": 1},
        ]
        qrels = Qrels.from_rows(rows)
        assert qrels.to_pytrec() == {"q1": {"d1": 1, "d2": 0}, "q2": {"d1": 1}}

    def test_relevant_for_unknown_query(self):
        qrels = Qrels()
        assert qrels.relevant_for("nope") == {}

    def test_all_query_ids(self):
        rows = [{"qid": "q1", "did": "d1", "rel": 1}, {"qid": "q2", "did": "d1", "rel": 0}]
        qrels = Qrels.from_rows(rows)
        assert sorted(qrels.all_query_ids()) == ["q1", "q2"]


class TestSearchResult:
    def test_from_pairs_and_to_pairs_roundtrip(self):
        sr = SearchResult.from_pairs([("d1", 0.9), ("d2", 0.8)])
        assert sr.to_pairs() == [("d1", 0.9), ("d2", 0.8)]

    def test_top_k(self):
        sr = SearchResult.from_pairs([(f"d{i}", 0.9 - i * 0.1) for i in range(5)])
        assert len(sr.top_k(3)) == 3

    def test_top_k_zero(self):
        sr = SearchResult.from_pairs([("d1", 0.5)])
        assert len(sr.top_k(0)) == 0

    def test_iteration(self):
        sr = SearchResult.from_pairs([("d1", 0.9), ("d2", 0.8)])
        assert [h.doc_id for h in sr] == ["d1", "d2"]

    def test_indexing(self):
        sr = SearchResult.from_pairs([("d1", 0.9), ("d2", 0.8)])
        assert sr[0] == Hit("d1", 0.9)

    def test_empty(self):
        assert len(SearchResult.empty()) == 0


class TestPipelineResult:
    def test_format_version_default(self):
        from trivium.evaluation.latency import LatencyStats
        from trivium.evaluation.metrics import EvaluationMetrics

        pr = PipelineResult(
            name="bm25",
            scale=5183,
            encoder="none",
            reranker="none",
            metrics=EvaluationMetrics(ndcg_at_10=0.66),
            latency=LatencyStats(0.0, 1.0, 2.0, 3.0, 0.5, 100),
        )
        d = pr.to_dict()
        assert d["format_version"] == FORMAT_VERSION
        assert d["mode"] == "bm25"
        assert d["scale"] == 5183
        assert d["ndcg_cut.10"] == 0.66
        assert d["lat_p95_ms"] == 1.0
