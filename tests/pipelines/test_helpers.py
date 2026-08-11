"""Unit tests for trivium.pipelines.helpers (public, no semi-private)."""

from __future__ import annotations

import numpy as np

from trivium.domain.document import Document
from trivium.domain.query import Query
from trivium.domain.result import SearchResult
from trivium.fusion.rrf import Rrf
from trivium.pipelines.helpers import (
    hybrid_query_step,
    query_doc_ids,
    rerank_query_step,
    retriever_search_step,
)


class StubRetriever:
    """Trivial retriever for testing pipeline helpers."""

    def __init__(self, hits: list[tuple[str, float]]) -> None:
        self._hits = hits
        self.slug = f"stub-{id(self)}"

    def add_documents(self, *_, **__):
        pass

    def search(self, query_vectors, k):
        # One SearchResult per input query (per ABC contract).
        n = max(1, len(query_vectors))
        return [SearchResult.from_pairs(self._hits[:k]) for _ in range(n)]

    def set_search_params(self, **params):
        pass

    def size_bytes(self):
        return 0


def test_query_doc_ids():
    docs = [Document("d1", "T1", "B1"), Document("d2", "T2", "B2")]
    result = SearchResult.from_pairs([("d1", 0.9), ("d2", 0.8)])
    pos_pairs = query_doc_ids(result, [d.doc_id for d in docs], top_k=2)
    assert pos_pairs == [(0, 0.9), (1, 0.8)]


def test_retriever_search_step_returns_results_and_stats():
    retr = StubRetriever([("d1", 0.9)])
    docs = [Document("d1", "T", "B")]
    query_vecs = np.zeros((3, 4), dtype=np.float32)
    results, stats = retriever_search_step(retr, docs, query_vecs, top_k=5, n_warmup=1, n_iter=5)
    assert len(results) == 3
    assert stats.n == 5


def test_hybrid_query_step_combines():
    a = StubRetriever([("d1", 0.9), ("d2", 0.5)])
    b = StubRetriever([("d2", 0.9), ("d1", 0.5)])
    docs = [Document("d1", "T1", "B1"), Document("d2", "T2", "B2")]
    query_vecs = np.zeros((2, 4), dtype=np.float32)
    fusion = Rrf(k=60)
    fused = hybrid_query_step(a, b, docs, query_vecs, top_k=2, fusion=fusion)
    assert len(fused) == 2


class CountingReranker:
    slug = "counting"

    def rerank(self, query, candidates, top_k):
        return SearchResult.from_pairs([(c.doc_id, 1.0) for c in candidates[:top_k]])


def test_rerank_query_step_applies_reranker():
    fusion_result = SearchResult.from_pairs([("d1", 0.9), ("d2", 0.5)])
    docs = [Document("d1", "T1", "B1"), Document("d2", "T2", "B2")]
    reranker = CountingReranker()
    ranked = rerank_query_step(fusion_result, "query", docs, reranker, top_k_rerank=2, top_k_out=5)
    assert all(h.doc_id in {"d1", "d2"} for h in ranked)
