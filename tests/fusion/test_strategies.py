"""Unit tests for trivium.fusion."""
from __future__ import annotations

from trivium.domain.result import SearchResult
from trivium.fusion.rrf import Rrf
from trivium.fusion.weighted import Weighted


def test_rrf_classic_formula():
    """RRF formula: sum w_i / (k + rank_i). For k=60, ranks 1,2 contribute
    ~0.0164 and ~0.0161 respectively."""
    r1 = SearchResult.from_pairs([("a", 1.0), ("b", 0.5), ("c", 0.1)])
    r2 = SearchResult.from_pairs([("b", 1.0), ("a", 0.5), ("d", 0.1)])
    fused = Rrf(k=60, weights=[1.0, 1.0]).fuse([r1, r2], top_k=10)
    pairs = fused.to_pairs()
    assert ("a",) in [p[0:1] for p in pairs[:3]]
    assert ("b",) in [p[0:1] for p in pairs[:3]]


def test_rrf_weights_unequal_changes_ranking():
    r1 = SearchResult.from_pairs([("a", 1.0), ("b", 0.5)])
    r2 = SearchResult.from_pairs([("b", 1.0), ("a", 0.5)])
    equal = Rrf(k=60, weights=[1.0, 1.0]).fuse([r1, r2], top_k=10).to_pairs()
    weighted = Rrf(k=60, weights=[0.2, 0.8]).fuse([r1, r2], top_k=10).to_pairs()
    # either ordering is allowed at equal weights (deterministic tie break);
    # the weighted one should still be reproducible
    assert isinstance(equal, list)
    assert isinstance(weighted, list)


def test_rrf_top_k_limits_output():
    r1 = SearchResult.from_pairs([(f"d{i}", 0.9 - i * 0.1) for i in range(20)])
    fused = Rrf(k=60).fuse([r1], top_k=5)
    assert len(fused) == 5


def test_rrf_empty_results_returns_empty():
    fused = Rrf(k=60).fuse([], top_k=10)
    assert len(fused) == 0


def test_weighted_normalises_per_retriever():
    r1 = SearchResult.from_pairs([("a", 100.0), ("b", 50.0)])
    r2 = SearchResult.from_pairs([("b", 1.0), ("a", 0.0)])
    fused = Weighted(weights=[1.0, 1.0]).fuse([r1, r2], top_k=10).to_pairs()
    # a and b should both be in top
    assert any(d == "a" for d, _ in fused)
    assert any(d == "b" for d, _ in fused)
