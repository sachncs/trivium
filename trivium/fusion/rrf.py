"""Rrf (Reciprocal Rank Fusion, Cormack 2009).

score(d) = sum_i w_i / (k + rank_i(d))

Replaces the legacy search/hybrid.py:rrf() function with a class
that returns SearchResult (was list[tuple[str, float]]).
"""
from __future__ import annotations

from collections.abc import Sequence

from trivium.domain.hit import Hit
from trivium.domain.result import SearchResult
from trivium.fusion.base import FusionStrategy


class Rrf(FusionStrategy):
    """Reciprocal Rank Fusion with per-list weights and a rank-constant k.

    Args:
        k: Rank-constant smoothing (default 60 = standard RRF).
        weights: Per-retriever weights. If None, equal weights.
    """

    def __init__(self, k: int = 60, weights: Sequence[float] | None = None) -> None:
        self.k = int(k)
        self.weights = list(weights) if weights is not None else None

    @property
    def slug(self) -> str:
        return f"rrf_k{self.k}"

    def fuse(self, results: Sequence[SearchResult], top_k: int) -> SearchResult:
        """Combine N per-query result lists into a single ranked list."""
        if not results:
            return SearchResult.empty()

        weights = self.weights or [1.0] * len(results)
        scores: dict[str, float] = {}
        for ri, hits_per_doc in enumerate(results):
            weight = weights[ri] if ri < len(weights) else 1.0
            for rank, hit in enumerate(hits_per_doc, start=1):
                scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + weight / (self.k + rank)

        ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return SearchResult(hits=(Hit(doc_id=d, score=float(s)) for d, s in ordered[:top_k]))
