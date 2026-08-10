"""Weighted score fusion: linear combination per-doc of normalised scores."""
from __future__ import annotations

from collections.abc import Sequence

from trivium.domain.hit import Hit
from trivium.domain.result import SearchResult
from trivium.fusion.base import FusionStrategy


class Weighted(FusionStrategy):
    """Linear weighted fusion after per-retriever min-max normalisation."""

    def __init__(self, weights: Sequence[float] | None = None) -> None:
        self.weights = list(weights) if weights is not None else None

    @property
    def slug(self) -> str:
        w = self.weights or []
        if not w:
            return "weighted"
        return "weighted_" + "_".join(f"{x:.2f}" for x in w).replace(".", "p")

    def fuse(self, results: Sequence[SearchResult], top_k: int) -> SearchResult:
        if not results:
            return SearchResult.empty()

        weights = self.weights or [1.0] * len(results)
        normed: list[dict[str, float]] = []
        for r in results:
            pairs = r.to_pairs()
            if not pairs:
                normed.append({})
                continue
            scores = [s for _, s in pairs]
            lo, hi = min(scores), max(scores)
            span = hi - lo if hi > lo else 1.0
            normed.append({d: (s - lo) / span for d, s in pairs})

        combined: dict[str, float] = {}
        for i, dmap in enumerate(normed):
            w = weights[i] if i < len(weights) else 1.0
            for d, s in dmap.items():
                combined[d] = combined.get(d, 0.0) + w * s

        ordered = sorted(combined.items(), key=lambda x: (-x[1], x[0]))
        return SearchResult(hits=(Hit(doc_id=d, score=float(s)) for d, s in ordered[:top_k]))
