"""Hybrid fusion: RRF (Cormack 2009) over BM25 + dense result lists."""
from __future__ import annotations


def rrf(
    bm25_hits: list[tuple[str, float]],
    vec_hits: list[tuple[str, float]],
    k: int = 60,
    weights: tuple[float, float] = (1.0, 1.0),
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Standard formula: score(d) = sum_i w_i / (k + rank_i(d))."""
    rb, rv = weights
    scores: dict[str, float] = {}
    for rank, (did, _) in enumerate(bm25_hits, start=1):
        scores[did] = scores.get(did, 0.0) + rb / (k + rank)
    for rank, (did, _) in enumerate(vec_hits, start=1):
        scores[did] = scores.get(did, 0.0) + rv / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def to_dict(hits: list[tuple[str, float]]) -> dict[str, float]:
    return {did: float(score) for did, score in hits}
