"""Public helpers used by the pipeline implementations.

Module-named `helpers.py` and function-named `*_step` rather than
semi-private `_do_*` so the no-semi-private rule holds.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.domain.document import Document
from trivium.domain.query import Query
from trivium.domain.result import SearchResult
from trivium.evaluation.latency import LatencyProbe, LatencyStats
from trivium.retrieval.base import Retriever


def hits_to_run_pairs(
    queries: Sequence[Query], results: Sequence[SearchResult]
) -> dict[str, SearchResult]:
    """Map query_id -> SearchResult for the runner that iterates per query."""
    return {q.query_id: r for q, r in zip(queries, results, strict=False)}


def query_doc_ids(result: SearchResult, doc_ids: list[str], top_k: int) -> list[tuple[int, float]]:
    """Convert SearchResult (doc_id, score) into (corpus_position, score)."""
    id_to_pos = {d: i for i, d in enumerate(doc_ids)}
    return [(id_to_pos[h.doc_id], h.score) for h in result if h.doc_id in id_to_pos][:top_k]


def retriever_search_step(
    retriever: Retriever,
    documents: Sequence[Document],
    query_vecs: np.ndarray,
    top_k: int,
    n_warmup: int = 1,
    n_iter: int = 100,
    rotate: bool = True,
) -> tuple[list[SearchResult], LatencyStats]:
    """Convenience: run a search, return (results, latency_stats) per call.

    Replaces the legacy _do_hybrid_query closure that built bm25_hits
    and vec_hits inline. Here the retriever is passed in, so this
    helper works for any single-retriever search.
    """
    results = retriever.search(query_vecs, top_k)

    if rotate and query_vecs.shape[0] > 0:
        rotate_inputs = [query_vecs[i % query_vecs.shape[0]] for i in range(n_iter)]
        probe = LatencyProbe(
            fn=lambda v: retriever.search(v.reshape(1, -1).astype(np.float32), top_k),
            n=n_iter,
            warmup=n_warmup,
            rotate=rotate_inputs,
        )
        stats = probe.run()
    else:
        probe = LatencyProbe(
            fn=lambda: retriever.search(query_vecs[:1], top_k),
            n=n_iter,
            warmup=n_warmup,
        )
        stats = probe.run()
    return results, stats


def hybrid_query_step(
    retriever_a: Retriever,
    retriever_b: Retriever,
    documents: Sequence[Document],
    query_vecs: np.ndarray,
    top_k: int,
    fusion,
):
    """Run both retrievers end-to-end and fuse the per-query result lists.

    Replaces the legacy _do_hybrid_query closure (which had 7 args and
    was module-bottom semi-private). Now a public function with a
    focus on composition.
    """
    results_a = retriever_a.search(query_vecs, top_k)
    results_b = retriever_b.search(query_vecs, top_k)
    return [fusion.fuse([a, b], top_k) for a, b in zip(results_a, results_b, strict=False)]


def rerank_query_step(
    fusion_result: SearchResult,
    query_text: str,
    documents: Sequence[Document],
    reranker,
    top_k_rerank: int,
    top_k_out: int,
) -> SearchResult:
    """Apply a Reranker to the fused candidate set.

    Replaces the legacy _do_rerank_query closure.
    """
    id_to_doc = {d.doc_id: d for d in documents}
    candidates = [id_to_doc[h.doc_id] for h in fusion_result if h.doc_id in id_to_doc][
        :top_k_rerank
    ]
    return reranker.rerank(query_text, candidates, top_k_out)
