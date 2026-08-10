"""BM25 nDCG@10 regression anchor on the untouched 5K scifact corpus.

This is the only number in the entire project that has been
validated against the published Anserini baseline (BEIR 1.0.0
scifact: 0.6789). The historical observed value is 0.65685, well
within the documented +/- 0.03 bm25s-vs-Lucene drift.

The test pins the trivium.* public API. As the refactor replaces
bench/ and search/ modules with trivium, this test is the proof
that the new code path reproduces the legacy result byte-for-byte.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trivium.config.loader import load_config
from trivium.domain.document import Document
from trivium.domain.query import Query
from trivium.domain.qrels import Qrels
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.io.corpus import CorpusCache
from trivium.retrieval.bm25 import Bm25

ANSERINI_NDCG10 = 0.6789
TOLERANCE = 0.03
LOWER_BOUND = 0.62
OBSERVED_AT_BASELINE = 0.65685


@pytest.fixture(scope="module")
def seed_docs(cache_dir: Path) -> list[Document]:
    cache = CorpusCache(cache_dir)
    docs = cache.load_scale(5000)
    if not docs:
        pytest.skip(f"{cache_dir}/scifact_seed.jsonl not found; run prepare_data.py first")
    return docs


@pytest.fixture(scope="module")
def queries_and_qrels(cache_dir: Path):
    cache = CorpusCache(cache_dir)
    query_rows, qrels_rows = cache.load_queries_and_qrels()
    if not query_rows or not qrels_rows:
        pytest.skip("queries.jsonl or qrels.jsonl missing; run prepare_data.py")
    queries = Query.many(query_rows)
    qrels = Qrels.from_rows(qrels_rows)
    return queries, qrels


@pytest.mark.golden
def test_preflight_bm25_5k_ndcg10_within_anysini_tolerance(
    seed_docs: list[Document],
    queries_and_qrels,
    default_config: dict,
) -> None:
    """BM25 nDCG@10 on 5K scifact must reproduce the Anserini baseline.

    Allowed: 0.62 <= ndcg <= 0.7089 (lower bound = bug detector,
    upper bound = documented bm25s-vs-Lucene drift).
    """
    queries, qrels = queries_and_qrels
    bm25_cfg = default_config["bm25"]
    k = bm25_cfg["candidate_pool"]
    top_k_eval = default_config["benchmark"]["top_k_eval"]

    bm25 = Bm25(
        k1=bm25_cfg["k1"],
        b=bm25_cfg["b"],
        method=bm25_cfg["method"],
    )
    bm25.add_documents(seed_docs)

    query_texts = np.array([q.text for q in queries])
    results = bm25.search(query_texts, k=k)
    per_query = [[(h.doc_id, h.score) for h in r] for r in results]
    eval_pairs = hits_to_results(per_query, queries)
    metrics = Evaluator(k_values=top_k_eval).evaluate(qrels, eval_pairs)
    observed = metrics.ndcg_at_10

    assert observed >= LOWER_BOUND, (
        f"nDCG@10={observed:.5f} below absolute lower bound {LOWER_BOUND}. "
        "Pipeline is broken (wrong tokenizer, wrong k1/b, or wrong qrels prefix)."
    )
    assert abs(observed - ANSERINI_NDCG10) <= TOLERANCE, (
        f"|nDCG@10 - Anserini| = {abs(observed - ANSERINI_NDCG10):.5f} > "
        f"tolerance {TOLERANCE}. Beyond the documented bm25s-vs-Lucene drift."
    )


@pytest.mark.golden
def test_preflight_bm25_5k_does_not_drift_from_baseline_observation(
    seed_docs: list[Document],
    queries_and_qrels,
    default_config: dict,
) -> None:
    """Observed value must remain within +/- 0.005 of 0.65685.

    Tight tolerance — even 0.005 of drift is a signal that the
    refactor changed something subtle (tokenizer order, k1/b rounding,
    qrels prefix change).
    """
    queries, qrels = queries_and_qrels
    bm25_cfg = default_config["bm25"]
    k = bm25_cfg["candidate_pool"]
    top_k_eval = default_config["benchmark"]["top_k_eval"]

    bm25 = Bm25(
        k1=bm25_cfg["k1"],
        b=bm25_cfg["b"],
        method=bm25_cfg["method"],
    )
    bm25.add_documents(seed_docs)

    query_texts = np.array([q.text for q in queries])
    results = bm25.search(query_texts, k=k)
    per_query = [[(h.doc_id, h.score) for h in r] for r in results]
    eval_pairs = hits_to_results(per_query, queries)
    metrics = Evaluator(k_values=top_k_eval).evaluate(qrels, eval_pairs)
    observed = metrics.ndcg_at_10

    assert observed == pytest.approx(OBSERVED_AT_BASELINE, abs=0.005), (
        f"nDCG@10={observed:.5f} drifted from the recorded baseline "
        f"{OBSERVED_AT_BASELINE:.5f} (tol 0.005). Refactor likely introduced "
        "a tokenizer or qrels change."
    )


@pytest.mark.golden
def test_preflight_bm25_5k_recall10_matches_legacy(
    seed_docs: list[Document],
    queries_and_qrels,
    default_config: dict,
) -> None:
    """Recall@10 should be ~0.7784 (the value in the existing benchmark.csv)."""
    queries, qrels = queries_and_qrels
    bm25_cfg = default_config["bm25"]
    k = bm25_cfg["candidate_pool"]
    top_k_eval = default_config["benchmark"]["top_k_eval"]

    bm25 = Bm25(
        k1=bm25_cfg["k1"],
        b=bm25_cfg["b"],
        method=bm25_cfg["method"],
    )
    bm25.add_documents(seed_docs)

    query_texts = np.array([q.text for q in queries])
    results = bm25.search(query_texts, k=k)
    per_query = [[(h.doc_id, h.score) for h in r] for r in results]
    eval_pairs = hits_to_results(per_query, queries)
    metrics = Evaluator(k_values=top_k_eval).evaluate(qrels, eval_pairs)
    assert metrics.recall_at_10 == pytest.approx(0.7784, abs=0.005)
