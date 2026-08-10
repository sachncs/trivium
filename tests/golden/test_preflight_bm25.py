"""BM25 nDCG@10 regression anchor on the untouched 5K scifact corpus.

This is the only number in the entire project that has been
validated against the published Anserini baseline (BEIR 1.0.0
scifact: 0.6789). The historical observed value is 0.65685, well
within the documented +/- 0.03 bm25s-vs-Lucene drift.

If this test fails the pipeline is broken; the refactor must have
shifted the tokenizer, k1/b parameters, or qrels prefix. The test
uses the cache at data/cache/{scifact_seed.jsonl, queries.jsonl,
qrels.jsonl} which the user must populate via 'python -m data.prepare'.

When the trivium refactor lands, this test will be updated to use
the new trivium.retrieval.bm25 + trivium.evaluation.metrics APIs.
Until then it pins the legacy bench/search API behaviour.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from bench.metrics import evaluate, make_qrels
from bench.runner import load_queries
from search.bm25_index import BM25Index

ANSERINI_NDCG10 = 0.6789
TOLERANCE = 0.03
LOWER_BOUND = 0.62
OBSERVED_AT_BASELINE = 0.65685


@pytest.fixture(scope="module")
def seed_docs(cache_dir: Path) -> list[dict]:
    path = cache_dir / "scifact_seed.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not found; run 'python -m data.prepare' first")
    with open(path) as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def queries_and_qrels(cache_dir: Path):
    qrels_path = cache_dir / "qrels.jsonl"
    queries_path = cache_dir / "queries.jsonl"
    if not qrels_path.exists() or not queries_path.exists():
        pytest.skip("qrels.jsonl or queries.jsonl missing; run 'python -m data.prepare'")
    queries = [json.loads(l) for l in open(queries_path)]
    qrels = make_qrels([json.loads(l) for l in open(qrels_path)])
    return queries, qrels


@pytest.mark.golden
def test_preflight_bm25_5k_ndcg10_within_anysini_tolerance(
    seed_docs: list[dict],
    queries_and_qrels: tuple,
    default_config: dict,
) -> None:
    """BM25 nDCG@10 on 5K scifact must reproduce Anserini baseline.

    Allowed: 0.62 <= ndcg <= 0.7089 (lower bound = bug detector,
    upper bound = documented bm25s-vs-Lucene drift).
    """
    queries, qrels = queries_and_qrels
    bm25_cfg = default_config["bm25"]
    k = bm25_cfg["candidate_pool"]
    top_k_eval: Iterable[int] = default_config["benchmark"]["top_k_eval"]

    bm25 = BM25Index(
        k1=bm25_cfg["k1"],
        b=bm25_cfg["b"],
        method=bm25_cfg["method"],
    )
    bm25.build(
        [d["title"] + "\n" + d["text"] for d in seed_docs],
        show_progress=False,
    )

    per_query: list[list[tuple[str, float]]] = []
    for q in queries:
        idxs, scores = bm25.query(q["text"], k=k)
        per_query.append([(seed_docs[i]["id"], float(s)) for i, s in zip(idxs, scores)])

    results = {q["id"]: dict(pairs) for q, pairs in zip(queries, per_query)}
    ndcg = evaluate(qrels, results, k_values=top_k_eval)
    observed = ndcg["ndcg_cut.10"]

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
    seed_docs: list[dict],
    queries_and_qrels: tuple,
    default_config: dict,
) -> None:
    """Observed value must remain within +/- 0.005 of 0.65685.

    Tight tolerance — even 0.005 of drift is a signal that the
    refactor changed something subtle (tokenizer order, k1/b rounding,
    qrels prefix).
    """
    queries, qrels = queries_and_qrels
    bm25_cfg = default_config["bm25"]
    k = bm25_cfg["candidate_pool"]
    top_k_eval = default_config["benchmark"]["top_k_eval"]

    bm25 = BM25Index(
        k1=bm25_cfg["k1"],
        b=bm25_cfg["b"],
        method=bm25_cfg["method"],
    )
    bm25.build(
        [d["title"] + "\n" + d["text"] for d in seed_docs],
        show_progress=False,
    )

    per_query = []
    for q in queries:
        idxs, scores = bm25.query(q["text"], k=k)
        per_query.append([(seed_docs[i]["id"], float(s)) for i, s in zip(idxs, scores)])

    results = {q["id"]: dict(pairs) for q, pairs in zip(queries, per_query)}
    ndcg = evaluate(qrels, results, k_values=top_k_eval)
    observed = ndcg["ndcg_cut.10"]

    assert observed == pytest.approx(OBSERVED_AT_BASELINE, abs=0.005), (
        f"nDCG@10={observed:.5f} drifted from the recorded baseline "
        f"{OBSERVED_AT_BASELINE:.5f} (tol 0.005). Refactor likely introduced "
        "a tokenizer or qrels change."
    )
