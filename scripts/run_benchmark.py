#!/usr/bin/env python3
"""Run the trivium hybrid search benchmark.

Registry-driven dispatch (no elif chains). Pipeline modes are
registered in trivium/pipelines/registry.py; encoders in
trivium/embeddings/registry.py; rerankers in trivium/reranking/registry.py.

Pre-flight: BM25 nDCG@10 on the untouched 5K scifact corpus must
reproduce Anserini 0.6789 +/- 0.03, the documented bm25s-vs-Lucene
implementation drift. Below 0.62 = pipeline is broken; fail fast.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from trivium.config.loader import load_config
from trivium.domain.document import Document
from trivium.domain.query import Query
from trivium.domain.qrels import Qrels
from trivium.embeddings.registry import get_embedder
from trivium.pipelines import bm25 as pipelines_bm25  # noqa: F401  registers 'bm25'
from trivium.pipelines import vector as pipelines_vector  # noqa: F401  registers 'vector'
from trivium.pipelines import hybrid_rrf as pipelines_hybrid_rrf  # noqa: F401
from trivium.pipelines import hybrid_rerank as pipelines_hybrid_rerank  # noqa: F401
from trivium.pipelines import diskbbq as pipelines_diskbbq  # noqa: F401
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.io.corpus import CorpusCache
from trivium.pipelines.base import PipelineInput
from trivium.pipelines.registry import PipelineRegistry, get_pipeline
from trivium.reranking.encoder import Encoder
from trivium.reranking.registry import get_reranker
from trivium.reproducibility import ReproducibilityManifest
from trivium.reporting.csv_writer import CsvResultWriter
from trivium.retrieval.bm25 import Bm25

DATA_DIR = Path(__file__).parent.parent / "data" / "cache"


def preflight_check(config) -> bool:
    """BM25 nDCG@10 on 5K scifact must reproduce the Anserini baseline."""
    cache = CorpusCache(DATA_DIR)
    seed_docs = cache.load_scale(5000)
    if not seed_docs:
        print("  no scifact_seed.jsonl; skipping pre-flight")
        return True

    query_rows, qrels_rows = cache.load_queries_and_qrels()
    queries = Query.many(query_rows)
    qrels = Qrels.from_rows(qrels_rows)

    bm25 = Bm25(
        k1=config.bm25.k1,
        b=config.bm25.b,
        method=config.bm25.method,
        stopwords=config.bm25.stopwords,
        stemmer=config.bm25.stemmer,
    )
    bm25.add_documents(seed_docs)
    pool = config.bm25.candidate_pool

    query_texts = np.array([q.text for q in queries])
    results = bm25.search(query_texts, k=pool)
    per_query = [[(h.doc_id, h.score) for h in r] for r in results]
    eval_pairs = hits_to_results(per_query, queries)
    metrics = Evaluator(k_values=config.benchmark.top_k_eval).evaluate(qrels, eval_pairs)
    observed = metrics.ndcg_at_10

    target = config.preflight.target_ndcg10
    tol = config.preflight.tolerance
    min_obs = config.preflight.min_observed

    print(f"\n=== Pre-flight: BM25 nDCG@10 on untouched 5K scifact ===", flush=True)
    print(f"  observed nDCG@10: {observed:.4f}", flush=True)
    print(f"  target (Anserini): {target:.4f} +/- {tol}", flush=True)

    if observed < min_obs:
        print(f"  FAIL: {observed:.4f} < {min_obs}. Pipeline is broken.", flush=True)
        return False
    if abs(observed - target) > tol:
        print(f"  WARN: |delta|={abs(observed - target):.4f} > {tol}. Implementation drift from Anserini.", flush=True)
        return True
    print(f"  PASS: |delta|={abs(observed - target):.4f} within tolerance.", flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "configs" / "default.yaml"))
    p.add_argument("--modes", default="bm25,vector,hybrid_rrf", help="Comma-separated subset of pipeline modes")
    p.add_argument("--scales", default=None, help="Comma-separated subset of scales, e.g. '5000,100000'")
    p.add_argument("--encoders", default=None, help="Comma-separated subset of encoder slugs")
    p.add_argument("--rerankers", default=None, help="Comma-separated subset of reranker slugs")
    p.add_argument("--output", default=str(Path(__file__).parent.parent / "results" / "benchmark.csv"))
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    config = load_config(args.config)

    if args.scales:
        config.scales = [int(x) for x in args.scales.split(",")]
    selected_modes = args.modes.split(",")
    selected_encoders = args.encoders.split(",") if args.encoders else [e.slug for e in config.encoders]
    selected_rerankers = args.rerankers.split(",") if args.rerankers else ["encoder"]

    os.environ["PYTHONHASHSEED"] = str(config.runtime.python_hash_seed)
    os.environ.setdefault("OMP_NUM_THREADS", str(config.runtime.faiss_omp_threads))

    if not args.skip_preflight:
        if not preflight_check(config):
            sys.exit(1)

    cache = CorpusCache(DATA_DIR)
    query_rows, qrels_rows = cache.load_queries_and_qrels()
    queries = Query.many(query_rows)
    qrels = Qrels.from_rows(qrels_rows)

    print(f"\n=== Running benchmark ===", flush=True)
    print(f"  modes:   {selected_modes}", flush=True)
    print(f"  scales:  {config.scales}", flush=True)
    print(f"  encoders:{selected_encoders}", flush=True)
    print(f"  queries: {len(queries)}", flush=True)

    query_vectors: dict[str, np.ndarray] = {}
    enc_by_slug = {e.slug: e for e in config.encoders}
    for slug in selected_encoders:
        if slug not in enc_by_slug:
            continue
        entry = enc_by_slug[slug]
        try:
            embedder = get_embedder(
                slug,
                slug=slug,
                model_id=entry.model_id,
                dimension=entry.dimension,
                batch_size=entry.batch_size,
                max_seq_length=entry.max_seq_length,
                normalize=entry.normalize,
                device=entry.device,
                prompt_prefix_doc=entry.prompt_prefix_doc,
                prompt_prefix_query=entry.prompt_prefix_query,
            )
        except KeyError:
            continue
        try:
            embedder.warmup(["warmup"])
            qv = embedder.encode_queries([q.text for q in queries])
            query_vectors[slug] = np.asarray(qv, dtype=np.float32)
            print(f"  encoded {len(qv)} queries with {slug}", flush=True)
        except Exception as exc:
            print(f"  encoder {slug} failed: {exc}", flush=True)
            continue

    all_rows = []
    for scale in config.scales:
        documents = cache.load_scale(scale)
        try:
            corpus_vectors, _ = cache.load_vectors()
        except FileNotFoundError:
            corpus_vectors = None
        print(f"\n=== Scale {scale:,} ({len(documents)} docs) ===", flush=True)

        for mode_name in selected_modes:
            try:
                pipeline = get_pipeline(mode_name)
            except KeyError:
                print(f"  unknown mode: {mode_name}", flush=True)
                continue

            if pipeline.encoders == ["none"]:
                inp = PipelineInput(
                    documents=documents,
                    queries=queries,
                    qrels=qrels,
                    encoder_slug="none",
                )
                rows = pipeline.run(inp, config)
                all_rows.extend(rows)
            else:
                for enc_slug in selected_encoders:
                    qv = query_vectors.get(enc_slug)
                    if qv is None:
                        continue
                    inp = PipelineInput(
                        documents=documents,
                        queries=queries,
                        qrels=qrels,
                        encoder_slug=enc_slug,
                        query_vectors=qv,
                        corpus_vectors=corpus_vectors,
                    )
                    rows = pipeline.run(inp, config)
                    all_rows.extend(rows)
                    break  # one encoder per (pipeline, scale)

    repro = ReproducibilityManifest.gather()
    repro_dict = repro.to_dict()
    for r in all_rows:
        for k, v in repro_dict.items():
            r.extras.setdefault(k, v)
    rows_with_repro = all_rows

    CsvResultWriter.write(rows_with_repro, args.output)
    print(f"\n=== Wrote {len(rows_with_repro)} rows to {args.output} ===", flush=True)


if __name__ == "__main__":
    main()
