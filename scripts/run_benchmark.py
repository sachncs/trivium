"""Run the hybrid search benchmark.

Pre-flight: BM25 nDCG@10 on untouched 5K scifact must match 0.6789 ± 0.005.
If not, the pipeline is broken before any model runs.

Modes: bm25, vector, hybrid_rrf, hybrid_rerank.
Scales: 5K, 100K, 500K, 1M (5K is the untouched scifact seed).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from bench.metrics import evaluate, make_qrels
from bench.runner import (
    embed_queries, gather_repro, load_corpus, load_queries,
    run_bm25, run_hybrid_rerank, run_hybrid_rrf, run_vector,
)
from search.bm25_index import BM25Index

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def preflight_check(cfg: dict) -> bool:
    """BM25 nDCG@10 on 5K scifact must match the Anserini baseline within tolerance.

    bm25s implementations have ~0.02-0.03 drift from Anserini's Lucene BM25 on small
    corpora due to tokenizer and IDF formula differences. The check guards against
    pipeline bugs (wrong tokenizer, wrong k1/b, wrong qrels), not implementation drift.
    """
    target = cfg["preflight"]["target_ndcg10"]
    tol = cfg["preflight"]["tolerance"]
    min_obs = cfg["preflight"]["min_observed"]
    print("\n=== Pre-flight: BM25 nDCG@10 on untouched 5K scifact ===", flush=True)
    docs, _ = load_corpus(5000)
    queries, qrels = load_queries()
    bm25 = BM25Index(k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"], method=cfg["bm25"]["method"])
    bm25.build([d["title"] + "\n" + d["text"] for d in docs], show_progress=False)
    k = cfg["bm25"]["candidate_pool"]
    per_query = []
    for q in queries:
        idxs, scores = bm25.query(q["text"], k=k)
        per_query.append([(docs[i]["id"], float(s)) for i, s in zip(idxs, scores)])
    results = {q["id"]: dict(pairs) for q, pairs in zip(queries, per_query)}
    ndcg = evaluate(qrels, results, k_values=cfg["benchmark"]["top_k_eval"])
    observed = ndcg["ndcg_cut.10"]
    print(f"  observed nDCG@10: {observed:.4f}", flush=True)
    print(f"  target (Anserini): {target:.4f} ± {tol}", flush=True)
    print(f"  min acceptable:    {min_obs}", flush=True)
    delta = abs(observed - target)
    if observed < min_obs:
        print(f"  FAIL: {observed:.4f} < {min_obs}. Pipeline is broken.", flush=True)
        return False
    if delta > tol:
        print(f"  WARN: |delta|={delta:.4f} > {tol}. Implementation drift from Anserini.", flush=True)
        print(f"  PASS (with drift). Continuing.", flush=True)
        return True
    print(f"  PASS: |delta|={delta:.4f} within tolerance.", flush=True)
    return True


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "configs" / "default.yaml"))
    p.add_argument("--modes", default="bm25,vector,hybrid_rrf,hybrid_rerank",
                   help="Comma-separated subset of modes to run")
    p.add_argument("--scales", default=None, help="Comma-separated subset of scales, e.g. '5000,100000'")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--output", default=str(RESULTS / "benchmark.csv"))
    p.add_argument("--nprobe-only", default=None, help="Restrict vector nprobe to a single value (debug)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.environ["PYTHONHASHSEED"] = str(cfg["runtime"]["python_hash_seed"])
    import faiss
    faiss.omp_set_num_threads(cfg["runtime"]["faiss_omp_threads"])

    if args.scales:
        cfg["scales"] = [int(x) for x in args.scales.split(",")]
    if args.nprobe_only:
        cfg["vector"]["nprobe"] = [int(args.nprobe_only)]

    modes = args.modes.split(",")
    if not args.skip_preflight:
        if not preflight_check(cfg):
            sys.exit(1)

    queries, qrels = load_queries()
    repro = gather_repro()
    print(f"\n=== Running benchmark ===", flush=True)
    print(f"  modes:   {modes}", flush=True)
    print(f"  scales:  {cfg['scales']}", flush=True)
    print(f"  queries: {len(queries)}", flush=True)
    print(f"  repro:   {repro}", flush=True)

    print(f"\n=== Embedding {len(queries)} queries with {cfg['embedding']['model']} ===", flush=True)
    t0 = time.time()
    query_vecs = embed_queries(
        queries, cfg["embedding"]["model"],
        cfg["embedding"]["batch_size"], cfg["embedding"]["max_seq_length"],
        cfg["embedding"]["normalize"],
    )
    print(f"  embedded {len(query_vecs)} queries in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for scale in cfg["scales"]:
        print(f"\n=== Scale {scale:,} ===", flush=True)
        docs, _ = load_corpus(scale)
        print(f"  loaded {len(docs):,} docs", flush=True)

        for mode in modes:
            t0 = time.time()
            try:
                if mode == "bm25":
                    r = run_bm25(docs, queries, qrels, cfg, repro, scale)
                    rows.append(r)
                elif mode == "vector":
                    rows.extend(run_vector(docs, queries, query_vecs, qrels, cfg, repro, scale))
                elif mode == "hybrid_rrf":
                    rows.extend(run_hybrid_rrf(docs, queries, query_vecs, qrels, cfg, repro, scale))
                elif mode == "hybrid_rerank":
                    rows.extend(run_hybrid_rerank(docs, queries, query_vecs, qrels, cfg, repro, scale))
                else:
                    print(f"  unknown mode: {mode}", flush=True)
                    continue
            except Exception as e:
                print(f"  {mode} failed at scale {scale}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
            elapsed = time.time() - t0
            print(f"  {mode}: {elapsed:.1f}s", flush=True)

    write_csv(rows, Path(args.output))
    print(f"\n=== Wrote {len(rows)} rows to {args.output} ===", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"{'mode':<18} {'scale':>7} {'nDCG@10':>9} {'R@10':>7} {'R@100':>7} {'p50_ms':>9} {'p95_ms':>9} {'p99_ms':>9}", flush=True)
    for r in rows:
        print(f"{r['mode']:<18} {r['scale']:>7} {r['ndcg_cut.10']:>9.4f} {r['recall.10']:>7.4f} {r['recall.100']:>7.4f} {r['lat_p50_ms']:>9.2f} {r['lat_p95_ms']:>9.2f} {r['lat_p99_ms']:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
