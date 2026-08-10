"""Benchmark runner. Four modes x N scales. Per-query single-stream latency."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import yaml

from bench.ground_truth import exact_topk
from bench.metrics import evaluate, latency_stats, make_qrels
from search.bm25_index import BM25Index
from search.hybrid import rrf
from search.rerank import CrossEncoderReranker
from search.vector_index import build_index, set_search_params

CACHE = Path(__file__).parent.parent / "data" / "cache"


def load_corpus(scale: int) -> tuple[list[dict], np.ndarray]:
    """Load the nested 1M corpus and the prefix slice for this scale.

    At scale=5000, returns the full scifact seed (the 5K equivalence check requires
    the entire 5183 docs, not the prefix slice).
    """
    if scale == 5000:
        with open(CACHE / "scifact_seed.jsonl") as f:
            docs = [json.loads(line) for line in f]
    else:
        with open(CACHE / "corpus.jsonl") as f:
            docs = [json.loads(line) for line in f]
        docs = docs[:scale]
    npz = np.load(CACHE / "vectors.npz", allow_pickle=True)
    ids = npz["ids"]
    vectors = npz["vectors"]
    id_to_idx = {did: i for i, did in enumerate(ids)}
    idxs = np.array([id_to_idx[d["id"]] for d in docs], dtype=np.int64)
    return docs, vectors[idxs]


def load_queries() -> tuple[list[dict], dict[str, dict[str, int]]]:
    queries = [json.loads(l) for l in open(CACHE / "queries.jsonl")]
    qrels = make_qrels([json.loads(l) for l in open(CACHE / "qrels.jsonl")])
    return queries, qrels


def embed_queries(queries: list[dict], model_name: str, batch_size: int, max_seq_length: int, normalize: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
    vecs = model.encode(
        [q["text"] for q in queries],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(vecs, dtype=np.float32)


def gather_repro() -> dict:
    """Reproducibility manifest: pinned per row."""
    m = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for mod, name in [("faiss", "faiss"), ("sentence_transformers", "sentence_transformers"),
                       ("torch", "torch"), ("bm25s", "bm25s")]:
        try:
            m[name] = __import__(mod).__version__
        except Exception:
            pass
    try:
        m["faiss_omp_threads"] = int(faiss.omp_get_max_threads())
    except Exception:
        pass
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=Path(__file__).parent.parent)
        if r.returncode == 0:
            m["git_sha"] = r.stdout.strip()
    except Exception:
        pass
    return m


def _hit_lists_to_results(queries: list[dict], per_query: list[list[tuple[str, float]]]) -> dict[str, dict[str, float]]:
    return {q["id"]: dict(pairs) for q, pairs in zip(queries, per_query)}


def run_bm25(docs, queries, qrels, cfg, repro, scale):
    bm25 = BM25Index(k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"], method=cfg["bm25"]["method"])
    bm25.build([d["title"] + "\n" + d["text"] for d in docs], show_progress=False)
    k = cfg["bm25"]["candidate_pool"]
    per_query = []
    for q in queries:
        idxs, scores = bm25.query(q["text"], k=k)
        per_query.append([(docs[i]["id"], float(s)) for i, s in zip(idxs, scores)])
    results = _hit_lists_to_results(queries, per_query)
    ndcg = evaluate(qrels, results, k_values=cfg["benchmark"]["top_k_eval"])
    lat = latency_stats(lambda: bm25.query(queries[0]["text"], k=k), n=len(queries), warmup=cfg["benchmark"]["warmup"])
    return {
        "mode": "bm25", "scale": scale, "n_docs": len(docs),
        "ndcg_cut.10": ndcg["ndcg_cut.10"], "recall.10": ndcg["recall.10"], "recall.100": ndcg["recall.100"],
        "lat_p50_ms": lat["p50_ms"], "lat_p95_ms": lat["p95_ms"], "lat_p99_ms": lat["p99_ms"],
        "lat_mean_ms": lat["mean_ms"], "lat_n": lat["n"],
        **repro,
    }


def run_vector(docs, queries, query_vecs, qrels, cfg, repro, scale):
    nlist = cfg["vector"]["nlist"][cfg["scales"].index(scale)]
    m = cfg["vector"]["m"]
    nbits = cfg["vector"]["nbits"]
    use_opq = cfg["vector"]["use_opq"]
    use_rflat = cfg["vector"]["use_rflat"]
    k = cfg["hybrid"]["candidate_pool"]
    nprobe = cfg["vector"]["nprobe"]

    corpus_vecs = _corpus_vectors(docs)

    # At very small scale, OPQ's internal k-means cannot cluster enough points.
    # Use IndexFlat for the 5K scale (exact dense ceiling) — honest baseline.
    use_exact = scale <= 5000

    t0 = time.time()
    if use_exact:
        index = faiss.IndexFlatIP(corpus_vecs.shape[1])
        index.add(corpus_vecs)
        build_s = time.time() - t0
        scores, ids = index.search(query_vecs, k)
        per_query = []
        for i_q in range(len(queries)):
            per_query.append([(docs[int(ids[i_q, j])]["id"], float(scores[i_q, j]))
                             for j in range(k) if ids[i_q, j] != -1])
        results = _hit_lists_to_results(queries, per_query)
        ndcg = evaluate(qrels, results, k_values=cfg["benchmark"]["top_k_eval"])
        lat = latency_stats(
            lambda: index.search(query_vecs[0].reshape(1, -1).astype(np.float32), k=k),
            n=len(query_vecs), warmup=cfg["benchmark"]["warmup"],
        )
        return [{
            "mode": "vector", "scale": scale, "n_docs": len(docs),
            "nlist": 0, "nprobe": 0, "m": 0, "nbits": 0,
            "index_type": "IndexFlatIP",
            "build_s": round(build_s, 2),
            "ndcg_cut.10": ndcg["ndcg_cut.10"], "recall.10": ndcg["recall.10"], "recall.100": ndcg["recall.100"],
            "lat_p50_ms": lat["p50_ms"], "lat_p95_ms": lat["p95_ms"], "lat_p99_ms": lat["p99_ms"],
            "lat_mean_ms": lat["mean_ms"], "lat_n": lat["n"],
            **repro,
        }]

    index = build_index(corpus_vecs, nlist=nlist, m=m, nbits=nbits, use_opq=use_opq, use_rflat=use_rflat)
    build_s = time.time() - t0

    rows = []
    for np_v in nprobe:
        set_search_params(index, nprobe=np_v, k_factor=cfg["vector"]["k_factor"])
        per_query = []
        for qv in query_vecs:
            scores, ids = index.search(qv.reshape(1, -1).astype(np.float32), k=k)
            per_query.append([(docs[i]["id"], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1])
        results = _hit_lists_to_results(queries, per_query)
        ndcg = evaluate(qrels, results, k_values=cfg["benchmark"]["top_k_eval"])
        lat = latency_stats(
            lambda: index.search(query_vecs[0].reshape(1, -1).astype(np.float32), k=k),
            n=len(query_vecs), warmup=cfg["benchmark"]["warmup"],
        )
        rows.append({
            "mode": "vector", "scale": scale, "n_docs": len(docs),
            "nlist": nlist, "nprobe": np_v, "m": m, "nbits": nbits,
            "index_type": "OPQ-IVFPQ-RFlat",
            "build_s": round(build_s, 2),
            "ndcg_cut.10": ndcg["ndcg_cut.10"], "recall.10": ndcg["recall.10"], "recall.100": ndcg["recall.100"],
            "lat_p50_ms": lat["p50_ms"], "lat_p95_ms": lat["p95_ms"], "lat_p99_ms": lat["p99_ms"],
            "lat_mean_ms": lat["mean_ms"], "lat_n": lat["n"],
            **repro,
        })
    return rows


def run_hybrid_rrf(docs, queries, query_vecs, qrels, cfg, repro, scale):
    """BM25 + vector fused via RRF. Sweep rrf_k and weights."""
    bm25 = BM25Index(k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"], method=cfg["bm25"]["method"])
    bm25.build([d["title"] + "\n" + d["text"] for d in docs], show_progress=False)
    nlist = cfg["vector"]["nlist"][cfg["scales"].index(scale)]
    m = cfg["vector"]["m"]
    nbits = cfg["vector"]["nbits"]
    k = cfg["hybrid"]["candidate_pool"]

    corpus_vecs = _corpus_vectors(docs)
    use_exact = scale <= 5000
    if use_exact:
        index = faiss.IndexFlatIP(corpus_vecs.shape[1])
        index.add(corpus_vecs)
    else:
        nprobe = cfg["vector"]["nprobe"][-1]
        index = build_index(corpus_vecs, nlist=nlist, m=m, nbits=nbits,
                            use_opq=cfg["vector"]["use_opq"], use_rflat=cfg["vector"]["use_rflat"])
        set_search_params(index, nprobe=nprobe, k_factor=cfg["vector"]["k_factor"])

    bm25_per_q = []
    for q in queries:
        idxs, scores = bm25.query(q["text"], k=k)
        bm25_per_q.append([(docs[i]["id"], float(s)) for i, s in zip(idxs, scores)])

    vec_per_q = []
    for qv in query_vecs:
        scores, ids = index.search(qv.reshape(1, -1).astype(np.float32), k=k)
        vec_per_q.append([(docs[i]["id"], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1])

    rows = []
    for rrf_k in cfg["hybrid"]["rrf_k_sweep"]:
        for wbm, wve in cfg["hybrid"]["rrf_weight_sweep"]:
            per_query = [rrf(bq, vq, k=rrf_k, weights=(wbm, wve))[:k] for bq, vq in zip(bm25_per_q, vec_per_q)]
            results = _hit_lists_to_results(queries, per_query)
            ndcg = evaluate(qrels, results, k_values=cfg["benchmark"]["top_k_eval"])
            lat = latency_stats(
                lambda: _do_hybrid_query(bm25, index, docs, queries[0], query_vecs[0], k, rrf_k, (wbm, wve)),
                n=len(queries), warmup=cfg["benchmark"]["warmup"],
            )
            rows.append({
                "mode": "hybrid_rrf", "scale": scale, "n_docs": len(docs),
                "rrf_k": rrf_k, "w_bm25": wbm, "w_vec": wve,
                "ndcg_cut.10": ndcg["ndcg_cut.10"], "recall.10": ndcg["recall.10"], "recall.100": ndcg["recall.100"],
                "lat_p50_ms": lat["p50_ms"], "lat_p95_ms": lat["p95_ms"], "lat_p99_ms": lat["p99_ms"],
                "lat_mean_ms": lat["mean_ms"], "lat_n": lat["n"],
                **repro,
            })
    return rows


def run_hybrid_rerank(docs, queries, query_vecs, qrels, cfg, repro, scale):
    """BM25 + vector fused via RRF, then MiniLM-L6 cross-encoder rerank on top-K."""
    bm25 = BM25Index(k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"], method=cfg["bm25"]["method"])
    bm25.build([d["title"] + "\n" + d["text"] for d in docs], show_progress=False)
    nlist = cfg["vector"]["nlist"][cfg["scales"].index(scale)]
    m = cfg["vector"]["m"]
    nbits = cfg["vector"]["nbits"]
    fusion_pool = cfg["hybrid"]["candidate_pool"]
    rerank_pool = cfg["rerank"]["candidate_pool"]

    corpus_vecs = _corpus_vectors(docs)
    use_exact = scale <= 5000
    if use_exact:
        index = faiss.IndexFlatIP(corpus_vecs.shape[1])
        index.add(corpus_vecs)
    else:
        nprobe = cfg["vector"]["nprobe"][-1]
        index = build_index(corpus_vecs, nlist=nlist, m=m, nbits=nbits,
                            use_opq=cfg["vector"]["use_opq"], use_rflat=cfg["vector"]["use_rflat"])
        set_search_params(index, nprobe=nprobe, k_factor=cfg["vector"]["k_factor"])

    reranker = CrossEncoderReranker(
        cfg["rerank"]["model"],
        max_length=cfg["rerank"]["max_length"],
        batch_size=cfg["rerank"]["batch_size"],
    )

    rows = []
    for k_rrf in cfg["hybrid"]["rrf_k_sweep"][:2]:  # cap sweep at 2 vals
        bm25_per_q = []
        for q in queries:
            idxs, scores = bm25.query(q["text"], k=fusion_pool)
            bm25_per_q.append([(docs[i]["id"], float(s)) for i, s in zip(idxs, scores)])
        vec_per_q = []
        for qv in query_vecs:
            scores, ids = index.search(qv.reshape(1, -1).astype(np.float32), k=fusion_pool)
            vec_per_q.append([(docs[i]["id"], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1])

        per_query = []
        id_to_doc = {d["id"]: d for d in docs}
        for q, bq, vq in zip(queries, bm25_per_q, vec_per_q):
            fused = rrf(bq, vq, k=k_rrf)[:rerank_pool]
            cand_ids = [did for did, _ in fused]
            cands = [id_to_doc[did] for did in cand_ids if did in id_to_doc]
            ranked = reranker.rerank(q["text"], cands, top_k=cfg["benchmark"]["top_k_eval"][-1])
            per_query.append([(c["id"], score) for c, score in ranked])
        results = _hit_lists_to_results(queries, per_query)
        ndcg = evaluate(qrels, results, k_values=cfg["benchmark"]["top_k_eval"])
        lat = latency_stats(
            lambda: _do_rerank_query(bm25, index, docs, queries[0], query_vecs[0], fusion_pool, rerank_pool, cfg["hybrid"]["rrf_k"], reranker),
            n=len(queries), warmup=cfg["benchmark"]["warmup"],
        )
        rows.append({
            "mode": "hybrid_rerank", "scale": scale, "n_docs": len(docs),
            "rrf_k": k_rrf, "rerank_pool": rerank_pool,
            "ndcg_cut.10": ndcg["ndcg_cut.10"], "recall.10": ndcg["recall.10"], "recall.100": ndcg["recall.100"],
            "lat_p50_ms": lat["p50_ms"], "lat_p95_ms": lat["p95_ms"], "lat_p99_ms": lat["p99_ms"],
            "lat_mean_ms": lat["mean_ms"], "lat_n": lat["n"],
            **repro,
        })
    return rows


def _corpus_vectors(docs: list[dict]) -> np.ndarray:
    """Stack vectors for the doc slice. Reads from cached vectors.npz via id lookup."""
    npz = np.load(CACHE / "vectors.npz", allow_pickle=True)
    ids = npz["ids"]
    vectors = npz["vectors"]
    id_to_idx = {did: i for i, did in enumerate(ids)}
    idxs = np.array([id_to_idx[d["id"]] for d in docs], dtype=np.int64)
    return vectors[idxs].astype(np.float32)


def _do_hybrid_query(bm25, index, docs, query, qvec, k_fusion, rrf_k, weights):
    idxs, scores = bm25.query(query["text"], k=k_fusion)
    bm25_hits = [(docs[i]["id"], float(s)) for i, s in zip(idxs, scores)]
    scores, ids = index.search(qvec.reshape(1, -1).astype(np.float32), k=k_fusion)
    vec_hits = [(docs[i]["id"], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    return rrf(bm25_hits, vec_hits, k=rrf_k, weights=weights)[:k_fusion]


def _do_rerank_query(bm25, index, docs, query, qvec, k_fusion, k_rerank, rrf_k, reranker):
    idxs, scores = bm25.query(query["text"], k=k_fusion)
    bm25_hits = [(docs[i]["id"], float(s)) for i, s in zip(idxs, scores)]
    scores, ids = index.search(qvec.reshape(1, -1).astype(np.float32), k=k_fusion)
    vec_hits = [(docs[i]["id"], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    fused = rrf(bm25_hits, vec_hits, k=rrf_k)[:k_rerank]
    id_to_doc = {d["id"]: d for d in docs}
    cands = [id_to_doc[did] for did, _ in fused if did in id_to_doc]
    return reranker.rerank(query["text"], cands, top_k=10)
