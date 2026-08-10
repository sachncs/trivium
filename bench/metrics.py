"""pytrec_eval wrapper + latency statistics. BEIR semantics: macro avg, log2, ignore_identical_ids=True."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pytrec_eval


def evaluate(qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k_values=(1, 3, 5, 10, 100, 1000)) -> dict[str, float]:
    """Returns {measure: float} averaged across queries. pytrec_eval macro semantics.

    pytrec_eval returns measure keys with underscores (`ndcg_cut_10`), not dots.
    We accept both names in input and normalize output keys to dot form.
    """
    def to_param(k: str) -> str:
        return k.replace(".", "_")

    def from_param(k: str) -> str:
        return k.replace("_", ".", 1) if "_" in k else k

    requested = {
        "map",
        "ndcg",
        "recip_rank",
    } | (
        {f"ndcg_cut.{k}" for k in k_values} |
        {f"recall.{k}" for k in k_values} |
        {f"P.{k}" for k in k_values if k >= 5}
    )
    ev = pytrec_eval.RelevanceEvaluator(qrels, {to_param(m) for m in requested})
    per_q = ev.evaluate(results)
    if not per_q:
        return {m: 0.0 for m in requested}
    out = {}
    for m in sorted(requested):
        key = to_param(m)
        vals = [v.get(key) for v in per_q.values() if v.get(key) is not None]
        if not vals:
            continue
        out[m] = round(sum(vals) / len(per_q), 5)
    return out


def latency_stats(fn: Callable[[], None], n: int, warmup: int = 20) -> dict[str, float]:
    """Run fn() n times, report p50/p95/p99/p99.9 in ms. perf_counter_ns for nanosecond int."""
    for _ in range(warmup):
        fn()
    samples = np.empty(n, dtype=np.int64)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples[i] = time.perf_counter_ns() - t0
    return {
        "p50_ms": float(np.percentile(samples, 50) / 1e6),
        "p95_ms": float(np.percentile(samples, 95) / 1e6),
        "p99_ms": float(np.percentile(samples, 99) / 1e6),
        "p999_ms": float(np.percentile(samples, 99.9) / 1e6),
        "mean_ms": float(samples.mean() / 1e6),
        "n": n,
    }


def make_qrels(qrels_rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in qrels_rows:
        out.setdefault(r["qid"], {})[r["did"]] = r["rel"]
    return out


def make_results(id_score_pairs: list[tuple[str, float]]) -> dict[str, float]:
    return {str(did): float(s) for did, s in id_score_pairs}
