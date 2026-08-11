"""BbqPipeline: orchestrates the disk-resident Bbq retriever.

Builds the Bbq index at `data/cache/diskbbq_index/<scale>/` once per
(scale, encoder) pair. Sweeps nprobe over `config.vector.nprobe_sweep`,
emitting one PipelineResult per (nprobe, mode) combination.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from trivium.config.schema import Config
from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyProbe
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.pipelines.base import BenchmarkPipeline, PipelineInput
from trivium.pipelines.registry import register_pipeline
from trivium.retrieval.diskbbq import Bbq


def _index_dir(scale: int, encoder_slug: str) -> Path:
    cache_root = Path(__file__).parent.parent.parent / "data" / "cache"
    return cache_root / "diskbbq_index" / f"{encoder_slug}_{scale}"


@register_pipeline("diskbbq")
class BbqPipeline(BenchmarkPipeline):
    """Pure-DiskBBQ pipeline (no fusion, no rerank)."""

    @property
    def name(self) -> str:
        return "diskbbq"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        if inp.corpus_vectors is None or inp.query_vectors is None:
            raise ValueError("BbqPipeline requires query and corpus vectors")

        cv = np.asarray(inp.corpus_vectors, dtype=np.float32)
        qv = np.asarray(inp.query_vectors, dtype=np.float32)
        scale = len(inp.documents)
        dim = int(cv.shape[1])

        if dim % 48 == 0:
            m = 48
        elif dim % 32 == 0:
            m = 32
        else:
            m = max(8, dim // max(1, dim.bit_length() - 1))
        try:
            nlist = config.vector.nlist_for_scale(scale)
        except KeyError:
            nlist = min(config.vector.nlist_by_scale.values())
        out_dir = _index_dir(scale, inp.encoder_slug)

        bbq = Bbq(out_dir=out_dir, nlist=nlist, m=m, nprobe=config.vector.nprobe_sweep[0], use_rescore=True)
        bbq.add_documents(list(inp.documents), vectors=cv)

        k = config.hybrid.candidate_pool
        rows: list[PipelineResult] = []
        for nprobe in config.vector.nprobe_sweep:
            bbq.set_search_params(nprobe=nprobe)
            for mode in ("ram", "disk"):
                results = bbq.search(qv, k=k, mode=mode)
                per_query = [[(h.doc_id, h.score) for h in r] for r in results]
                eval_pairs = hits_to_results(per_query, list(inp.queries))
                metrics = Evaluator(k_values=config.benchmark.top_k_eval).evaluate(inp.qrels, eval_pairs)

                rotate = [qv[i % len(qv)] for i in range(min(len(qv), 100))]

                def step(v: np.ndarray, _m: str = mode) -> list:
                    return bbq.search(v.reshape(1, -1).astype(np.float32), k=k, mode=_m)

                probe = LatencyProbe(
                    fn=step,
                    n=min(len(qv), 100),
                    warmup=config.benchmark.warmup,
                    rotate=rotate,
                )
                lat = probe.run()

                rows.append(
                    PipelineResult(
                        name=self.name,
                        scale=scale,
                        encoder=inp.encoder_slug,
                        reranker="none",
                        metrics=metrics,
                        latency=lat,
                        extras={
                            "index_type": "DiskBBQ",
                            "nprobe": nprobe,
                            "nlist": bbq.nlist,
                            "m": bbq.m,
                            "nbits": bbq.nbits,
                            "search_mode": mode,
                            "index_bytes": bbq.size_bytes(),
                        },
                    )
                )
        return rows