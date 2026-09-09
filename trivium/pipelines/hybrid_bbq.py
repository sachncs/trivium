"""HybridBbq pipeline: BM25 + DiskBBQ fused via RRF."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.config.schema import Config
from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyProbe
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.fusion.rrf import Rrf
from trivium.pipelines.base import BenchmarkPipeline, PipelineInput
from trivium.pipelines.diskbbq import _index_dir
from trivium.pipelines.registry import register_pipeline
from trivium.retrieval.bm25 import Bm25 as Bm25Retriever
from trivium.retrieval.diskbbq import Bbq


@register_pipeline("hybrid_bbq")
class HybridBbq(BenchmarkPipeline):
    """BM25 + DiskBBQ fused via Reciprocal Rank Fusion.

    Mirrors `hybrid_rrf.py` but substitutes the Bbq retriever for the
    FAISS IVFPQ retriever. RAM search mode is used (faster; the disk
    I/O story is measured separately in the `diskbbq` pipeline).
    """

    @property
    def name(self) -> str:
        return "hybrid_bbq"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        if inp.corpus_vectors is None or inp.query_vectors is None:
            raise ValueError("HybridBbq requires query and corpus vectors")

        cv = np.asarray(inp.corpus_vectors, dtype=np.float32)
        qv = np.asarray(inp.query_vectors, dtype=np.float32)
        scale = len(inp.documents)
        dim = int(cv.shape[1])

        bm25 = Bm25Retriever(
            k1=config.bm25.k1,
            b=config.bm25.b,
            method=config.bm25.method,
            stopwords=config.bm25.stopwords,
            stemmer=config.bm25.stemmer,
        )
        bm25.add_documents(list(inp.documents))

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

        pool = config.hybrid.candidate_pool
        query_texts = np.array([q.text for q in inp.queries])
        bm25_results = bm25.search(query_texts, k=pool)
        bbq_results = bbq.search(qv, k=pool, mode="ram")

        nprobe = config.vector.nprobe_sweep[-1]
        bbq.set_search_params(nprobe=nprobe)
        bbq_results = bbq.search(qv, k=pool, mode="ram")

        rows: list[PipelineResult] = []
        for rrf_k in config.hybrid.rrf_k_sweep:
            for wbm, wve in config.hybrid.rrf_weight_sweep:
                fusion = Rrf(k=rrf_k, weights=[wbm, wve])
                fused_results = [
                    fusion.fuse([bm25_results[i], bbq_results[i]], top_k=pool)
                    for i in range(len(inp.queries))
                ]
                per_query = [[(h.doc_id, h.score) for h in r] for r in fused_results]
                eval_pairs = hits_to_results(per_query, list(inp.queries))
                metrics = Evaluator(k_values=config.benchmark.top_k_eval).evaluate(inp.qrels, eval_pairs)

                rotate_q = [str(query_texts[i % len(query_texts)]) for i in range(min(len(inp.queries), 30))]
                bound_fusion = fusion

                def probe_step(i: int, _fuse_obj=bound_fusion, _mode: str = "ram") -> None:
                    s = str(query_texts[i % len(query_texts)])
                    v = qv[i % len(qv)]
                    r1 = bm25.search(s, k=pool)
                    r2 = bbq.search(v.reshape(1, -1).astype(np.float32), k=pool, mode=_mode)
                    _fuse_obj.fuse([r1, r2], top_k=pool)

                probe = LatencyProbe(
                    fn=probe_step,
                    n=len(rotate_q),
                    warmup=min(config.benchmark.warmup, len(rotate_q)),
                    rotate=list(range(len(rotate_q))),
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
                            "rrf_k": rrf_k,
                            "w_bm25": wbm,
                            "w_vec": wve,
                            "nprobe": nprobe,
                            "search_mode": "ram",
                            "index_type": "DiskBBQ",
                        },
                    )
                )
        return rows