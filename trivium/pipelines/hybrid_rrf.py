"""HybridRrf pipeline: BM25 + vector fused via RRF."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.config.schema import Config
from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyProbe
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.fusion.rrf import Rrf
from trivium.pipelines.base import BenchmarkPipeline, PipelineInput
from trivium.pipelines.registry import register_pipeline
from trivium.retrieval.bm25 import Bm25 as Bm25Retriever
from trivium.retrieval.faiss import Faiss


@register_pipeline("hybrid_rrf")
class HybridRrf(BenchmarkPipeline):
    """BM25 + dense retrieval fused via Reciprocal Rank Fusion."""

    @property
    def name(self) -> str:
        return "hybrid_rrf"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        if inp.corpus_vectors is None or inp.query_vectors is None:
            raise ValueError("HybridRrf requires query and corpus vectors")

        bm25 = Bm25Retriever(
            k1=config.bm25.k1,
            b=config.bm25.b,
            method=config.bm25.method,
            stopwords=config.bm25.stopwords,
            stemmer=config.bm25.stemmer,
        )
        bm25.add_documents(list(inp.documents))

        qv = np.asarray(inp.query_vectors, dtype=np.float32)
        cv = np.asarray(inp.corpus_vectors, dtype=np.float32)
        scale = len(inp.documents)
        use_exact = scale <= config.vector.min_scale_for_ivfpq
        dense: Faiss.Flat | Faiss.Ivpq = (
            Faiss.Flat() if use_exact else Faiss.Ivpq(config=config, scale=scale)
        )
        dense.add_documents(list(inp.documents), vectors=cv)

        nprobe = config.vector.nprobe_sweep[-1]
        dense.set_search_params(nprobe=nprobe)

        pool = config.hybrid.candidate_pool
        query_texts = np.array([q.text for q in inp.queries])
        bm25_results = bm25.search(query_texts, k=pool)
        dense_results = dense.search(qv, k=pool)

        rows: list[PipelineResult] = []
        for rrf_k in config.hybrid.rrf_k_sweep:
            for wbm, wve in config.hybrid.rrf_weight_sweep:
                fusion = Rrf(k=rrf_k, weights=[wbm, wve])
                fused_results = [
                    fusion.fuse([bm25_results[i], dense_results[i]], top_k=pool)
                    for i in range(len(inp.queries))
                ]
                per_query = [[(h.doc_id, h.score) for h in r] for r in fused_results]
                eval_pairs = hits_to_results(per_query, list(inp.queries))
                metrics = Evaluator(k_values=config.benchmark.top_k_eval).evaluate(
                    inp.qrels, eval_pairs
                )

                rotate_q = [
                    str(query_texts[i % len(query_texts)]) for i in range(min(len(inp.queries), 30))
                ]
                bound_fusion = fusion  # default-arg binding fixes the B023 loop-var warning

                def probe_step(i: int, _fuse_obj=bound_fusion):
                    s = str(query_texts[i % len(query_texts)])
                    v = qv[i % len(qv)]
                    r1 = bm25.search(s, k=pool)
                    r2 = dense.search(v.reshape(1, -1).astype(np.float32), k=pool)
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
                        extras={"rrf_k": rrf_k, "w_bm25": wbm, "w_vec": wve, "nprobe": nprobe},
                    )
                )
        return rows
