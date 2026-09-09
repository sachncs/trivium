"""HybridRerank pipeline: BM25 + vector + cross-encoder rerank."""

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


@register_pipeline("hybrid_rerank")
class HybridRerank(BenchmarkPipeline):
    """BM25 + dense fused via RRF, top-K candidates reranked by a cross-encoder."""

    @property
    def name(self) -> str:
        return "hybrid_rerank"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        if inp.corpus_vectors is None or inp.query_vectors is None or inp.reranker is None:
            raise ValueError("HybridRerank requires query/corpus vectors and a reranker")

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

        fusion_pool = config.hybrid.candidate_pool
        rerank_pool = config.rerank.candidate_pool
        fusion = Rrf(k=config.hybrid.rrf_k)
        nprobe = config.vector.nprobe_sweep[-1]
        dense.set_search_params(nprobe=nprobe)

        query_texts = np.array([q.text for q in inp.queries])
        id_to_doc = {d.doc_id: d for d in inp.documents}
        bm25_results = bm25.search(query_texts, k=fusion_pool)
        dense_results = dense.search(qv, k=fusion_pool)

        fused_per_query = [
            fusion.fuse([bm25_results[i], dense_results[i]], top_k=fusion_pool)
            for i in range(len(inp.queries))
        ]

        per_query_ranked: list[list[tuple[str, float]]] = []
        for q, fused in zip(inp.queries, fused_per_query, strict=False):
            cands = [id_to_doc[h.doc_id] for h in fused if h.doc_id in id_to_doc][:rerank_pool]
            ranked = inp.reranker.rerank(q.text, cands, top_k=config.benchmark.top_k_eval[-1])
            per_query_ranked.append([(c.doc_id, s) for c, s in ranked])

        eval_pairs = hits_to_results(per_query_ranked, list(inp.queries))
        metrics = Evaluator(k_values=config.benchmark.top_k_eval).evaluate(inp.qrels, eval_pairs)

        def probe_step(idx: int):
            s_idx = idx % len(query_texts)
            v_idx = idx % len(qv)
            r1 = bm25.search(str(query_texts[s_idx]), k=fusion_pool)
            r2 = dense.search(qv[v_idx].reshape(1, -1).astype(np.float32), k=fusion_pool)
            fused = fusion.fuse([r1, r2], top_k=fusion_pool)
            cands = [id_to_doc[h.doc_id] for h in fused if h.doc_id in id_to_doc][:rerank_pool]
            inp.reranker.rerank(
                str(query_texts[s_idx]), cands, top_k=config.benchmark.top_k_eval[-1]
            )

        n_iter = min(len(inp.queries), 50)
        probe = LatencyProbe(
            fn=probe_step,
            n=n_iter,
            warmup=config.benchmark.warmup,
            rotate=list(range(n_iter)),
        )
        lat = probe.run()

        return [
            PipelineResult(
                name=self.name,
                scale=scale,
                encoder=inp.encoder_slug,
                reranker=getattr(inp.reranker, "slug", "unknown"),
                metrics=metrics,
                latency=lat,
                extras={
                    "rrf_k": config.hybrid.rrf_k,
                    "rerank_pool": rerank_pool,
                    "nprobe": nprobe,
                },
            )
        ]
