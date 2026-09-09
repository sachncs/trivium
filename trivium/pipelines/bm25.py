"""Bm25 pipeline: pure BM25 mode."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.config.schema import Config
from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyProbe
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.pipelines.base import BenchmarkPipeline, PipelineInput
from trivium.pipelines.registry import register_pipeline
from trivium.retrieval.bm25 import Bm25 as Bm25Retriever


@register_pipeline("bm25")
class Bm25(BenchmarkPipeline):
    """Pure-BM25 pipeline. No vector path, no reranker."""

    @property
    def name(self) -> str:
        return "bm25"

    @property
    def encoders(self) -> Sequence[str]:
        return ["none"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        bm25 = Bm25Retriever(
            k1=config.bm25.k1,
            b=config.bm25.b,
            method=config.bm25.method,
            stopwords=config.bm25.stopwords,
            stemmer=config.bm25.stemmer,
        )
        bm25.add_documents(list(inp.documents))
        pool = config.bm25.candidate_pool

        query_texts = np.array([q.text for q in inp.queries])
        raw = bm25.search(query_texts, k=pool)
        # bm25.search returns list[SearchResult]
        per_query = [
            [(h.doc_id, h.score) for h in (r if isinstance(r, list) else [r])[0]] if r else []
            for r in raw
        ]

        eval_pairs = hits_to_results(per_query, list(inp.queries))
        metrics = Evaluator(k_values=config.benchmark.top_k_eval).evaluate(inp.qrels, eval_pairs)

        rotate = [str(s) for s in query_texts]
        probe = LatencyProbe(
            fn=lambda s: bm25.search(s, k=pool),
            n=min(len(query_texts), 300),
            warmup=config.benchmark.warmup,
            rotate=rotate,
        )
        lat = probe.run()

        return [
            PipelineResult(
                name=self.name,
                scale=len(inp.documents),
                encoder="none",
                reranker="none",
                metrics=metrics,
                latency=lat,
                extras={"candidate_pool": pool, "method": config.bm25.method},
            )
        ]
