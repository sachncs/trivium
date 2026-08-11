"""Vector pipeline: pure FAISS mode (Flat below threshold, Ivpq above)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trivium.config.schema import Config
from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyProbe
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.pipelines.base import BenchmarkPipeline, PipelineInput
from trivium.pipelines.registry import register_pipeline
from trivium.retrieval.faiss import Faiss


@register_pipeline("vector")
class Vector(BenchmarkPipeline):
    """Pure-vector pipeline. Selects Flat or Ivpq based on scale."""

    @property
    def name(self) -> str:
        return "vector"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        if inp.query_vectors is None or inp.corpus_vectors is None:
            raise ValueError("Vector pipeline requires query and corpus vectors")

        qv = np.asarray(inp.query_vectors, dtype=np.float32)
        cv = np.asarray(inp.corpus_vectors, dtype=np.float32)
        scale = len(inp.documents)
        use_exact = scale <= config.vector.min_scale_for_ivfpq

        if use_exact:
            retriever: Faiss.Flat | Faiss.Ivpq = Faiss.Flat()
            index_type = "IndexFlatIP"
            retriever.add_documents(list(inp.documents), vectors=cv)
            nprobe_sweep = [0]
        else:
            retriever = Faiss.Ivpq(config=config, scale=scale)
            retriever.add_documents(list(inp.documents), vectors=cv)
            index_type = "OPQ-IVFPQ-RFlat"
            nprobe_sweep = list(config.vector.nprobe_sweep)

        k = config.hybrid.candidate_pool
        rows: list[PipelineResult] = []
        for nprobe in nprobe_sweep:
            retriever.set_search_params(nprobe=nprobe if nprobe else 1)
            results = retriever.search(qv, k=k)
            per_query = [[(h.doc_id, h.score) for h in r] for r in results]
            eval_pairs = hits_to_results(per_query, list(inp.queries))
            metrics = Evaluator(k_values=config.benchmark.top_k_eval).evaluate(
                inp.qrels, eval_pairs
            )

            rotate = [qv[i % len(qv)] for i in range(min(len(qv), 100))]
            probe = LatencyProbe(
                fn=lambda v: retriever.search(v.reshape(1, -1).astype(np.float32), k=k),
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
                        "index_type": index_type,
                        "nprobe": nprobe,
                        "nlist": getattr(retriever, "nlist", 0),
                        "m": getattr(retriever, "_m", 0),
                        "nbits": getattr(retriever, "_nbits", 0),
                    },
                )
            )
        return rows
