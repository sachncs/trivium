"""BbqPipeline: orchestrates the disk-resident Bbq retriever."""

from __future__ import annotations

from collections.abc import Sequence

from trivium.config.schema import Config
from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyStats
from trivium.evaluation.metrics import EvaluationMetrics
from trivium.pipelines.base import BenchmarkPipeline, PipelineInput


class BbqPipeline(BenchmarkPipeline):
    """Wraps the disk-resident Bbq retriever for orchestrator-level bookkeeping."""

    @property
    def name(self) -> str:
        return "diskbbq"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        return [
            PipelineResult(
                name=self.name,
                scale=len(inp.documents),
                encoder=inp.encoder_slug,
                reranker="none",
                metrics=EvaluationMetrics(),
                latency=LatencyStats(
                    p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, p999_ms=0.0, mean_ms=0.0, n=0
                ),
                extras={"build_status": "not_implemented_yet"},
            )
        ]
