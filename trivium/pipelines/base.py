"""BenchmarkPipeline ABC: the contract every benchmark orchestrator implements.

A BenchmarkPipeline composes one or more Retrievers (and optionally
one FusionStrategy and one Reranker) into a single benchmark mode
that the registry-driven scripts/run_benchmark.py can invoke
without if/elif dispatch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from trivium.config.schema import Config
from trivium.domain.document import Document
from trivium.domain.pipeline_result import PipelineResult
from trivium.domain.query import Query
from trivium.domain.qrels import Qrels


class BenchmarkPipeline(ABC):
    """One CSV-row-producing benchmark mode.

    Implementations:
        - trivium.pipelines.bm25.Bm25
        - trivium.pipelines.vector.Vector
        - trivium.pipelines.hybrid_rrf.HybridRrf
        - trivium.pipelines.hybrid_rerank.HybridRerank
        - trivium.pipelines.diskbbq.BbqPipeline
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Mode name in the CSV (`mode` column)."""

    @property
    @abstractmethod
    def encoders(self) -> Sequence[str]:
        """Encoder slugs this pipeline is compatible with."""

    @abstractmethod
    def run(
        self,
        documents: Sequence[Document],
        queries: Sequence[Query],
        qrels: Qrels,
        query_vectors: dict[str, object],
        config: Config,
        encoder_slug: str,
    ) -> list[PipelineResult]:
        """Run the pipeline and emit one row per (hyperparam-combination).

        The `query_vectors` dict is keyed by encoder slug; pipelines
        pick the slug they need.

        The Config-driven mode dispatch in the runner calls this and
        extends the global row list. No elif chains anywhere in
        trivial land.
        """
