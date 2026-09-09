"""BenchmarkPipeline ABC: the contract every benchmark orchestrator implements.

A BenchmarkPipeline composes one or more Retrievers (and optionally
one FusionStrategy and one Reranker) into a single benchmark mode
that the registry-driven scripts/run_benchmark.py can invoke
without if/elif dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from trivium.config.schema import Config
from trivium.domain.document import Document
from trivium.domain.pipeline_result import PipelineResult
from trivium.domain.qrels import Qrels
from trivium.domain.query import Query


class PipelineInput:
    """The complete input to a pipeline run.

    Bundles everything the pipeline needs so the ABC signature stays
    bounded and we can add new inputs (corpus_vectors, reranker) without
    touching every concrete pipeline.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        queries: Sequence[Query],
        qrels: Qrels,
        encoder_slug: str,
        query_vectors: np.ndarray | None = None,
        corpus_vectors: np.ndarray | None = None,
        reranker=None,
    ) -> None:
        self.documents = documents
        self.queries = queries
        self.qrels = qrels
        self.encoder_slug = encoder_slug
        self.query_vectors = query_vectors
        self.corpus_vectors = corpus_vectors
        self.reranker = reranker


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
        inp: PipelineInput,
        config: Config,
    ) -> list[PipelineResult]:
        """Run the pipeline and emit one row per (hyperparam-combination).

        The Config-driven mode dispatch in the runner calls this and
        extends the global row list. No elif chains anywhere in
        trivial land.
        """
