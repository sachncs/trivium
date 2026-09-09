---
layout: page
title: Add a pipeline
subtitle: Compose a new benchmark mode from existing retrievers, fusion strategies, and rerankers.
section: How-to
permalink: /docs/guides/add-pipeline/
prev:
  title: Add a reranker
  url: /docs/guides/add-reranker/
next:
  title: Tune BM25
  url: /docs/guides/tune-bm25/
---

A pipeline is one benchmark mode. It owns the orchestration of
one or more retrievers, optionally a fusion strategy, and
optionally a reranker. Pipelines emit `PipelineResult` rows that
the CSV writer flattens into the output file.

The default registry ships with six pipeline modes:

| slug           | retriever(s)             | fusion      | reranker |
|----------------|--------------------------|-------------|----------|
| `bm25`         | Bm25                     | —           | —        |
| `vector`       | Faiss.Flat or Faiss.Ivpq | —           | —        |
| `hybrid_rrf`   | Bm25 + Faiss             | Rrf         | —        |
| `hybrid_rerank`| Bm25 + Faiss             | Rrf         | Cross-encoder |
| `diskbbq`      | Bbq (DiskBBQ algorithm)  | —           | —        |
| `hybrid_bbq`   | Bm25 + Bbq               | Rrf         | —        |

## The contract

```python
class BenchmarkPipeline(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Mode name in the CSV (`mode` column)."""

    @property
    @abstractmethod
    def encoders(self) -> Sequence[str]:
        """Encoder slugs this pipeline is compatible with.
        Use ['*'] for any encoder; ['none'] for non-vector modes.
        """

    @abstractmethod
    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        """Run the pipeline and emit one row per (hyperparam-combination)."""
```

`PipelineInput` bundles everything the pipeline needs:

```python
@dataclass
class PipelineInput:
    documents: Sequence[Document]
    queries: Sequence[Query]
    qrels: Qrels
    encoder_slug: str
    query_vectors: np.ndarray | None = None
    corpus_vectors: np.ndarray | None = None
    reranker: Reranker | None = None
```

## Implement a new pipeline

Suppose you want to add a `hybrid_colbert` mode that fuses
BM25 + a ColBERT retriever via RRF, with no rerank.

```python
# trivium/pipelines/hybrid_colbert.py

from collections.abc import Sequence
from trivium.config.schema import Config
from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyProbe
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.fusion.rrf import Rrf
from trivium.pipelines.base import BenchmarkPipeline, PipelineInput
from trivium.pipelines.registry import register_pipeline
from trivium.retrieval.bm25 import Bm25
# from trivium.retrieval.colbert import ColBert  # your retriever
import numpy as np


@register_pipeline("hybrid_colbert")
class HybridColbert(BenchmarkPipeline):
    """BM25 + ColBERT fused via RRF."""

    @property
    def name(self) -> str:
        return "hybrid_colbert"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]:
        if inp.query_vectors is None or inp.corpus_vectors is None:
            raise ValueError("HybridColbert requires query and corpus vectors")

        bm25 = Bm25(k1=config.bm25.k1, b=config.bm25.b,
                    method=config.bm25.method)
        bm25.add_documents(list(inp.documents))

        colbert = ColBert()
        colbert.add_documents(list(inp.documents),
                              vectors=inp.corpus_vectors)

        pool = config.hybrid.candidate_pool
        query_texts = np.array([q.text for q in inp.queries])
        bm25_results = bm25.search(query_texts, k=pool)
        colbert_results = colbert.search(inp.query_vectors, k=pool)

        rows = []
        for rrf_k in config.hybrid.rrf_k_sweep:
            fusion = Rrf(k=rrf_k)
            fused = [fusion.fuse([bm25_results[i], colbert_results[i]],
                                 top_k=pool)
                     for i in range(len(inp.queries))]
            per_query = [[(h.doc_id, h.score) for h in r] for r in fused]
            eval_pairs = hits_to_results(per_query, list(inp.queries))
            metrics = Evaluator(k_values=config.benchmark.top_k_eval) \
                .evaluate(inp.qrels, eval_pairs)

            rows.append(PipelineResult(
                name=self.name,
                scale=len(inp.documents),
                encoder=inp.encoder_slug,
                reranker="none",
                metrics=metrics,
                latency=LatencyProbe(fn=lambda: None, n=1, warmup=0).run(),
                extras={"rrf_k": rrf_k},
            ))
        return rows
```

After this runs, `--modes hybrid_colbert` works on the CLI and
the runner emits one CSV row per `(rrf_k, scale, encoder)`
combination.

## Use the pipeline in the runner

The runner discovers registered pipelines automatically. Pass the
slug on the CLI:

```bash
trivium-benchmark \
  --modes hybrid_colbert \
  --encoders minilm-l6 \
  --scales 5000 \
  --output results/hybrid-colbert.csv
```

You do NOT need to add the pipeline to
`trivium/configs/default.yaml` — the registry is the source of
truth.

## Latency probes

`LatencyProbe` is a small utility for measuring per-query
latency under realistic warmup:

```python
from trivium.evaluation.latency import LatencyProbe

def step_fn(idx: int) -> None:
    """Run one query end-to-end."""
    bm25.search(query_texts[idx], k=pool)
    dense.search(qv[idx].reshape(1, -1), k=pool)
    fusion.fuse(...)

probe = LatencyProbe(
    fn=step_fn,
    n=100,            # number of samples
    warmup=20,        # discard first 20
    rotate=list(range(100)),
)
stats = probe.run()    # LatencyStats(mean, std, p50, p95, ...)
```

The probe uses `time.perf_counter_ns()` and reports
`mean`, `std`, `p50_ms`, `p95_ms`, `min_ms`, `max_ms`, and
`n_samples`. These fields land in the CSV `lat_*` columns.

## Use the helpers

`trivium.pipelines.helpers` ships two functions that every
hybrid pipeline uses:

```python
from trivium.pipelines.helpers import bm25_query_step, rerank_query_step

def probe_step(idx: int, _fuse_obj=bound_fusion):
    s = str(query_texts[idx % len(query_texts)])
    v = qv[idx % len(qv)]
    r1 = bm25_query_step(bm25, s, k=pool)
    r2 = bm25_query_step(dense, v, k=pool)  # or vector_query_step for float32
    return _fuse_obj.fuse([r1, r2], top_k=pool)
```

`bm25_query_step` accepts either a string query or a numpy
array; `vector_query_step` reshapes a single query vector to
`(1, dim)`.

## Test the new pipeline

Add a test that exercises the pipeline with a stub retriever:

```python
# tests/pipelines/test_hybrid_colbert.py

from trivium.config.schema import Config
from trivium.domain.document import Document
from trivium.domain.query import Query
from trivium.domain.qrels import Qrels
from trivium.pipelines.base import PipelineInput
from trivium.pipelines.hybrid_colbert import HybridColbert


def test_hybrid_colbert_runs():
    docs = [Document(doc_id=f"d{i}", title=f"t{i}", text=f"body {i}") for i in range(10)]
    queries = [Query(query_id=f"q{i}", text=f"query {i}") for i in range(3)]
    qrels = Qrels.from_rows([])
    inp = PipelineInput(documents=docs, queries=queries, qrels=qrels,
                        encoder_slug="minilm-l6",
                        query_vectors=np.zeros((3, 8), dtype=np.float32),
                        corpus_vectors=np.zeros((10, 8), dtype=np.float32))
    cfg = Config()
    pipe = HybridColbert()
    rows = pipe.run(inp, cfg)
    assert len(rows) > 0
    assert all(r.name == "hybrid_colbert" for r in rows)
```

## Avoid the common pitfalls

- **Always return at least one row.** Returning `[]` silently
  drops the pipeline from the CSV.
- **Bound closures over loop variables.** Use the
  `default-arg` pattern (`def step(i, _fuse=bound_fuse): ...`) so
  each iteration captures its own fusion strategy.
- **Test with `pytest -m "not golden"` first.** The BM25
  pre-flight suite requires the corpus cache; the rest of the
  suite runs without it.

## Next steps

- Read [Tune BM25](../../guides/tune-bm25/) for the existing
  pipeline hyperparameters.
- Read [Tune the vector index](../../guides/tune-vector/) for the
  IVFPQ + RFlat knobs.
