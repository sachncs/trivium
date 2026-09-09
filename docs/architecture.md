---
layout: page
title: Architecture
subtitle: How the pipeline registry, embedder registry, reranker registry, and retriever ABCs fit together.
section: Understand
permalink: /docs/architecture/
prev:
  title: Getting started
  url: /docs/getting-started/
next:
  title: Concepts
  url: /docs/concepts/
---

trivium is a registry-driven benchmark. Every benchmark mode is a
class decorated with `@register_pipeline`; every embedder is a
class decorated with `@register_embedder`; every reranker is a
class decorated with `@register_reranker`. The runner
(`trivium-benchmark`) loops over the registered slugs and emits one
CSV row per `(mode, encoder, reranker, scale)` combination.

This page walks through the layering from top to bottom.

## The five-layer stack

```
┌─────────────────────────────────────────────────────────────────┐
│  scripts/run_benchmark.py  (CLI: trivium-benchmark)              │
│  - arg parsing                                                  │
│  - pre-flight BM25 check                                        │
│  - corpus cache load                                           │
│  - per-(mode, encoder, reranker, scale) loop                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ instantiate
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  trivium.pipelines  (BenchmarkPipeline ABC + registry)         │
│  Bm25 / Vector / HybridRrf / HybridRerank / BbqPipeline         │
│  / HybridBbq                                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ compose
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  trivium.retrieval  (Retriever ABC + concrete impls)            │
│  Bm25 (bm25s) / Faiss.Flat / Faiss.Ivpq / Bbq (DiskBBQ)         │
│                                                                  │
│  trivium.fusion  (FusionStrategy ABC + concrete impls)          │
│  Rrf / Weighted                                                 │
│                                                                  │
│  trivium.reranking  (Reranker ABC + concrete impls)            │
│  Encoder (cross-encoder ms-marco) / Bge / Monot5               │
└────────────────────────────┬────────────────────────────────────┘
                             │ encode documents and queries
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  trivium.embeddings  (Embedder ABC + concrete impls)            │
│  Sentence (sentence-transformers) / E5 (HF transformers)       │
└─────────────────────────────────────────────────────────────────┘
```

## Registries

Each registry is a module-level dict of `slug -> factory function`,
plus an `EmbedderRegistry` / `RerankerRegistry` / `PipelineRegistry`
façade class. Decorating a factory is the only step required to
make a new component available to the runner.

### Pipeline registry

```python
from trivium.pipelines.registry import register_pipeline

@register_pipeline("hybrid_rrf")
class HybridRrf(BenchmarkPipeline):
    @property
    def name(self) -> str:
        return "hybrid_rrf"

    @property
    def encoders(self) -> Sequence[str]:
        return ["*"]

    def run(self, inp, config):
        ...
```

The runner looks up the class with
`PipelineRegistry.get("hybrid_rrf")` (or `get_pipeline`).
Adding a new mode is one decorator call — there are no `elif`
chains anywhere in the runner.

### Embedder registry

```python
from trivium.embeddings.registry import register_embedder

@register_embedder("bge-large")
def make_bge_large(**kwargs) -> Embedder:
    from trivium.embeddings.sentence import Sentence
    return Sentence(slug="bge-large", model_id="BAAI/bge-large-en-v1.5",
                    dimension=1024, **kwargs)
```

Embedders are factory functions, not classes, so the runner can
defer the heavy `sentence_transformers` import to first use.

### Reranker registry

```python
from trivium.reranking.registry import register_reranker

@register_reranker("bge-large")
def make_bge_large(**kwargs) -> Reranker:
    from trivium.reranking.bge import Bge
    return Bge(slug="bge-large", model_id="BAAI/bge-reranker-large", **kwargs)
```

## Contracts

### `Embedder`

```python
class Embedder(ABC):
    @property
    @abstractmethod
    def slug(self) -> str: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    @abstractmethod
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    @abstractmethod
    def warmup(self, sample_texts: Sequence[str]) -> None: ...
```

`encode_*` returns `(N, dimension)` float32. `warmup` is called once
before any search so the first-query cost is bounded.

### `Retriever`

```python
class Retriever(ABC):
    @property
    @abstractmethod
    def slug(self) -> str: ...

    @abstractmethod
    def add_documents(self, documents: Sequence[Document],
                     vectors: np.ndarray | None = None) -> None: ...

    @abstractmethod
    def search(self, query_vectors: np.ndarray, k: int) -> list[SearchResult]: ...

    @abstractmethod
    def set_search_params(self, **params) -> None: ...
```

`query_vectors` is intentionally typed loosely so the same
retriever accepts both BM25 raw strings and dense float32 vectors.
The retriever interprets the input based on its slug.

### `Reranker`

```python
class Reranker(ABC):
    @property
    @abstractmethod
    def slug(self) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[Document],
                top_k: int) -> SearchResult: ...
```

### `FusionStrategy`

```python
class FusionStrategy(ABC):
    @property
    @abstractmethod
    def slug(self) -> str: ...

    @abstractmethod
    def fuse(self, results: Sequence[SearchResult], top_k: int) -> SearchResult: ...
```

`Rrf` (Cormack et al. 2009) and `Weighted` are the two concrete
implementations.

### `BenchmarkPipeline`

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
        """Run the pipeline and emit one or more rows."""
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

## The pre-flight invariant

Before any benchmark row is emitted, `trivium-benchmark` runs the
BM25 nDCG@10 equivalence check on the untouched 5K scifact corpus.
It must hit `0.6789 ± 0.03` (the Anserini published baseline; ±0.03
is the documented bm25s-vs-Lucene drift). Below `0.62` the runner
exits 1 — fail fast.

This check is implemented as a pytest golden marker. The same check
runs in CI as the `-m golden` selection so a refactor that
silently changes tokenizer behaviour will fail the suite.

## Reproducibility manifest

`ReproducibilityManifest.gather()` runs at the end of every
benchmark invocation and merges its fields into each CSV row:

- `python`: `sys.version`
- `platform`: `platform.platform()`
- `numpy`, `faiss`, `sentence_transformers`, `torch`, `bm25s`,
  `pydantic`: package versions, probed through a subprocess so a
  faulty native import cannot abort the runner.
- `faiss_omp_threads`: `faiss.omp_get_max_threads()`
- `git_sha`: `git rev-parse HEAD` from the repo root.
- `device`: the embedder's `device` argument.

Together with the per-row hyperparameter knobs from
`trivium/configs/default.yaml`, this makes every CSV row a
fully-specified benchmark recipe.

## Next steps

- Read [Concepts](../concepts/) for the domain types
  (`Document`, `Query`, `Qrels`, `Hit`, `SearchResult`).
- Read [Methodology](../../methodology/) for the experimental
  design and what the benchmark does NOT measure.
