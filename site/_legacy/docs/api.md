---
layout: page
title: API reference
subtitle: The public Python API for the embedder, reranker, retriever, and pipeline registries.
section: Reference
permalink: /docs/api/
prev:
  title: CLI
  url: /docs/reference/cli/
next:
  title: Contributing
  url: /docs/contributing/
---

This page is the developer-facing API reference. It covers every
public symbol you can import after `pip install trivium`.

## `trivium.domain` — value types

### `Document`

```python
@dataclass
class Document:
    doc_id: str
    title: str = ""
    text: str = ""

    @property
    def body(self) -> str:
        return f"{self.title}\n{self.text}".strip()
```

### `Query`

```python
@dataclass
class Query:
    query_id: str
    text: str
```

### `Qrels`

```python
class Qrels:
    def __init__(self, judgments: dict[str, dict[str, int]]) -> None: ...

    @classmethod
    def from_rows(cls, rows: list[dict]) -> Qrels:
        """rows = [{"qid": ..., "did": ..., "rel": int}, ...]"""
```

### `Hit`

```python
@dataclass
class Hit:
    doc_id: str
    score: float
```

### `SearchResult`

```python
class SearchResult:
    def __init__(self, hits: Iterable[Hit | tuple[str, float]]) -> None: ...

    @classmethod
    def empty(cls) -> SearchResult: ...

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, float]]) -> SearchResult:
        """Build from (doc_id, score) tuples."""
```

### `PipelineResult`

```python
@dataclass
class PipelineResult:
    name: str            # pipeline slug
    scale: int           # corpus size
    encoder: str         # encoder slug
    reranker: str        # reranker slug or "none"
    metrics: EvaluationMetrics
    latency: LatencyStats
    extras: dict         # rrf_k, w_bm25, nprobe, ...
```

## `trivium.embeddings` — text vectorizers

### `Embedder` (ABC)

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

### Concrete classes

```python
from trivium.embeddings.sentence import Sentence
from trivium.embeddings.e5 import E5

# Sentence wraps any sentence-transformers model
encoder = Sentence(
    slug="minilm-l6",
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    dimension=384,
    device="cpu",
    normalize=True,
)

# E5 wraps intfloat/e5-mistral-7b-instruct
encoder = E5(slug="e5-mistral-7b", max_memory_gb=None, device="mps")
```

### `EmbedderRegistry` and factory functions

```python
from trivium.embeddings.registry import get_embedder, register_embedder

# Build from slug
encoder = get_embedder("minilm-l6")              # default kwargs
encoder = get_embedder("bge-large", device="cuda")

# Available slugs
from trivium.embeddings.registry import EmbedderRegistry
print(EmbedderRegistry.slugs())
# ['bge-base', 'bge-large', 'bge-small', 'e5-mistral-7b', 'minilm-l6', 'mpnet-base']

# Register a new slug
@register_embedder("my-encoder")
def make_my_encoder(**kwargs) -> Embedder:
    ...
```

## `trivium.retrieval` — sparse + dense retrievers

### `Retriever` (ABC)

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

### Concrete retrievers

```python
from trivium.retrieval.bm25 import Bm25
from trivium.retrieval.faiss import Faiss
from trivium.retrieval.diskbbq import Bbq

bm25 = Bm25(k1=0.9, b=0.4, method="lucene")
bm25.add_documents(docs)
results = bm25.search(["query text"], k=10)

flat = Faiss.Flat()
flat.add_documents(docs, vectors=doc_vecs)
results = flat.search(query_vecs, k=10)

bbq = Bbq(out_dir="data/cache/bbq", nlist=4096, m=48, nprobe=64)
bbq.add_documents(docs, vectors=doc_vecs)
results = bbq.search(query_vecs, k=10, mode="ram")   # or "disk" / "auto"
```

### Save and load

```python
bm25.save("path/to/bm25")
bm25 = Bm25()
bm25.load("path/to/bm25")    # restores doc_ids
```

## `trivium.fusion` — score combinators

### `FusionStrategy` (ABC)

```python
class FusionStrategy(ABC):
    @property
    @abstractmethod
    def slug(self) -> str: ...

    @abstractmethod
    def fuse(self, results: Sequence[SearchResult], top_k: int) -> SearchResult: ...
```

### `Rrf` (Reciprocal Rank Fusion)

```python
from trivium.fusion.rrf import Rrf

rrf = Rrf(k=60, weights=[0.5, 0.5])     # Cormack et al. 2009
fused = rrf.fuse([bm25_results, dense_results], top_k=100)
```

### `Weighted`

```python
from trivium.fusion.weighted import Weighted
```

## `trivium.reranking` — cross-encoders

### `Reranker` (ABC)

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

### Concrete rerankers

```python
from trivium.reranking.encoder import Encoder
from trivium.reranking.bge import Bge
from trivium.reranking.monot5 import Monot5

# Sentence-transformers CrossEncoder
rr = Encoder(slug="encoder",
             model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
             max_length=256, batch_size=32)

# BGE (sigmoid on logit)
rr = Bge(slug="bge-large", model_id="BAAI/bge-reranker-large")

# MonoT5 (T5 seq2seq)
rr = Monot5(slug="monot5-3b", model_id="castorini/monot5-3b-msmarco-10k",
            batch_size=4, fp16=True)
```

### `RerankerRegistry`

```python
from trivium.reranking.registry import get_reranker, register_reranker, RerankerRegistry

rr = get_reranker("encoder")
rr = get_reranker("bge-large")
print(RerankerRegistry.slugs())
# ['bge-base', 'bge-large', 'encoder', 'monot5-3b']
```

## `trivium.pipelines` — benchmark orchestration

### `BenchmarkPipeline` (ABC)

```python
class BenchmarkPipeline(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def encoders(self) -> Sequence[str]: ...

    @abstractmethod
    def run(self, inp: PipelineInput, config: Config) -> list[PipelineResult]: ...
```

### Built-in modes

```python
from trivium.pipelines import (
    bm25, vector, hybrid_rrf, hybrid_rerank, diskbbq, hybrid_bbq,
)
# Importing each module registers its @register_pipeline decorated class.
```

### `PipelineRegistry`

```python
from trivium.pipelines.registry import get_pipeline, PipelineRegistry

pipe = get_pipeline("hybrid_rrf")
print(PipelineRegistry.slugs())
# ['bm25', 'vector', 'hybrid_rrf', 'hybrid_rerank', 'diskbbq', 'hybrid_bbq']
```

### `PipelineInput`

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

## `trivium.config` — configuration

```python
from trivium.config.loader import load_config, default_config_path

cfg = load_config()                                # default
cfg = load_config("path/to/custom.yaml")
```

The returned `Config` is a Pydantic v2 model with `extra="forbid"`.
See [Config schema](../../reference/config/) for every field.

## `trivium.evaluation` — metrics and latency

```python
from trivium.evaluation.metrics import Evaluator, hits_to_results
from trivium.evaluation.latency import LatencyProbe
```

`Evaluator` wraps `pytrec_eval.RelevanceEvaluator`:

```python
ev = Evaluator(k_values=[10, 100])
metrics = ev.evaluate(qrels, eval_pairs)
print(metrics.ndcg_at_10, metrics.recall_at_10)
```

`LatencyProbe` measures per-query latency under warmup:

```python
probe = LatencyProbe(fn=step_fn, n=100, warmup=20, rotate=list(range(100)))
stats = probe.run()    # LatencyStats(mean, std, p50, p95, ...)
```

## `trivium.io` — corpus cache

```python
from trivium.io.corpus import CorpusCache

cache = CorpusCache("data/cache")
seed = cache.load_scale(5000)
vectors, ids = cache.load_vectors()
```

## Next steps

- Read [Contributing](../../contributing/) to learn the
  development workflow.
- Read [Methodology](../../methodology/) to understand the
  experimental design.
