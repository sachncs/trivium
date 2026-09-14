---
layout: page
title: Concepts
subtitle: The domain types the benchmark is built on.
section: Understand
permalink: /docs/concepts/
prev:
  title: Architecture
  url: /docs/architecture/
next:
  title: Methodology
  url: /docs/methodology/
---

trivium models a retrieval benchmark with five small value
types. They live in `trivium.domain`.

## `Document`

A `Document` is the unit indexed by a retriever and ranked by a
reranker.

```python
from trivium.domain.document import Document

doc = Document(doc_id="scifact:42", title="Coffee and sleep",
               text="Coffee reduces slow-wave sleep by 18% ...")
```

`doc_id` is the canonical id; every retrieval hit references the
same id. `title` and `text` are concatenated into `doc.body` so
the BM25 tokeniser sees a single string.

The id namespace convention is:

- `scifact:<id>` for gold documents in the BEIR scifact seed.
- `distractor:<id>` for the SCIDOCS / TREC-COVID / NFCorpus / FEVER
  distractors.

The strict prefix prevents collisions across sources.

## `Query`

```python
from trivium.domain.query import Query

query = Query(query_id="q12", text="Does coffee affect sleep?")
```

A `Query` is what the user types. The benchmark stores the raw
text; the embedder chooses the encoding template.

## `Qrels`

A `Qrels` (query relevance judgments) maps `query_id` to a
`doc_id -> relevance` mapping.

```python
from trivium.domain.qrels import Qrels

qrels = Qrels.from_rows([
    {"qid": "q12", "did": "scifact:42", "rel": 1},
    {"qid": "q12", "did": "scifact:99", "rel": 0},
])
```

The benchmark uses the closed-world assumption: unjudged
distractors are treated as non-relevant, following the Voorhees
2005 pooling convention.

## `SearchResult` and `Hit`

A `SearchResult` is the per-query output of a retriever or a
reranker. It contains a sequence of `Hit`s.

```python
from trivium.domain.result import SearchResult

result = SearchResult.from_pairs([
    ("scifact:42", 0.91),
    ("scifact:99", 0.42),
])
hits = list(result)  # [Hit(doc_id=..., score=...), ...]
```

`from_pairs` is the constructor used by every retriever; `Hit` is
the `(doc_id, score)` tuple with named attributes.

A retriever returns `list[SearchResult]`, one per query:

```python
results = retriever.search(query_vectors, k=10)
# results[i] is the top-k hits for query i
```

A `FusionStrategy` consumes that list and emits one fused
`SearchResult` per query.

## `PipelineResult`

A `PipelineResult` is the per-row CSV emission. Each pipeline run
returns one or more of these (RRF sweeps over `k` and weights
emit multiple rows from a single run).

```python
@dataclass
class PipelineResult:
    name: str                  # pipeline slug
    scale: int                 # corpus size at this row
    encoder: str               # encoder slug
    reranker: str              # reranker slug or "none"
    metrics: EvaluationMetrics # nDCG@10, R@10, R@100, etc.
    latency: LatencyStats      # p50, p95, mean, std, n
    extras: dict               # rrf_k, w_bm25, nprobe, ...
```

The CSV writer flattens this into one row; `extras` keys are
deduplicated against the metric names so they don't collide with
the headline columns.

## Domain modules

```text
trivium/domain/
├── __init__.py
├── document.py        # Document
├── hit.py             # Hit
├── pipeline_result.py # PipelineResult, LatencyStats-shaped result
├── qrels.py           # Qrels
├── query.py           # Query
└── result.py          # SearchResult
```

All five types are pure data classes. They are constructed and
consumed by every layer above, but they have no dependency on
faiss, bm25s, sentence-transformers, or torch — so you can build
tests against them without paying the import cost.

## Next steps

- Read [Methodology](../../methodology/) for the experimental
  design.
- Read the [API reference](../../api/) for every public symbol.
