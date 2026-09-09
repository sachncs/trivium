---
layout: page
title: Add a reranker
subtitle: Plug your own cross-encoder or seq2seq reranker into the benchmark.
section: How-to
permalink: /docs/guides/add-reranker/
prev:
  title: Add an embedder
  url: /docs/guides/add-embedder/
next:
  title: Add a pipeline
  url: /docs/guides/add-pipeline/
---

This page walks through adding a new reranker slug so the runner
can use your cross-encoder with `--rerankers your-slug`.

The default registry ships with four slugs:

| slug         | model                                  | weight |
|--------------|----------------------------------------|-------:|
| `encoder`    | cross-encoder/ms-marco-MiniLM-L-6-v2   |  90 MB |
| `bge-large`  | BAAI/bge-reranker-large                | 2.3 GB |
| `bge-base`   | BAAI/bge-reranker-base                 | 450 MB |
| `monot5-3b`  | castorini/monot5-3b-msmarco-10k        |  11 GB |

Two concrete implementations:

- `Encoder` — any `sentence_transformers.CrossEncoder` model.
- `Bge` — BGE rerankers with sigmoid activation on the logit.
- `Monot5` — T5 seq2seq that reads `Query: ... Document: ...
  Relevant:` and scores the `true` vs `false` decision token.

## Add a sentence-transformers cross-encoder

For any model loadable via `CrossEncoder`:

```python
# trivium/reranking/registry.py

@register_reranker("my-reranker")
def make_my_reranker(**kwargs) -> Reranker:
    from trivium.reranking.encoder import Encoder

    defaults = {
        "slug": "my-reranker",
        "model_id": "your-org/your-cross-encoder",
        "max_length": 512,
        "batch_size": 16,
    }
    defaults.update(kwargs)
    return Encoder(**defaults)
```

After this runs, `get_reranker("my-reranker")` returns a working
reranker and `--rerankers my-reranker` works on the CLI.

## Add a BGE-style reranker

For BGE rerankers, subclass `Bge` with a different `model_id`:

```python
@register_reranker("bge-m3-reranker")
def make_bge_m3(**kwargs) -> Reranker:
    from trivium.reranking.bge import Bge

    defaults = {
        "slug": "bge-m3-reranker",
        "model_id": "BAAI/bge-reranker-v2-m3",
    }
    defaults.update(kwargs)
    return Bge(**defaults)
```

The `Bge` class already handles the sigmoid-on-logit post-processing
that BGE rerankers need.

## Add a custom reranker backend

For non-cross-encoder backends (LLaMA-based rerankers,
rank-T5, ColBERT-style late interaction, etc.), subclass
`Reranker` directly.

```python
# trivium/reranking/my_backend.py

import numpy as np
from trivium.domain.document import Document
from trivium.domain.result import SearchResult
from trivium.reranking.base import Reranker


class MyReranker(Reranker):
    def __init__(self, slug: str, model_id: str) -> None:
        self.slug_value = slug
        self.model_id_value = model_id
        self.model = None

    @property
    def slug(self) -> str:
        return self.slug_value

    @property
    def model_id(self) -> str:
        return self.model_id_value

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return SearchResult.empty()
        self.ensure_model()
        scores = self.model.score(query, [c.body for c in candidates])
        order = np.argsort(-np.asarray(scores))[:top_k]
        return SearchResult(hits=[
            (candidates[int(i)], float(scores[int(i)]))
            for i in order
        ])

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        # ... lazy-load ...
        self.model = ...
```

Then register the factory:

```python
@register_reranker("my-backend")
def make_my_backend(**kwargs) -> Reranker:
    from trivium.reranking import my_backend as _mod
    defaults = {"slug": "my-backend", "model_id": "your/model"}
    defaults.update(kwargs)
    return _mod.MyReranker(**defaults)
```

## Test the new reranker

Add a unit test that pins the slug and model_id:

```python
# tests/reranking/test_registry.py

def test_my_reranker_factory(monkeypatch):
    from trivium.reranking import registry as rr_registry
    from trivium.reranking.base import Reranker

    class _Stub(MyReranker):
        def ensure_model(self): pass

    # Stub the heavy backend module
    import sys, types
    backend_stub = types.ModuleType("trivium.reranking.my_backend")
    backend_stub.MyReranker = _Stub
    monkeypatch.setitem(sys.modules, "trivium.reranking.my_backend", backend_stub)

    rr = rr_registry.get_reranker("my-backend")
    assert isinstance(rr, Reranker)
    assert rr.slug == "my-backend"
```

## Avoid the common pitfalls

- **Max length matters.** BGE-large uses 512, MiniLM-L-6 uses 256.
  The default `max_length` on `Encoder` is 256; override it via
  `kwargs` if your model supports longer.
- **Batch size vs latency.** The published `hybrid_rerank` latency
  is batch=32. Smaller batches degrade throughput significantly.
- **GPU OOMs.** The runner doesn't pin a device; pass
  `device="cpu"` if you don't want to silently OOM on a shared
  machine.
- **Score semantics.** BGE rerankers apply `sigmoid` to the
  CrossEncoder logit; `Encoder` does not. Mixing these in a
  threshold-based downstream will surprise you.

## Next steps

- Read [Add a pipeline](../add-pipeline/) to compose your reranker
  into a new benchmark mode.
- Read [Tune the vector index](../../reference/cli/) for how the
  CLI flags interact with the reranker registry.
