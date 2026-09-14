---
layout: page
title: Add an embedder
subtitle: Plug your own text encoder into the benchmark.
section: How-to
permalink: /docs/guides/add-embedder/
prev:
  title: Methodology
  url: /docs/methodology/
next:
  title: Add a reranker
  url: /docs/guides/add-reranker/
---

This page walks through adding a new embedder slug so the runner
can use your model with `--encoders your-slug`.

The default registry ships with six slugs:

| slug             | model                                  | dim   | weight |
|------------------|----------------------------------------|------:|-------:|
| `minilm-l6`      | sentence-transformers/all-MiniLM-L6-v2 |  384  | 90 MB  |
| `bge-small`      | BAAI/bge-small-en-v1.5                  |  384  | 130 MB |
| `bge-base`       | BAAI/bge-base-en-v1.5                   |  768  | 440 MB |
| `bge-large`      | BAAI/bge-large-en-v1.5                 | 1024  | 1.3 GB |
| `mpnet-base`     | sentence-transformers/all-mpnet-base-v2 |  768  | 440 MB |
| `e5-mistral-7b`  | intfloat/e5-mistral-7b-instruct        | 4096  | 14 GB  |

You can register any HuggingFace sentence-transformers-compatible
model. The two implementation backends are:

- `Sentence` — wraps any sentence-transformers model.
- `E5` — wraps E5-Mistral-7B-instruct with the
  `query: ...\npassage: ...` template.

## Add a sentence-transformers model

If the model you want is loadable via `SentenceTransformer`, the
fastest path is to write a one-line factory.

```python
# trivium/embeddings/registry.py

@register_embedder("my-encoder")
def make_my_encoder(**kwargs) -> Embedder:
    from trivium.embeddings.sentence import Sentence

    defaults = {
        "slug": "my-encoder",
        "model_id": "your-org/your-model",
        "dimension": 768,            # match the model
        "max_seq_length": 512,
        "prompt_prefix_doc": "",     # BGE-style: "passage: "
        "prompt_prefix_query": "",   # BGE-style: "query: "
        "device": "cpu",             # or "cuda" / "mps"
    }
    defaults.update(kwargs)
    return _sentence_module.Sentence(**defaults)
```

After the decorator runs, `get_embedder("my-encoder")` returns a
working encoder and `--encoders my-encoder` works on the CLI.

## Add a custom embedder backend

For non-sentence-transformers backends, subclass `Embedder`
directly.

```python
# trivium/embeddings/my_backend.py

import numpy as np
from trivium.embeddings.base import Embedder


class MyBackend(Embedder):
    def __init__(self, slug: str, model_id: str, dimension: int) -> None:
        self.slug_value = slug
        self.model_id_value = model_id
        self._dimension = dimension
        self.model = None  # lazy-loaded

    @property
    def slug(self) -> str:
        return self.slug_value

    @property
    def model_id(self) -> str:
        return self.model_id_value

    @property
    def dimension(self) -> int:
        return self._dimension

    def warmup(self, sample_texts=("warmup",)) -> None:
        if self.model is None:
            self.ensure_model()
        # ... your warmup logic ...

    def encode_documents(self, texts):
        self.ensure_model()
        vectors = self.model.encode_documents(list(texts))
        return np.asarray(vectors, dtype=np.float32)

    def encode_queries(self, texts):
        self.ensure_model()
        vectors = self.model.encode_queries(list(texts))
        return np.asarray(vectors, dtype=np.float32)

    def ensure_model(self) -> None:
        if self.model is not None:
            return
        # ... your model load ...
        self.model = ...
```

Then register the factory in
`trivium/embeddings/registry.py`:

```python
@register_embedder("my-backend")
def make_my_backend(**kwargs) -> Embedder:
    import trivium.embeddings.my_backend as _mod
    defaults = {"slug": "my-backend", "model_id": "your/model",
                "dimension": 768}
    defaults.update(kwargs)
    return _mod.MyBackend(**defaults)
```

`from trivium.embeddings.my_backend import MyBackend` is
deliberately inside the function body so the heavy import is
deferred to first use.

## Decide whether to add a CLI flag

If you want `--encoders my-encoder` to work without a config
change, that's enough. If you want the embedder to participate in
the default loop, also add it to
`trivium/configs/default.yaml`:

```yaml
encoders:
  - slug: minilm-l6
    ...
  - slug: my-encoder
    model_id: your-org/your-model
    dimension: 768
    device: cpu
    batch_size: 128
    max_seq_length: 512
    normalize: true
```

The runner sweeps every encoder listed in `config.encoders` when
no `--encoders` flag is passed.

## Test the new embedder

Add a unit test that pins the slug, model_id, and dimension:

```python
# tests/embeddings/test_registry.py

def test_my_encoder_factory() -> None:
    from trivium.embeddings import registry as embedder_registry
    emb = embedder_registry.get_embedder("my-encoder")
    assert emb.slug == "my-encoder"
    assert emb.model_id == "your-org/your-model"
    assert emb.dimension == 768
```

Use the same `sys.modules` stubbing pattern as the existing tests
to avoid loading `sentence_transformers` in CI:

```python
import sys
import types

@pytest.fixture(autouse=True)
def _stub_heavy_modules(monkeypatch):
    sentence_stub = types.ModuleType("trivium.embeddings.sentence")
    sentence_stub.Sentence = _StubSentence
    monkeypatch.setitem(sys.modules, "trivium.embeddings.sentence", sentence_stub)
```

## Avoid the common pitfalls

- **Match the dimension.** Setting the wrong dimension produces
  silent shape errors at `IndexFlatIP.add()`.
- **Pin the prompt prefixes.** BGE and E5 use different
  query/passage templates. Use the project's recommended prefix;
  mis-prefixing drops recall by ~5 nDCG points.
- **Normalise for cosine.** For `IndexFlatIP` cosine, set
  `normalize=true` (the default in `Sentence`) so the inner
  product equals cosine similarity.
- **Warm up before timing.** The runner calls `warmup()` once per
  encoder so the first-query cost doesn't poison p50 / p95.

## Next steps

- Read [Add a reranker](../add-reranker/) for the same pattern on
  the reranking side.
- Read [Add a pipeline](../add-pipeline/) to compose your new
  embedder into a new benchmark mode.
