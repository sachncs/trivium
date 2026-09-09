---
layout: page
title: Config schema
subtitle: Every field in trivium/configs/default.yaml.
section: Reference
permalink: /docs/reference/config/
prev:
  title: Tune the vector index
  url: /docs/guides/tune-vector/
next:
  title: CLI
  url: /docs/reference/cli/
---

The config file is loaded by `trivium.config.loader.load_config`
into a Pydantic v2 model with `extra="forbid"`. A typo in any
field name raises `ValueError` at load time, not deep in a
benchmark row.

The full Pydantic model lives in
`trivium/config/schema.py`; this page is the developer-facing
reference.

## Top-level keys

```yaml
version: 1

corpus:        # corpus construction rules
scales:        # [5000, 100000, 500000, 1000000]
encoders:      # list of EncoderEntry
bm25:          # BM25Config
vector:        # VectorConfig
hybrid:        # HybridConfig
rerank:        # RerankConfig
benchmark:     # BenchmarkConfig
preflight:     # PreflightConfig
runtime:       # RuntimeConfig
```

## `corpus`

`CorpusConfig`

| field                   | type   | default                    | description |
|-------------------------|--------|----------------------------|-------------|
| `seed`                  | str    | `"scifact"`               | BEIR seed corpus. |
| `distractors`           | list   | `["scidocs", "trec-covid", "nfcorpus"]` | Distractor sources. |
| `fever_sample`          | int    | `0`                        | FEVER documents to sample for padding past the BEIR ceiling. |
| `seed_random`           | int    | `42`                       | Random seed for the per-scale shuffle. |
| `dedup_jaccard_threshold`| float  | `0.85`                     | MinHash LSH threshold for near-duplicate detection. |

## `scales`

`list[int]` — the corpus sizes to benchmark at. Each scale is a
true prefix of the next.

Default: `[5000, 100000, 500000, 1000000]`.

## `encoders`

`list[EncoderEntry]` — the encoder slugs to use by default.

```yaml
encoders:
  - slug: minilm-l6
    model_id: sentence-transformers/all-MiniLM-L6-v2
    dimension: 384
    device: cpu
    batch_size: 128
    max_seq_length: 256
    normalize: true
    prompt_prefix_doc: ""
    prompt_prefix_query: ""
```

| field                  | type   | description |
|------------------------|--------|-------------|
| `slug`                 | str    | Identifier; must be registered via `@register_embedder`. |
| `model_id`             | str    | HuggingFace model id. |
| `dimension`            | int    | Embedding dimension (must match the model). |
| `device`               | str    | `cpu`, `cuda`, or `mps`. |
| `batch_size`           | int    | Encoding batch size. |
| `max_seq_length`       | int    | Tokeniser max length. |
| `normalize`            | bool   | L2-normalise vectors (recommended for cosine). |
| `prompt_prefix_doc`    | str    | Optional passage-side prefix. |
| `prompt_prefix_query`  | str    | Optional query-side prefix. |

## `bm25`

`Bm25Config`

| field            | type   | default  | description |
|------------------|--------|----------|-------------|
| `method`         | str    | `"lucene"` | bm25s method: `"lucene"`, `"atire"`, or `"bm25l"`. |
| `k1`             | float  | `0.9`    | Term-frequency saturation. **Do not change without updating the pre-flight target.** |
| `b`              | float  | `0.4`    | Document-length normalisation. **Same caveat.** |
| `candidate_pool` | int    | `100`    | Top-k candidates per query for BM25-only modes. |
| `stopwords`      | str    | `"en"`   | Stopword list (passed to bm25s.tokenization.Tokenizer). |
| `stemmer`        | str \| None | `null` | Stemmer (e.g. `"snowball"`); null for none. |

## `vector`

`VectorConfig`

| field                     | type                | default                              | description |
|---------------------------|---------------------|--------------------------------------|-------------|
| `nlist_by_scale`          | dict[int, int]      | (see below)                          | Coarse-quantizer count per scale. |
| `nprobe_sweep`            | list[int]           | `[8, 16, 32, 64, 128, 256]`           | Number of cells visited per query. |
| `m`                       | int                 | `48`                                 | Sub-quantizer count. |
| `nbits`                   | int                 | `4`                                  | Bits per sub-quantizer. |
| `use_opq`                 | bool                | `true`                               | Apply OPQ rotation before PQ. |
| `use_rflat`               | bool                | `true`                               | Apply exact refinement after PQ. |
| `k_factor`                | int                 | `4`                                  | Re-score top `k_factor * k` candidates. |
| `min_scale_for_ivfpq`     | int                 | `5000`                               | Below this scale, use IndexFlatIP. |
| `train_size_strategy`     | `"sqrt_n"` \| `"50_x_nlist"` \| `"fixed_N"` | `"50_x_nlist"` | OPQ-IVFPQ training-sample strategy. |
| `train_size_fixed`        | int                 | `30_000`                             | Sample size when strategy is `fixed_N`. |

Default `nlist_by_scale` is the formula-derived
`{5000: 512, 100000: 2048, 500000: 4096, 1000000: 4096}` (see
[Methodology](../../methodology/) for the rationale).

## `hybrid`

`HybridConfig`

| field               | type                  | default                          | description |
|---------------------|-----------------------|----------------------------------|-------------|
| `rrf_k`             | int                   | `60`                             | Default RRF smoothing constant. |
| `rrf_k_sweep`       | list[int]             | `[10, 30, 60, 100, 200]`         | Sweep over RRF k. |
| `rrf_weight_sweep`  | list[tuple[float, float]] | `[(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]` | Sweep over BM25-vs-vector weights. |
| `candidate_pool`    | int                   | `100`                            | Top-k candidates per query for hybrid modes. |

## `rerank`

`RerankConfig`

| field            | type   | default                                  | description |
|------------------|--------|------------------------------------------|-------------|
| `slug`           | str    | `"encoder"`                              | Reranker slug (must be registered). |
| `model_id`       | str    | `"cross-encoder/ms-marco-MiniLM-L-6-v2"`  | HuggingFace model id. |
| `max_length`     | int    | `256`                                    | Tokeniser max length. |
| `batch_size`     | int    | `32`                                     | Rerank batch size. |
| `candidate_pool` | int    | `50`                                     | Top-k candidates fed to the reranker (smaller than `hybrid.candidate_pool` because rerank is expensive). |

## `benchmark`

`BenchmarkConfig`

| field        | type      | default           | description |
|--------------|-----------|-------------------|-------------|
| `warmup`     | int       | `20`              | Number of warmup queries discarded before timing. |
| `iters`      | int       | `3`               | Iterations per measurement (averaged into latency). |
| `top_k_eval` | list[int] | `[10, 100]`       | Cutoffs for nDCG and recall. |

## `preflight`

`PreflightConfig`

| field           | type  | default | description |
|-----------------|-------|---------|-------------|
| `target_ndcg10` | float | `0.6789` | Anserini BM25 nDCG@10 on BEIR scifact. |
| `tolerance`     | float | `0.03`   | Drift band; runner warns if `|observed - target| > tolerance`. |
| `min_observed`  | float | `0.62`   | Bug detector; runner fails if `observed < min_observed`. |

## `runtime`

`RuntimeConfig`

| field             | type | default | description |
|-------------------|------|---------|-------------|
| `random_seed`     | int  | `42`    | NumPy / Python random seed. |
| `python_hash_seed`| int  | `42`    | `PYTHONHASHSEED` value (set in env before benchmark). |
| `faiss_omp_threads`| int | `1`     | OpenMP thread count for FAISS. Set to `os.cpu_count()` for parallel benches. |

## Loading

```python
from trivium.config.loader import load_config, default_config_path

cfg = load_config()                                 # bundled default
cfg = load_config("/path/to/custom.yaml")           # explicit path
```

`default_config_path()` returns:

1. `$PWD/configs/default.yaml` if it exists (so contributors
   can edit the canonical file in place).
2. Otherwise, `trivium/configs/default.yaml` from the installed
   package.

## Next steps

- Read [CLI](../../reference/cli/) for the runner flags.
- Read [API](../../api/) for the public Python API.
