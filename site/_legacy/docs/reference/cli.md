---
layout: page
title: CLI
subtitle: Every flag on trivium-benchmark, trivium-prepare, trivium-summarise, and trivium-gigatoken-bench.
section: Reference
permalink: /docs/reference/cli/
prev:
  title: Config schema
  url: /docs/reference/config/
next:
  title: API
  url: /docs/api/
---

This page documents the four console scripts that `pip install
trivium` puts on your `$PATH`. Every flag is also discoverable via
`--help`.

## `trivium-benchmark`

Run the benchmark. Emits one CSV row per `(mode, encoder,
reranker, scale)` combination.

```text
trivium-benchmark [options]

options:
  --config CONFIG         Path to YAML config. Default: bundled default_config_path().
  --modes MODES           Comma-separated subset of pipeline modes.
                          Default: bm25,vector,hybrid_rrf
  --scales SCALES         Comma-separated subset of scales (e.g. '5000,100000').
                          Default: every scale in config.scales.
  --encoders ENCODERS     Comma-separated subset of encoder slugs.
                          Default: every encoder in config.encoders.
  --rerankers RERANKERS   Comma-separated subset of reranker slugs.
                          Default: encoder
  --output OUTPUT         CSV output path. Default: results/benchmark.csv
  --skip-preflight        Skip the BM25 nDCG@10 equivalence check.
```

### Examples

```bash
# Default 5K sweep with all default modes
trivium-benchmark --scales 5000

# Full 5K + 100K with rerank
trivium-benchmark --modes bm25,vector,hybrid_rrf,hybrid_rerank \
                  --scales 5000,100000

# Compare two encoders on hybrid_rrf
trivium-benchmark --modes hybrid_rrf --encoders minilm-l6,bge-base \
                  --scales 5000 --output results/compare.csv

# Restrict the IVFPQ nprobe sweep
trivium-benchmark --modes vector --nprobe-only 32
```

### Output schema

The CSV has one row per `(mode, encoder, reranker, scale,
hyperparam-combination)`. The columns are:

- `mode` — pipeline slug.
- `scale` — corpus size.
- `encoder` — encoder slug (or `none` for non-vector modes).
- `reranker` — reranker slug (or `none`).
- `ndcg_cut.10`, `ndcg_cut.100` — nDCG at 10 and 100.
- `recall.10`, `recall.100` — recall at 10 and 100.
- `p50_ms`, `p95_ms` — per-query latency percentiles.
- `python`, `platform`, `numpy`, `faiss`, `sentence_transformers`,
  `torch`, `bm25s`, `pydantic`, `faiss_omp_threads`, `git_sha` —
  reproducibility fields from `ReproducibilityManifest`.
- Per-mode extras: `rrf_k`, `w_bm25`, `w_vec`, `nprobe`,
  `nlist`, `m`, `nbits`, `search_mode`, `index_type`, ...

### Exit codes

| code | meaning |
|------|---------|
| 0    | All selected modes completed; CSV written. |
| 1    | Pre-flight BM25 nDCG@10 below 0.62 (pipeline broken). |
| 1    | Any selected encoder / reranker failed to instantiate (also reported per line). |

## `trivium-prepare`

Build the BEIR-augmented scifact corpus and embeddings. Cache
lives under `data/cache/`.

```text
trivium-prepare [options]

options:
  --config CONFIG         Path to YAML config. Default: bundled default_config_path().
  --max-scales MAX_SCALES Cap the scales to build, e.g. '5000,100000'.
  --skip-embeddings        Skip the per-encoder embedding pass.
```

### Examples

```bash
# Default 5K + 100K
trivium-prepare --max-scales 100000

# Build the full 1M scale (1-2 hours on a laptop)
trivium-prepare

# Just dedupe and slice the corpus, embed later
trivium-prepare --max-scales 100000 --skip-embeddings
```

### Output

`data/cache/`:

- `scifact_seed.jsonl` — the 5K BEIR scifact seed.
- `corpus.jsonl` — the largest scale, all docs.
- `vectors.npz` — float32 corpus embeddings.
- `queries.jsonl`, `qrels.jsonl` — the 300 BEIR scifact test
  queries and their relevance judgments.
- `manifest.json` — corpus SHA-256, encoder revisions, per-scale
  counts.

## `trivium-summarise`

Print headline rows from a benchmark CSV.

```text
trivium-summarise --csv CSV
```

### Examples

```bash
trivium-summarise --csv results/benchmark.csv
```

### Output

```
  scale    mode          encoder       ndcg@10    R@10   p95_ms
   5_000    bm25          minilm-l6     0.6569  0.7784     0.90
   5_000    vector        minilm-l6     0.6451  0.7833     0.10
   5_000    hybrid_rrf    minilm-l6     0.7016  0.8482     1.10
   5_000    hybrid_rerank minilm-l6     0.6820  0.8062   167.30
 ...
```

The summariser is the source of truth for the README headline
numbers. Re-generate the README by running the benchmark, then
`summarise`, then updating the table.

## `trivium-gigatoken-bench`

Throughput benchmark for the `GigaToken` wrapper
(backed by the gigatoken Rust engine).

```text
trivium-gigatoken-bench [options]

options:
  --config CONFIG         Path to YAML config. Default: bundled default_config_path().
  --tokenizer TOKENIZER   Tokenizer to use. Default: gpt2.
  --scales SCALES         Comma-separated scales. Default: every scale in config.scales.
  --output OUTPUT         CSV output path. Default: results/gigatoken_benchmark.csv.
```

### Examples

```bash
trivium-gigatoken-bench --scales 5000,100000
trivium-gigatoken-bench --tokenizer gpt2 --output /tmp/gigabench.csv
```

## Next steps

- Read [API](../../api/) for the Python interfaces.
- Read [Config schema](../../reference/config/) for the YAML
  fields.
