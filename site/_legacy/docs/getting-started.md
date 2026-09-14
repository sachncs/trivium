---
layout: page
title: Getting started
subtitle: Install trivium, prepare the corpus, and run your first benchmark.
section: Getting started
permalink: /docs/getting-started/
prev:
  title: trivium home
  url: /
next:
  title: Architecture
  url: /docs/architecture/
---

This page walks you through installing trivium, preparing the
BEIR-augmented scifact corpus, and running your first benchmark.
Allow about 30 minutes on a laptop for the default 5K + 100K sweep.

## Before you start

You need:

- Python 3.12 or newer.
- About 6 GB of free disk space for the BEIR corpora.
- About 1 GB of free disk space for the Python dependencies.
- About 2 GB of free RAM for the MiniLM-L6 model + IVFPQ index at
  the 100K scale.

trivium runs on Linux, macOS, and Windows. The benchmark itself is
CPU-only by default; a CUDA or MPS device is optional and only
required if you register a GPU-bound encoder or reranker.

## Install

```bash
pip install trivium
```

This installs four console scripts (`trivium-benchmark`,
`trivium-prepare`, `trivium-summarise`, `trivium-gigatoken-bench`)
and the `trivium` Python package.

To verify the install:

```bash
trivium-benchmark --help
trivium-prepare --help
```

Both should print a usage block and exit 0.

### Install from source

If you want to hack on the benchmark itself:

```bash
git clone https://github.com/sachncs/trivium
cd trivium
pip install -e ".[dev]"
```

The editable install exposes the same `trivium` package, but you
can now edit the source and re-run tests without re-installing.

## Prepare the corpus and embeddings

```bash
trivium-prepare --max-scales 100000
```

This step:

1. Downloads the BEIR scifact seed (5,183 documents, 300 queries,
   qrels).
2. Downloads scientific distractors from SCIDOCS, TREC-COVID, and
   NFCorpus, with FEVER sampled for padding past the BEIR ceiling.
3. Deduplicates by SHA-256 (exact) and MinHash LSH (near-dup,
   threshold 0.85).
4. Slices the corpus into 5K, 100K, 500K, and 1M shards, all
   true prefixes of each other.
5. Embeds the corpus with `all-MiniLM-L6-v2` (384d, l2-normalized).
6. Writes a `manifest.json` with the corpus SHA-256, encoder
   revisions, and per-scale counts.

The cache lives under `data/cache/`. Expect roughly 5 minutes on a
laptop for the 5K + 100K sweep. The 1M scale adds about 90 minutes.

## Run the benchmark

```bash
trivium-benchmark \
  --modes bm25,vector,hybrid_rrf,hybrid_rerank \
  --scales 5000,100000 \
  --output results/benchmark.csv
```

You should see the pre-flight check print first:

```
=== Pre-flight: BM25 nDCG@10 on untouched 5K scifact ===
  observed nDCG@10: 0.6569
  target (Anserini): 0.6789 +/- 0.03
  PASS: |delta|=0.0220 within tolerance.
```

If the pre-flight check fails with `observed < 0.62`, the pipeline
is broken — your tokenizer, BM25 hyperparameters, or qrels prefix
have drifted. Stop and debug before re-running.

## Summarise the headline rows

```bash
trivium-summarise --csv results/benchmark.csv
```

prints a per-(scale, mode, encoder) pivot of the headline numbers:

```
  scale    mode          encoder         ndcg@10    R@10   p95_ms
   5_000    bm25          minilm-l6       0.6569  0.7784     0.90
   5_000    vector        minilm-l6       0.6451  0.7833     0.10
   5_000    hybrid_rrf    minilm-l6       0.7016  0.8482     1.10
   5_000    hybrid_rerank minilm-l6       0.6820  0.8062   167.30
 100_000    bm25          minilm-l6       0.3206  0.3793     6.50
 ...
```

These numbers match the headline table in the README and the
landing page hero.

## Try a different encoder or reranker

The runner wires encoders and rerankers through the public
registries. Pass any registered slug with `--encoders` or
`--rerankers`:

```bash
trivium-benchmark \
  --modes hybrid_rrf,hybrid_rerank \
  --encoders minilm-l6,bge-base \
  --rerankers encoder,bge-large \
  --scales 5000 \
  --output results/multi-encoder.csv
```

The benchmark emits one CSV row per `(mode, encoder, reranker,
scale)` combination. See [How-to guides](../guides/add-embedder/)
for how to register your own encoder.

## Next steps

- Read [Architecture](../architecture/) to understand the
  pipeline registry, embedder registry, reranker registry, and
  retriever ABCs.
- Read [Methodology](../../methodology/) to understand the BM25
  pre-flight invariant, the IVFPQ + RFlat story, and the corpus
  construction rules.
- Read [How-to: Add an embedder](../../guides/add-embedder/) to
  plug your own model into the benchmark.
