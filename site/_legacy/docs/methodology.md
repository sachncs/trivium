---
layout: page
title: Methodology
subtitle: Corpus construction, embedding trade-offs, the IVFPQ + RFlat story, and the BM25 pre-flight invariant.
section: Understand
permalink: /docs/methodology/
prev:
  title: Concepts
  url: /docs/concepts/
next:
  title: Add an embedder
  url: /docs/guides/add-embedder/
---

This page documents what the benchmark measures, what it does
NOT measure, and why each methodological choice was made.

## What we measure

For every `(mode, encoder, reranker, scale, hyperparameter-knob)`
combination, the runner emits one CSV row with:

- **nDCG@10**: `pytrec_eval.RelevanceEvaluator` with BEIR semantics
  (macro average, log₂ gain, `ignore_identical_ids=True`,
  `2^rel - 1` gain).
- **R@10** and **R@100**: same evaluator.
- **p50** and **p95** latency in milliseconds, measured per query
  with `time.perf_counter_ns()`, 20-query warmup, single-stream.

The CSV row also carries the `python`, `platform`, `numpy`,
`faiss`, `sentence-transformers`, `torch`, `bm25s`, `faiss_omp_threads`,
`git_sha`, encoder model id, scale, mode, and every knob from
`trivium/configs/default.yaml`.

## Corpus construction

### Seed

The 5K seed is BEIR scifact (`beir-v1.0.0-scifact`): 5,183 documents,
300 test queries, qrels unchanged. We load it via
`beir.util.download_and_unzip` so the canonical BEIR zip is the
single source of truth.

### Distractors

Distractors come from three BEIR scientific corpora:

- SCIDOCS (~25.7K documents)
- TREC-COVID (~171K documents)
- NFCorpus (~3.6K documents)

We deliberately exclude MS MARCO because the domain mismatch
(web passages vs scientific abstracts) would inflate the
nDCG gap between methods.

For the 1M scale we additionally sample 800K documents from FEVER,
the closest non-scientific BEIR corpus, to pad past the BEIR
scientific ceiling.

### Dedup

Dedup runs in two passes:

1. SHA-256 over `title + text` for exact duplicates.
2. MinHash LSH with Jaccard threshold 0.85 for near-duplicates.

Dedup removes about 2% of the raw pool.

### Scale slicing

The benchmark exposes four scales: 5K, 100K, 500K, 1M. Each scale
is a true prefix of the next so the scale curve is comparable.

The slice order is fixed by `random.Random(42).shuffle(pool)` so
the 100K scale is a prefix of the 1M scale regardless of how
many documents actually exist after dedup.

### Naming

Each document id carries a source prefix to prevent collisions:

- `scifact:<id>` for gold documents.
- `distractor:<id>` for SCIDOCS / TREC-COVID / NFCorpus / FEVER.

### Closed-world qrels

The BEIR scifact qrels are used unchanged. Unjudged distractors
are treated as non-relevant (Voorhees 2005 pooling convention).

## Embeddings

The default encoder is `sentence-transformers/all-MiniLM-L6-v2`
(384 dimensions, l2-normalised). Three reasons:

1. **Speed**: roughly 5× faster than `mpnet-base-v2` on CPU.
2. **Memory**: 90 MB download.
3. **Throughput**: large batches at the 1M scale without a GPU.

The published trade-off is roughly 0.03 nDCG@10 vs BGE-large
(`BAAI/bge-large-en-v1.5`). The encoder registry also ships
BGE-small, BGE-base, BGE-large, mpnet-base, and E5-Mistral-7B.
Register any HuggingFace sentence-transformers-compatible model
with one decorator call.

`IndexFlatIP` (the exact dense ceiling) uses cosine similarity
over l2-normalised vectors.

## Vector index

At each scale the runner uses FAISS `IndexFlatIP` for ≤ 5K and the
approximate composite index `IndexOPQ48,IVF{nlist},PQ48x4fs,RFlat`
above 5K.

### nlist

`nlist` follows the FAISS wiki recipe: `4·sqrt(N)` rounded up to
the next power of two. The literal values are in
`trivium/configs/default.yaml` under `vector.nlist_by_scale`:

| scale       | `4·sqrt(N)` | rounded to power of 2 |
|------------:|------------:|----------------------:|
| 5_000       | 283         | 512                   |
| 100_000     | 1_265       | 2_048                 |
| 500_000     | 2_828       | 4_096                 |
| 1_000_000   | 4_000       | 4_096                 |

The 1M scale is intentionally capped at 4_096 so coarse-quantizer
training stays `O(N)` rather than the next-power-of-two 8_192.

### nprobe sweep

The runner sweeps `nprobe` over {8, 16, 32, 64, 128, 256} and
keeps the Pareto-frontier row per `(scale, mode, encoder)`
combination.

### RFlat

`k_factor = 4` rescores 4× more candidates than `k` exactly. This
recovers ~+5-10% ANN recall vs plain IVFPQ at the cost of one
exact inner product per additional candidate.

## BM25

The BM25 retriever is `bm25s` with:

- `method="lucene"`
- `k1=0.9`
- `b=0.4` (Anserini defaults)

The choice of `k1=0.9, b=0.4` reproduces the BEIR published
baseline of `0.6789` nDCG@10 on scifact within ±0.03 — the
documented bm25s-vs-Lucene drift.

### The pre-flight invariant

Before any benchmark row is emitted, `trivium-benchmark` runs the
BM25 nDCG@10 equivalence check on the untouched 5K scifact corpus:

- If the observed value is below `0.62`, the pipeline is broken —
  exit 1.
- If `|observed - 0.6789| > 0.03`, the runner prints a warning
  and continues (the drift is documented).
- Otherwise the runner prints `PASS`.

This check is implemented as a pytest golden marker and runs in CI.
Any refactor that changes the tokenizer, BM25 hyperparameters, or
qrels prefix will fail the suite.

## Hybrid RRF

Reciprocal Rank Fusion (Cormack et al. 2009):

```text
score(d) = sum_i w_i / (k + rank_i(d))
```

The runner sweeps:

- `k` ∈ {10, 30, 60, 100, 200} (default 60).
- `w_bm25, w_vec` ∈ {(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)}.

The headline table reports the top row per scale (the
Pareto-frontier row).

## Hybrid rerank

```text
bm25_results  = bm25.search(queries, k=100)
vector_results = vector.search(query_vecs, k=100)
fused         = Rrf().fuse([bm25_results, vector_results], top_k=50)
candidates    = [doc_lookup[h.doc_id] for h in fused][:50]
final         = cross_encoder.rerank(query, candidates, top_k=10)
```

The default cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`
(90 MB, +180 ms p95 per query at batch 32, max_length 256, 50
candidates).

The published estimate was 20-40 ms; the measured value is 5-10×
higher. The published number was wrong; updating it is a finding.

## DiskBBQ

The `Bbq` retriever is an algorithmic analog of Google's DiskBBQ
(coarse IVF + binary-like PQ + exact refinement). It is RAM-resident
in this benchmark. DiskBBQ's actual advantage is at 100M+ vectors
where the index cannot fit in RAM.

Three search modes:

- `ram`: in-memory arrays (fastest at small scale).
- `disk`: lazy-load only the top-`nprobe` lists via
  `ThreadPoolExecutor`.
- `auto`: `disk` if the on-disk footprint exceeds
  `auto_disk_threshold_bytes`; else `ram`.

## Reproducibility

Every CSV row carries the `python`, `platform`, `numpy`, `faiss`,
`sentence_transformers`, `torch`, `bm25s`, `faiss_omp_threads`,
and `git_sha`. `ReproducibilityManifest.gather()` runs at the end
of every benchmark invocation and merges its fields into each
row.

## Honest limitations

This is the 0.1.0 release and the headline numbers are
reproducible on the published commit. The benchmark does NOT
measure:

- E5-Mistral-7B (14 GB, 4096d) ceiling (`0.749` nDCG@10 dense).
- BGE-large-en-v1.5 (1.34 GB, 1024d) ceiling (`0.741`).
- MonoT5-3B rerank ceiling (`0.777`).
- The vector index is RAM-resident; DiskBBQ's actual advantage
  is at 100M+ vectors.
- bm25s's `lucene` method differs from Anserini's Lucene BM25 by
  roughly 0.02-0.03 nDCG@10 on small corpora.

These are listed explicitly in the README and CHANGELOG so
no reader is misled.
