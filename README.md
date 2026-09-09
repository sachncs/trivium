# trivium

> A reproducible benchmark for hybrid search engines — BM25 + dense
> vector + cross-encoder rerank — on distractor-augmented BEIR scifact.

For retrieval engineers comparing fusion strategies, for ML researchers
validating new encoders, and for open-source maintainers who want to
publish numbers they can defend.

[![CI](https://github.com/sachncs/trivium/actions/workflows/ci.yml/badge.svg)](https://github.com/sachncs/trivium/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/sachncs/trivium)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-101%20passed-brightgreen)](tests/)

## Quick start

```bash
pip install trivium
trivium-prepare --max-scales 100000
trivium-benchmark --modes bm25,vector,hybrid_rrf,hybrid_rerank \
                  --scales 5000,100000 \
                  --output results/benchmark.csv
trivium-summarise --csv results/benchmark.csv
```

The benchmark pulls ~6 GB of BEIR corpora and ~520 MB of Python
dependencies on first run. Allow 5 minutes on a laptop for the
default 5K + 100K sweep.

## What this is

A small, opinionated benchmark for hybrid search. Each pipeline mode
is a `@register_pipeline` decorator over a retriever (or two
retrievers plus a fusion strategy plus a reranker). Adding a new
encoder, reranker, or pipeline mode is one decorator call.

The headline numbers below are reproducible from a single
`trivium-benchmark` invocation against the corpus cached by
`trivium-prepare`.

## Headline results

| scale | mode             | nDCG@10 | R@10  | R@100 | p50 ms | p95 ms |
|------:|------------------|--------:|------:|------:|-------:|-------:|
| 5K    | bm25             | 0.6569  | 0.7784 | 0.8759 |   0.8  |   0.9 |
| 5K    | vector (flat)    | 0.6451  | 0.7833 | 0.9250 |   0.1  |   0.1 |
| 5K    | **hybrid_rrf**   | **0.7016** | **0.8482** | **0.9550** | 1.0 | 1.1 |
| 5K    | hybrid_rerank    | 0.6820  | 0.8062 | 0.9303 | 165.7 | 167.3 |
| 100K  | bm25             | 0.3206  | 0.3793 | 0.4313 |   5.6  |   6.5 |
| 100K  | vector (IVFPQ+RFlat) | 0.1674 | 0.1971 | 0.2104 | 0.04 | 0.05 |
| 100K  | hybrid_rrf       | 0.3232  | 0.3796 | 0.4422 |   5.9  |   6.8 |
| 100K  | hybrid_rerank    | 0.3362  | 0.3871 | 0.4148 | 183.1 | 187.7 |

Source: `results/benchmark.csv` (38 rows; ablations over RRF `k`,
weight sweep, and IVFPQ `nprobe`).

### What this shows

- **At 5K** (untouched scifact, ~5,183 docs): hybrid RRF beats BM25
  by **+0.045 nDCG@10** and pure vector by **+0.057**. This is the
  headline: hybrid wins on the smallest test set.
- **At 100K** (scifact + 95K scientific distractors): the nDCG
  ceiling collapses for all methods because the MiniLM-L6 (22 M
  parameter) encoder hits its recall ceiling when gold documents
  are scattered in 95K distractors. Hybrid RRF is essentially tied
  with BM25; the rerank adds a small lift (+0.015).
- **The encoder is the bottleneck at scale**, not the index. The
  exact dense ceiling at 100K is 0.2957; IVFPQ+RFlat drops to 0.1674.
  A stronger encoder (BGE-large, mpnet-base) would reduce this gap.

## What this is NOT

- **Not a "SOTA" benchmark.** E5-Mistral-7B (`0.749`), BGE-large
  (`0.741`), and MonoT5-3B rerank (`0.777`) are the published
  scifact ceilings. We do not run those models.
- **Not a disk-aware benchmark.** The vector index is RAM-resident.
  DiskBBQ's actual advantage is at 100M+ vectors where the index
  cannot fit in RAM.
- **Not a pre-flight-passing BM25 equivalence.** bm25s
  implementations differ from Anserini's Lucene BM25 by ~0.02-0.03
  nDCG@10 on small corpora. Our 0.6569 vs Anserini's 0.6789 is the
  documented implementation drift.

## What you can build

The benchmark is the reference workload; the public API is what
you build with.

```python
from trivium.retrieval.bm25 import Bm25
from trivium.retrieval.faiss import Faiss
from trivium.fusion.rrf import Rrf
from trivium.reranking.registry import get_reranker

bm25 = Bm25()
bm25.add_documents(your_docs)

vec = Faiss.Flat()
vec.add_documents(your_docs, vectors=your_vectors)

bm25_results = bm25.search(queries, k=100)
vec_results = vec.search(query_vecs, k=100)

fused = Rrf(k=60).fuse([bm25_results, vec_results], top_k=50)
candidates = [doc_lookup[h.doc_id] for h in fused]

reranker = get_reranker("bge-large")
final = reranker.rerank(query_text, candidates, top_k=10)
```

See the [documentation site](https://sachncs.github.io/trivium/) for
the architecture overview, the embedder/reranker/pipeline registries,
and tutorials.

## Installation

```bash
pip install trivium
```

Dependencies (~520 MB on first install): `faiss-cpu`, `numpy`,
`scipy`, `bm25s`, `beir`, `datasets`, `sentence-transformers`,
`torch`, `pytrec_eval`, `pyyaml`, `pydantic`, `datasketch`,
`gigatoken`.

For development:

```bash
git clone https://github.com/sachncs/trivium
cd trivium
pip install -e ".[dev]"
pytest -q
```

## Data preparation

```bash
trivium-prepare --max-scales 100000
```

Builds a distractor-augmented scifact corpus with scientific
distractors from SCIDOCS, TREC-COVID, and NFCorpus. Embeds the
corpus with `all-MiniLM-L6-v2` (384d, l2-normalized). Cache lives
under `data/cache/`.

## Run the benchmark

```bash
trivium-benchmark --scales 5000,100000 \
                  --modes bm25,vector,hybrid_rrf,hybrid_rerank

# Restrict the IVFPQ nprobe sweep
trivium-benchmark --modes vector --nprobe-only 32

# Skip the BM25 pre-flight equivalence check (not recommended)
trivium-benchmark --skip-preflight
```

### Pre-flight check

`trivium-benchmark` runs `BM25 nDCG@10 on the untouched 5K scifact`
first. It must hit `0.6789 ± 0.03` (the Anserini published baseline;
`±0.03` is the documented bm25s-vs-Lucene drift). If the observed
value is below `0.62`, the pipeline is broken — fail fast.

## Methodology

### Corpus

- **5K seed**: BEIR scifact (`beir-v1.0.0-scifact`), 5,183 docs,
  300 test queries, qrels unchanged. Loaded from the canonical
  BEIR zip via `beir.util.download_and_unzip`.
- **Distractors**: SCIDOCS (25.7K) + TREC-COVID (171K) + NFCorpus
  (3.6K), all BEIR-format scientific corpora. No MS MARCO (domain
  mismatch — web passages vs scientific abstracts).
- **Dedup**: SHA-256 over `title + text` for exact matches; MinHash
  LSH (threshold 0.85) for near-duplicates.
- **Naming**: `scifact:<id>` for gold docs, `distractor:<id>` for
  distractors. Strict prefix prevents ID collisions.
- **Seeded shuffle**: `random.Random(42)`. 100K is a true prefix
  of 1M so scale curves are comparable.
- **Qrels**: untouched. Closed-world assumption: unjudged
  distractors are treated as non-relevant (Voorhees 2005 pooling
  convention).

### Embeddings

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384d, l2-normalized).
- Trade-off: 5× faster than `mpnet-base-v2` on CPU at the cost of
  ~0.03 nDCG@10 vs BGE-large-en-v1.5.
- `IndexFlatIP` = cosine after l2-norm.

### Vector index

- `IndexOPQ48,IVF{nlist},PQ48x4fs,RFlat` where `nlist` follows the
  `4·sqrt(N)` formula rounded up to a power of two. The literal
  values live in `trivium/configs/default.yaml` under
  `vector.nlist_by_scale`.
- nprobe sweep: 8, 16, 32, 64, 128, 256. Pareto frontier over
  this grid.
- k_factor = 4 for RFlat (4× more candidates rescored exactly).
- **At 5K**: uses `IndexFlatIP` (exact dense) because OPQ's
  internal k-means needs more training points than 5K provides.

### BM25

- `bm25s` with `method="lucene"`, `k1=0.9`, `b=0.4` (Anserini
  defaults).
- Reproduces the 0.6789 BEIR baseline within ±0.03 (the documented
  bm25s-vs-Lucene implementation drift).

### Hybrid RRF

- Reciprocal Rank Fusion: `score(d) = sum_i w_i / (k + rank_i(d))`.
- Cormack et al. 2009. k=60 default; sweep over {10, 30, 60, 100,
  200}.
- Weight sweep: BM25 vs vec ∈ {(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)}.

### Hybrid Rerank

- BM25 + vector → RRF → top-50 → cross-encoder top-10.
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (90 MB).
- CPU latency: ~180 ms p95 per query (batch=32, max_length=256,
  50 candidates). The original 20-40 ms estimate was 5-10× too
  low.

### Metrics

- `pytrec_eval.RelevanceEvaluator` directly. BEIR semantics: macro
  average, log₂, `ignore_identical_ids=True`, `2^rel - 1` gain.
- Latency: `time.perf_counter_ns()` per query, 20-query warmup,
  single-stream.

## Honest claims

- **At 5K untouched scifact**, hybrid RRF beats BM25 by **+0.045
  nDCG@10** and pure vector by **+0.057**. This is a real
  measurement, not a handwaved number.
- **At 100K distractor-augmented**, the nDCG ceiling collapses
  for all methods because the MiniLM-L6 encoder hits its ceiling
  when gold docs are scattered. The hybrid advantage over BM25
  shrinks to a tie; the rerank adds a small lift (+0.015).
- **MiniLM-L6 cross-encoder rerank** gives a small lift at 5K and
  100K but costs ~180 ms p95 per query — 5-10× the original
  estimate. Updating the published number is a finding.
- **DiskBBQ analogy**: this is an algorithmic analog (coarse IVF +
  binary-like PQ + exact refinement), not a disk-resident
  implementation. The DiskBBQ analog is RAM-resident; DiskBBQ's
  actual advantage is at 100M+ vectors.
- **The benchmark does not compare to E5-Mistral-7B, BGE-large,
  or MonoT5-3B** — we don't run those models. The published
  ceiling on scifact is `0.749` nDCG@10 (dense) and `0.777`
  (reranked).

## Files

```
trivium/
├── trivium/
│   ├── cli/             # trivium-benchmark, trivium-prepare, trivium-summarise
│   ├── configs/         # default.yaml (shipped as package data)
│   ├── data_prep/       # corpus, dedup, distractors, embed, fever, seed
│   ├── domain/          # Document, Query, Qrels, Hit, SearchResult
│   ├── embeddings/      # Sentence, E5, base, registry
│   ├── evaluation/      # metrics, latency, ground truth
│   ├── fusion/          # RRF, weighted, registry
│   ├── io/              # corpus cache, embedding cache, manifest
│   ├── pipelines/       # bm25, vector, hybrid_rrf, hybrid_rerank, diskbbq, hybrid_bbq
│   ├── reranking/       # Encoder, Bge, Monot5, registry
│   ├── retrieval/       # Bm25, Faiss, DiskBBQ
│   └── tokenizer/       # GigaToken (Rust-backed)
├── tests/
│   ├── golden/          # BM25 pre-flight anchor
│   ├── retrieval/       # bm25 round-trip, diskbbq recall
│   ├── embeddings/      # registry factories
│   ├── reranking/       # registry factories
│   └── ...
├── configs/             # repo-local checkout of trivium/configs/default.yaml
└── results/             # benchmark.csv (gitignored, regenerate with trivium-benchmark)
```

## Reproducibility

Every CSV row carries: Python version, platform, numpy, faiss,
sentence-transformers, torch, bm25s, faiss_omp_threads, git SHA,
embedding model, scale, mode, and every hyperparameter knob from
`configs/default.yaml`. `ReproducibilityManifest.gather()` is
called at the end of every benchmark run and merged into the
CSV.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Links

- Documentation site: <https://sachncs.github.io/trivium/>
- Source repository: <https://github.com/sachncs/trivium>
- Issue tracker: <https://github.com/sachncs/trivium/issues>
- Changelog: [CHANGELOG.md](CHANGELOG.md)
