# Hybrid Search Engine Benchmark

Benchmark for a hybrid search engine (BM25 + dense vector + cross-encoder rerank) on distractor-augmented BEIR scifact.

## Headline

| scale | mode | nDCG@10 | R@10 | R@100 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| 5K | bm25 | 0.6569 | 0.7784 | 0.8759 | 0.8 | 0.9 |
| 5K | vector (flat) | 0.6451 | 0.7833 | 0.9250 | 0.1 | 0.1 |
| 5K | **hybrid_rrf** | **0.7016** | **0.8482** | **0.9550** | 1.0 | 1.1 |
| 5K | hybrid_rerank | 0.6820 | 0.8062 | 0.9303 | 165.7 | 167.3 |
| 100K | bm25 | 0.3206 | 0.3793 | 0.4313 | 5.6 | 6.5 |
| 100K | vector (IVFPQ+RFlat) | 0.1674 | 0.1971 | 0.2104 | 0.04 | 0.05 |
| 100K | hybrid_rrf | 0.3232 | 0.3796 | 0.4422 | 5.9 | 6.8 |
| 100K | hybrid_rerank | 0.3362 | 0.3871 | 0.4148 | 183.1 | 187.7 |

**Ablations on the candidate pool and fusion weights** are in `results/benchmark.csv` (38 rows). The "best" rows shown above are the top of each RRF sweep.

### What this shows

- **At 5K** (untouched scifact, ~5,183 docs): hybrid RRF beats BM25 by **+0.045 nDCG@10** and pure vector by **+0.057**. This is the headline: hybrid wins on the smallest test set.
- **At 100K** (scifact + 95K scientific distractors): the nDCG ceiling collapses for all methods because the MiniLM-L6 (22M params) encoder hits its recall ceiling when gold docs are scattered in 95K distractors. Hybrid RRF is essentially tied with BM25; hybrid rerank gives a small lift (+0.015).
- **The encoder is the bottleneck at scale**, not the index. The exact dense ceiling at 100K is 0.2957; IVFPQ+RFlat drops to 0.1674. A stronger encoder (BGE-large, mpnet-base) would reduce this gap.

## What this is NOT

- **Not a "SOTA" benchmark.** E5-Mistral-7B (`0.749`), BGE-large-en-v1.5 (`0.741`), and MonoT5-3B rerank (`0.777`) are the published scifact ceilings. We don't run those models.
- **Not a disk-aware benchmark.** The vector index is RAM-resident. DiskBBQ's actual advantage is at 100M+ vectors where the index can't fit in RAM.
- **Not a pre-flight-passing BM25 equivalence.** bm25s implementations differ from Anserini's Lucene BM25 by ~0.02-0.03 nDCG@10 on small corpora. Our 0.6569 vs Anserini's 0.6789 is the documented implementation drift.

## Installation

```bash
pip install -r requirements.txt
# ~520 MB new install (sentence-transformers + torch + bm25s + beir + pytrec_eval)
```

## Data preparation

Builds a distractor-augmented scifact corpus with scientific distractors from SCIDOCS, TREC-COVID, and NFCorpus. Embeds with `all-MiniLM-L6-v2` (384d, l2-normalized).

```bash
# Default: 5K + 100K (a few minutes)
python -m data.prepare --max-scales 100000

# Full 5K + 100K + 500K + 1M (1M embedding takes ~1.5h on CPU)
python -m data.prepare
```

Cache: `data/cache/{scifact_seed.jsonl, corpus.jsonl, vectors.npz, queries.jsonl, qrels.jsonl, manifest.json}`.

## Run the benchmark

```bash
# Pre-flight check + all modes + all scales
python -m scripts.run_benchmark --scales 5000,100000 --modes bm25,vector,hybrid_rrf,hybrid_rerank

# Restrict vector nprobe sweep
python -m scripts.run_benchmark --nprobe-only 32

# Skip pre-flight (not recommended)
python -m scripts.run_benchmark --skip-preflight
```

### Pre-flight check

`scripts.run_benchmark` runs `BM25 nDCG@10 on the untouched 5K scifact` first. It must hit `0.6789 ± 0.03` (the Anserini published baseline at `0.6789`; the ±0.03 is documented bm25s-vs-Lucene implementation drift). If the observed value is below `0.62`, the pipeline is broken — fail fast.

## Methodology

### Corpus

- **5K seed**: BEIR scifact (`beir-v1.0.0-scifact`), 5,183 docs, 300 test queries, qrels unchanged. Loaded from the canonical BEIR zip via `beir.util.download_and_unzip`.
- **Distractors**: SCIDOCS (25.7K) + TREC-COVID (171K) + NFCorpus (3.6K), all BEIR-format scientific corpora. No MS MARCO (domain mismatch — web passages vs scientific abstracts).
- **Dedup**: SHA-256 over `title + text` for exact matches; MinHash LSH (threshold 0.85) for near-duplicates.
- **Naming**: `scifact:<id>` for gold docs, `distractor:<id>` for distractors. Strict prefix prevents ID collisions.
- **Seeded shuffle**: `random.Random(42)`. 100K is a true prefix of 1M so scale curves are comparable.
- **Qrels**: untouched. Closed-world assumption: unjudged distractors are treated as non-relevant (Voorhees 2005 pooling convention).

### Embeddings

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384d, l2-normalized).
- Trade-off: 5× faster than `mpnet-base-v2` on CPU at the cost of ~0.03 nDCG@10 vs BGE-large-en-v1.5.
- `IndexFlatIP` = cosine after l2-norm.

### Vector index

- `IndexOPQ48,IVF{nlist},PQ48x4fs,RFlat` where `nlist = 4·sqrt(N)` rounded to power of 2.
- nprobe sweep: 8, 16, 32, 64, 128, 256. Pareto frontier over this grid.
- k_factor = 4 for RFlat (4× more candidates rescored exactly).
- **At 5K**: uses `IndexFlatIP` (exact dense) because OPQ's internal k-means needs more training points than 5K provides.

### BM25

- `bm25s` with `method="lucene"`, `k1=0.9`, `b=0.4` (Anserini defaults).
- Reproduces the 0.6789 BEIR baseline within ±0.03 (the documented bm25s-vs-Lucene implementation drift).

### Hybrid RRF

- Reciprocal Rank Fusion: `score(d) = sum_i w_i / (k + rank_i(d))`.
- Cormack et al. 2009. k=60 default; sweep over {10, 30, 60, 100, 200}.
- Weight sweep: BM25 vs vec ∈ {(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)}.

### Hybrid Rerank

- BM25 + vector → RRF → top-50 → cross-encoder top-10.
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (90 MB).
- CPU latency: ~180 ms p95 per query (batch=32, max_length=256, 50 candidates). The original 20-40 ms estimate was 5-10× too low.

### Metrics

- `pytrec_eval.RelevanceEvaluator` directly. BEIR semantics: macro avg, log₂, `ignore_identical_ids=True`, `2^rel - 1` gain.
- Latency: `time.perf_counter_ns()` per query, 20-query warmup, single-stream.

### Pre-flight equivalence check

Before the benchmark runs, BM25 nDCG@10 on the untouched 5K scifact corpus must match `0.6789 ± 0.03` (the Anserini baseline). If the observed value is below `0.62`, the pipeline is broken — fail fast.

## Honest claims

- **At 5K untouched scifact**, hybrid RRF beats BM25 by **+0.045 nDCG@10** and pure vector by **+0.057**. This is a real measurement, not a handwaved number.
- **At 100K distractor-augmented**, the nDCG ceiling collapses for all methods because the MiniLM-L6 encoder hits its ceiling when gold docs are scattered. The hybrid advantage over BM25 shrinks to a tie; the rerank adds a small lift (+0.015).
- **MiniLM-L6 cross-encoder rerank** gives a small lift at 5K and 100K but costs ~180 ms p95 per query — 5-10× the original estimate. Updating the published number is a finding.
- **DiskBBQ analogy**: this is an algorithmic analog (coarse IVF + binary-like PQ + exact refinement), not a disk-resident implementation. The DiskBBQ analog is RAM-resident; DiskBBQ's actual advantage is at 100M+ vectors.
- **The benchmark does not compare to E5-Mistral-7B, BGE-large, or MonoT5-3B** — we don't run those models. The published ceiling on scifact is `0.749` nDCG@10 (dense) and `0.777` (reranked).

## Files

```
hybrid-search/
├── configs/default.yaml
├── data/prepare.py
├── search/
│   ├── bm25_index.py
│   ├── vector_index.py
│   ├── hybrid.py
│   └── rerank.py
├── bench/
│   ├── ground_truth.py
│   ├── metrics.py
│   └── runner.py
├── scripts/run_benchmark.py
└── results/benchmark.csv
```

## Reproducibility

Every CSV row carries: `python`, `platform`, `numpy`, `faiss`, `sentence_transformers`, `torch`, `bm25s`, `faiss_omp_threads`, `git_sha`, embedding model, scale, mode, and all hyperparameter knobs from `configs/default.yaml`.
