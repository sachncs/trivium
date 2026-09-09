# Changelog

All notable changes to trivium are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-09

### Added

- Hybrid search benchmark over BM25 + dense vector + cross-encoder
  rerank on distractor-augmented BEIR scifact.
- Five pipeline modes: `bm25`, `vector`, `hybrid_rrf`, `hybrid_rerank`,
  `diskbbq`, `hybrid_bbq`.
- Embedder registry with five first-party encoders: `minilm-l6`,
  `bge-small`, `bge-base`, `bge-large`, `mpnet-base`, and `e5-mistral-7b`.
- Reranker registry with four first-party cross-encoders: `encoder`
  (MiniLM-L-6), `bge-large`, `bge-base`, `monot5-3b`.
- Pydantic v2 config schema with `extra="forbid"` so typos in
  `configs/default.yaml` fail at load time, not deep in a benchmark
  row.
- Reproducibility manifest: per-row capture of Python, OS, numpy,
  faiss, sentence-transformers, torch, bm25s, pydantic, faiss OMP
  thread count, and git SHA.
- Cached embedding cache keyed on model revision hash.
- Console scripts: `trivium-benchmark`, `trivium-prepare`,
  `trivium-summarise`, `trivium-gigatoken-bench`.
- BM25 pre-flight equivalence test against the Anserini baseline
  (`0.6789 ± 0.03`, observed `0.65685`).
- GitHub Actions CI on Python 3.12.

### Fixed

- `pip install trivium` now produces a working wheel and console
  scripts (the previous `[tool.setuptools.package-data]` empty-string
  key broke the build under setuptools ≥ 68).
- DiskBBQ RAM-mode RFlat rescoring now preserves the inner-product
  score (was hard-coded to 0).
- `Bm25.save()` round-trips `doc_ids` across reload (previously the
  bm25s format had no place for them; reload returned zero hits).
- `--encoders a,b` runs both encoders (the inner `break` truncated
  to the first).
- `hybrid_rerank` mode receives the reranker instance end-to-end.
- `vector.nlist_by_scale` is now explicit and formula-derived.
- Reproducibility manifest probes package versions through a
  subprocess so a faulty native import cannot abort the runner.

### Changed

- Moved `scripts/`, `data_prep/`, and `configs/default.yaml` into
  the `trivium` package so a single wheel install makes everything
  importable.
- Renamed informal internal jargon ("Ponytail") to standard
  terminology throughout README and config.
- README hero rewritten to answer the four landing-page questions
  (what, audience, why, next action) above the data table.

### Honest limitations

This is the 0.1.0 release and the headline numbers below are the
ones from the in-repo `results/benchmark.csv`. They are reproducible
on the published commit; see `README.md §Reproducibility` for the
exact env.

| scale | mode           | nDCG@10 | R@10  | R@100 | p50 ms | p95 ms |
|------:|----------------|--------:|------:|------:|-------:|-------:|
| 5K    | bm25           | 0.6569  | 0.7784 | 0.8759 |   0.8  |   0.9 |
| 5K    | vector (flat)  | 0.6451  | 0.7833 | 0.9250 |   0.1  |   0.1 |
| 5K    | **hybrid_rrf** | **0.7016** | **0.8482** | **0.9550** | 1.0 | 1.1 |
| 5K    | hybrid_rerank  | 0.6820  | 0.8062 | 0.9303 | 165.7 | 167.3 |
| 100K  | bm25           | 0.3206  | 0.3793 | 0.4313 |   5.6  |   6.5 |
| 100K  | vector (IVFPQ+RFlat) | 0.1674 | 0.1971 | 0.2104 | 0.04 | 0.05 |
| 100K  | hybrid_rrf     | 0.3232  | 0.3796 | 0.4422 |   5.9  |   6.8 |
| 100K  | hybrid_rerank  | 0.3362  | 0.3871 | 0.4148 | 183.1 | 187.7 |

What these numbers do **not** show:

- The MiniLM-L6 (22 M parameters) encoder is the recall ceiling at
  scale. E5-Mistral-7B (`0.749` nDCG@10 dense, `0.777` reranked)
  and BGE-large-en-v1.5 (`0.741`) are the published scifact
  ceilings; we do not run those models in this benchmark.
- The vector index is RAM-resident. DiskBBQ's actual advantage is
  at 100M+ vectors where the index cannot fit in RAM.
- bm25s's `lucene` method differs from Anserini's Lucene BM25 by
  roughly `0.02-0.03` nDCG@10 on small corpora. Our `0.6569` vs
  Anserini's `0.6789` is the documented implementation drift.

[0.1.0]: https://github.com/sachncs/trivium/releases/tag/v0.1.0
