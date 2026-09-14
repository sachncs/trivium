---
layout: page
title: Tune BM25
subtitle: BM25 hyperparameters, the pre-flight invariant, and the candidate_pool knob.
section: How-to
permalink: /docs/guides/tune-bm25/
prev:
  title: Add a pipeline
  url: /docs/guides/add-pipeline/
next:
  title: Tune the vector index
  url: /docs/guides/tune-vector/
---

BM25 has three knobs you actually control in trivium:

- `k1` — term-frequency saturation.
- `b` — document-length normalisation.
- `candidate_pool` — how many candidates each pipeline pulls
  before fusion or rerank.

Everything else (stopword list, stemmer, BM25 method) is fixed at
the Anserini defaults to keep the pre-flight invariant meaningful.

## Anserini defaults

`trivium/configs/default.yaml` ships:

```yaml
bm25:
  method: lucene
  k1: 0.9
  b: 0.4
  candidate_pool: 100
```

These values reproduce the BEIR published baseline of
`0.6789` nDCG@10 on scifact within ±0.03 — the documented bm25s-vs-Lucene
drift. The pre-flight check enforces this; if you change `k1` or
`b`, the pre-flight will fail and the runner will exit 1.

## When to deviate

You almost never want to. The Anserini defaults are the result of
decades of IR research and the BEIR paper picks them because they
work on scientific text.

The one knob you might legitimately tune is `candidate_pool`:

- **Smaller (`50`)** — lower latency for `hybrid_rerank` because
  the cross-encoder sees fewer candidates. Trade-off: lower
  recall ceiling on the fused list.
- **Larger (`200`)** — higher recall ceiling. Trade-off: cross-
  encoder gets slower (latency scales linearly with candidates).

The default `100` matches the BEIR paper's typical retrieval
depth.

## How to change them

Edit `trivium/configs/default.yaml`:

```yaml
bm25:
  method: lucene
  k1: 0.9
  b: 0.4
  candidate_pool: 200     # was 100
```

Or pass `--config path/to/your.yaml` on the CLI.

If you change `k1` or `b`, also update
`trivium/configs/default.yaml`'s `preflight.target_ndcg10` to the
new observed baseline.

## The pre-flight invariant

`trivium-benchmark` runs this check before the first CSV row:

```python
observed = bm25.evaluate(scifact_seed_5k).ndcg_at_10
assert observed >= 0.62, "pipeline broken"
assert abs(observed - 0.6789) <= 0.03, "out of drift band"
```

- `0.62` is the absolute lower bound. Below this is a real bug
  (wrong tokenizer, wrong qrels prefix, wrong `k1`/`b`).
- `0.6789 ± 0.03` is the documented bm25s-vs-Lucene drift band.

You can skip the pre-flight with `--skip-preflight`, but the
golden test in `tests/golden/test_preflight_bm25.py` is still
enforced in CI.

## Verify with the summariser

```bash
trivium-summarise --csv results/benchmark.csv
```

prints the per-`(scale, mode, encoder)` pivot. The `bm25` row at
the 5K scale should always show nDCG@10 in `[0.62, 0.7089]`.

## Diagnostics

If the pre-flight fails:

1. Re-run with `--skip-preflight` to see the per-row output and
   isolate the regression.
2. Check `data/cache/manifest.json` — the `corpus_sha256` must
   match the published commit if you're trying to reproduce the
   headline numbers.
3. Check the BM25 hyperparameters: `python -c "from trivium.config.loader import load_config; print(load_config().bm25)"`.

## Next steps

- Read [Tune the vector index](../tune-vector/) for the IVFPQ + RFlat
  knobs.
- Read the [config schema](../../reference/config/) for every
  config field.
