---
layout: page
title: Tune the vector index
subtitle: The IVFPQ + RFlat knobs, nlist formula, and the nprobe sweep.
section: How-to
permalink: /docs/guides/tune-vector/
prev:
  title: Tune BM25
  url: /docs/guides/tune-bm25/
next:
  title: Config schema
  url: /docs/reference/config/
---

The vector index in trivium is FAISS
`IndexOPQ48,IVF{nlist},PQ48x4fs,RFlat`. There are five knobs:

- `vector.nlist_by_scale` — coarse-quantizer count per scale.
- `vector.nprobe` — number of coarse cells visited per query.
- `vector.m` — sub-quantizer count.
- `vector.nbits` — bits per sub-quantizer.
- `vector.k_factor` — RFlat refinement factor.

## When does the vector index run?

The pipeline chooses the index type by scale:

| scale    | index type                   |
|----------|------------------------------|
| ≤ 5K     | `IndexFlatIP` (exact)        |
| > 5K     | `IndexOPQ48,IVF{nlist},PQ48x4fs,RFlat` |

The threshold is configurable via
`vector.min_scale_for_ivfpq` in the config schema. At the 5K
scale the OPQ's internal k-means needs more training points
than the corpus provides, so we fall back to the exact dense
ceiling.

## The `4·sqrt(N)` recipe for `nlist`

`nlist` follows the FAISS wiki recipe: `4·sqrt(N)` rounded up to
the next power of two. The literal values are in
`trivium/configs/default.yaml`:

| scale       | `4·sqrt(N)` | rounded to power of 2 |
|------------:|------------:|----------------------:|
| 5_000       | 283         | 512                   |
| 100_000     | 1_265       | 2_048                 |
| 500_000     | 2_828       | 4_096                 |
| 1_000_000   | 4_000       | 4_096                 |

The 1M scale is intentionally capped at 4_096 (the next power of
two would be 8_192) so coarse-quantizer training stays `O(N)`.

If your corpus has a different dimensional profile, edit
`nlist_by_scale`:

```yaml
vector:
  nlist_by_scale:
    5000: 512
    100000: 2048
    500000: 4096
    1000000: 4096
    5000000: 8192    # new scale
```

## The `nprobe` sweep

`vector.nprobe` is a list, not a scalar. The runner sweeps every
value and keeps the Pareto-frontier row per `(scale, mode,
encoder)` combination.

```yaml
vector:
  nprobe: [8, 16, 32, 64, 128, 256]
```

The default sweep covers the practical operating range:
- `nprobe=8` — fast but recall-poor at large scale.
- `nprobe=64` — typical Pareto-frontier value.
- `nprobe=256` — recall-saturated; only useful for `nprobe-only`
  ablation runs.

Use `--nprobe-only 32` on the CLI to restrict the sweep to a
single value (faster iteration when you're tuning something else).

## The `m` and `nbits` knobs

- `m` — number of sub-quantizers. The default `48` matches the
  384-d MiniLM-L6 dimensions (384 / 48 = 8 dims per sub-quantizer).
- `nbits` — bits per sub-quantizer. The default `4` gives
  ~55 bytes/vec (roughly `m * nbits / 8`).

Higher `m` means more accurate but slower quantizer training.
Lower `nbits` means smaller index but more recall loss.

## The RFlat `k_factor`

`k_factor` is how many approximate candidates RFlat rescores
exactly. The default `4` means: if you ask for top-`k`, the index
returns top-`4k` from IVFPQ, then reranks them with exact
inner-product rescoring.

```yaml
vector:
  k_factor: 4
```

Trade-off: higher `k_factor` recovers more recall at the cost of
more exact inner products. The FAISS recommendation is `4`; we
keep that as the default.

## When to switch to IndexFlatIP

For corpus sizes under ~10K, OPQ-IVFPQ has nothing to gain because
the coarse quantizer needs more training points than the corpus
provides. The pipeline falls back to `IndexFlatIP` (exact dense)
automatically for `scale <= vector.min_scale_for_ivfpq`.

If you want to force `IndexFlatIP` at a larger scale (for example,
to validate that the approximate index isn't hurting recall),
set:

```yaml
vector:
  min_scale_for_ivfpq: 1000000    # 1M
```

Then every scale ≤ 1M uses exact dense.

## The `train_size_strategy` knob

OPQ-IVFPQ training needs ~50× `nlist` vectors to converge. For
small scales we sample from the corpus itself:

```yaml
vector:
  train_size_strategy: 50_x_nlist   # default
  train_size_fixed: 30000            # used when strategy = fixed_N
```

Options:

- `sqrt_n` — sample `sqrt(N)` vectors. Conservative for small
  corpora.
- `50_x_nlist` — sample `max(50 * nlist, train_size_fixed)`. The
  FAISS recommendation.
- `fixed_N` — sample exactly `train_size_fixed` vectors. Useful
  for reproducibility.

## Verify with the summariser

```bash
trivium-summarise --csv results/benchmark.csv
```

The `vector` rows show the nDCG@10 vs the exact dense ceiling at
each scale. The IVFPQ+RFlat row should be within 0.05 nDCG of the
`IndexFlatIP` row at the 5K scale.

## Diagnostics

If vector nDCG is much lower than BM25 nDCG at a given scale:

1. Check the `nlist` value at that scale. A too-small `nlist`
   means each cell holds too many vectors; a too-large `nlist`
   means each cell holds too few and the coarse quantizer
   over-fits.
2. Sweep `nprobe`. The Pareto-frontier value is usually between
   32 and 128.
3. Increase `k_factor`. Values up to `8` are reasonable for
   sub-100K corpora.
4. Switch to `IndexFlatIP` (set `min_scale_for_ivfpq`) and check
   that recall recovers. If yes, the index is the bottleneck.

## Next steps

- Read [Config schema](../../reference/config/) for the full
  schema.
- Read [Methodology](../../methodology/) for the experimental
  design.
