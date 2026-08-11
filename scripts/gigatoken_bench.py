#!/usr/bin/env python3
"""Standalone gigatoken throughput benchmark.

Measures how fast the trivium.tokenizer.GigaToken wrapper
(backed by gigatoken's Rust engine) tokenises the corpus
loaded via CorpusCache. Reports a single GigaTokenSummary
per (scale, tokenizer) pair so CSV readers can pick it up.

Run:
    python -m scripts.gigatoken_bench --scales 5000,100000
    python -m scripts.gigatoken_bench --tokenizer gpt2 --output /tmp/gigabench.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from trivium.config.loader import load_config
from trivium.io.corpus import CorpusCache
from trivium.tokenizer.gigatoken import GigaToken

CACHE = Path(__file__).parent.parent / "data" / "cache"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).parent.parent / "configs" / "default.yaml"))
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--scales", default=None)
    p.add_argument("--output", default=str(Path(__file__).parent.parent / "results" / "gigatoken_benchmark.csv"))
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.scales:
        cfg.scales = [int(s) for s in args.scales.split(",")]

    cache = CorpusCache(CACHE)
    gt = GigaToken(args.tokenizer)

    rows = []
    for scale in cfg.scales:
        documents = cache.load_scale(scale)
        if not documents:
            print(f"  scale {scale}: empty (cache not populated)", flush=True)
            continue
        summary = gt.count_streaming((d.body for d in documents), batch_size=8192)
        print(
            f"  scale {scale:>9,}: {summary.total_tokens:>15,} tokens  "
            f"{summary.doc_count:>9,} docs  "
            f"{summary.tokens_per_second:>12,.0f} tok/s  "
            f"{summary.bytes_per_second:>10,.0f} B/s  "
            f"vocab={summary.vocab_size}",
            flush=True,
        )
        rows.append(
            {
                "tokenizer": args.tokenizer,
                "scale": scale,
                "doc_count": summary.doc_count,
                "total_tokens": summary.total_tokens,
                "vocab_size": summary.vocab_size,
                "bytes_processed": summary.bytes_processed,
                "seconds_elapsed": round(summary.seconds_elapsed, 3),
                "tokens_per_second": round(summary.tokens_per_second, 1),
                "bytes_per_second": round(summary.bytes_per_second, 1),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
