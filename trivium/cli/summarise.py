#!/usr/bin/env python3
"""Print headline tables from a benchmark CSV.

Usage:
    trivium-summarise --csv results/benchmark.csv

Output is the per-(scale, mode, encoder) pivot that the README
references. summariser is the single source of truth — README
headline numbers are produced from this script, not hand-edited.
"""
from __future__ import annotations

import argparse

from trivium.reporting.summariser import summarise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    result = summarise(args.csv)
    headline = result["headline"]

    print(f"\n{'scale':>7} {'mode':<14} {'encoder':<14} {'ndcg@10':>9} {'R@10':>7} {'p95_ms':>9}")
    print("-" * 70)
    for key in sorted(headline.keys()):
        scale, mode, encoder = key
        v = headline[key]
        print(f"{scale:>7} {mode:<14} {encoder:<14} {v['ndcg_cut.10']:>9.4f} {v['recall.10']:>7.4f} {v['p95_ms']:>9.2f}")


if __name__ == "__main__":
    main()
