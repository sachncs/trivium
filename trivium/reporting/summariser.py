"""Summariser: produce README-style headlines from a benchmark CSV.

Used by scripts/summarise.py and called by V7 to verify that the
CSV exactly matches the published numbers in the README.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarise(csv_path: str | Path) -> dict[str, Any]:
    """Read the CSV and produce per-(scale, encoder, mode) pivot tables.

    Returns a dict with:
        - headline: {(scale, mode, encoder): {ndcg_cut.10, recall.10, p95_ms}}
        - per_encoder: {encoder: [(scale, mode, ndcg_cut.10)]}
    """
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    headline = {}
    for r in rows:
        key = (
            int(r.get("scale", 0)),
            r.get("mode", ""),
            r.get("encoder", ""),
        )
        ndcg = safe_float(r.get("ndcg_cut.10"))
        recall = safe_float(r.get("recall.10"))
        p95 = safe_float(r.get("lat_p95_ms"))
        headline[key] = {"ndcg_cut.10": ndcg, "recall.10": recall, "p95_ms": p95}

    per_encoder = defaultdict(list)
    for r in rows:
        per_encoder[r.get("encoder", "")].append(
            (int(r.get("scale", 0)), r.get("mode", ""), safe_float(r.get("ndcg_cut.10")))
        )

    return {"headline": headline, "per_encoder": dict(per_encoder)}


def safe_float(value) -> float:
    """Parse a CSV cell into a float; default to 0.0 on failure."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
