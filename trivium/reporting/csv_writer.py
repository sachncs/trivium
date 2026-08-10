"""CsvResultWriter: stable-schema writer for PipelineResult lists.

Column order is sorted for diff-friendly CSVs. The union of keys
across all rows becomes the column set; rows missing a key
output as empty cells. This is what makes V7 (backward-compat
CSV diff) tractable.
"""
from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from trivium.domain.pipeline_result import PipelineResult


class CsvResultWriter:
    """Write PipelineResult lists to a CSV with stable column order."""

    @staticmethod
    def write(rows: Iterable[PipelineResult], path: str | Path) -> None:
        rows = list(rows)
        if not rows:
            return
        keys = sorted({k for r in rows for k in r.to_dict()})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r.to_dict())
