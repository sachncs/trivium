"""CSV output for PipelineResult lists.

Column order is sorted for diff-friendly CSVs. The union of keys
 across all rows becomes the column set; rows missing a key
 output as empty cells, keeping generated benchmark reports
 deterministic and diff-friendly.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from trivium.domain.pipeline_result import PipelineResult


def write_csv(rows: Iterable[PipelineResult], path: str | Path) -> None:
    """Write PipelineResult rows with stable columns and an atomic rename."""
    rows = list(rows)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.to_dict()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=path.parent, suffix=".csv", delete=False
    ) as stream:
        temp_path = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
