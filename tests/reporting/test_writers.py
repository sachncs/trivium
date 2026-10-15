from __future__ import annotations

import csv

from trivium.domain.pipeline_result import PipelineResult
from trivium.evaluation.latency import LatencyStats
from trivium.evaluation.metrics import EvaluationMetrics
from trivium.reporting.csv_writer import write_csv


def test_csv_writer_creates_parent_and_stable_schema(tmp_path):
    row = PipelineResult(
        name="demo",
        scale=2,
        encoder="none",
        reranker="none",
        metrics=EvaluationMetrics(ndcg_at_10=0.5),
        latency=LatencyStats.from_samples([1_000_000, 2_000_000]),
    )
    path = tmp_path / "nested" / "results.csv"

    write_csv([row], path)

    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert records[0]["mode"] == "demo"
    assert records[0]["ndcg_cut.10"] == "0.5"
