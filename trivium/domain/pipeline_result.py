"""PipelineResult: the row-shaped return value of every BenchmarkPipeline.run().

The to_dict() method produces a stable-schema dict suitable for
the CSV writer. Column set is locked by the format_version so
schema bumps are explicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trivium.evaluation.latency import LatencyStats
from trivium.evaluation.metrics import EvaluationMetrics


FORMAT_VERSION = 2


@dataclass
class PipelineResult:
    """One row of benchmark output."""

    name: str
    scale: int
    encoder: str
    reranker: str
    metrics: EvaluationMetrics
    latency: LatencyStats
    extras: dict = field(default_factory=dict)
    format_version: int = FORMAT_VERSION

    def to_dict(self) -> dict:
        """Stable column order, sorted alphabetically for diff-friendly CSV."""
        out = {
            "format_version": self.format_version,
            "mode": self.name,
            "scale": self.scale,
            "encoder": self.encoder,
            "reranker": self.reranker,
            "lat_n": self.latency.n,
            "lat_mean_ms": round(self.latency.mean_ms, 4),
            "lat_p50_ms": round(self.latency.p50_ms, 4),
            "lat_p95_ms": round(self.latency.p95_ms, 4),
            "lat_p99_ms": round(self.latency.p99_ms, 4),
            "lat_p999_ms": round(self.latency.p999_ms, 4),
            "ndcg_cut.10": round(self.metrics.ndcg_at_10, 5),
            "recall.10": round(self.metrics.recall_at_10, 5),
            "recall.100": round(self.metrics.recall_at_100, 5),
        }
        for k, v in sorted(self.extras.items()):
            if k not in out:
                out[k] = v
        return out
