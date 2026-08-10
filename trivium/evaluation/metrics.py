"""Evaluator: BEIR semantics via pytrec_eval, frozen metrics dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pytrec_eval

from trivium.domain.qrels import Qrels


@dataclass(frozen=True)
class EvaluationMetrics:
    """A frozen bundle of every measure computed for one evaluation run."""

    ndcg_at_10: float = 0.0
    ndcg_at_100: float = 0.0
    recall_at_10: float = 0.0
    recall_at_100: float = 0.0
    map: float = 0.0
    reciprocal_rank: float = 0.0
    p5: float = 0.0
    p10: float = 0.0
    extras: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extras is None:
            object.__setattr__(self, "extras", {})


_DEFAULT_MEASURES = ("map", "ndcg", "recip_rank")


class Evaluator:
    """Wraps pytrec_eval.RelevanceEvaluator with BEIR semantics.

    BEIR defaults: macro-avg across queries, log2 gain,
    ignore_identical_ids=True, 2^rel - 1 gain. We accept measure
    names in either 'ndcg_cut.10' or 'ndcg_cut_10' form.
    """

    def __init__(self, k_values: Iterable[int] = (10, 100), extra_measures: Iterable[str] = ()) -> None:
        self.k_values = tuple(k_values)
        self.extra_measures = tuple(extra_measures)
        self.measures = (
            set(_DEFAULT_MEASURES)
            | {f"ndcg_cut.{k}" for k in self.k_values}
            | {f"recall.{k}" for k in self.k_values}
            | {f"P.{k}" for k in self.k_values if k >= 5}
            | set(self.extra_measures)
        )

    @staticmethod
    def to_measure_name(name: str) -> str:
        """Convert 'ndcg_cut.10' -> 'ndcg_cut_10' for pytrec_eval."""
        return name.replace(".", "_")

    @staticmethod
    def from_measure_name(name: str) -> str:
        """Inverse of to_measure_name. Inserts the dot before the trailing k-cut digits.

        pytrec_eval reports 'ndcg_cut_10' which means ndcg_cut at k=10. We
        translate that back to the canonical 'ndcg_cut.10' by splitting on
        the last underscore if the tail is all digits, otherwise on the first.
        """
        if "_" in name and name.rsplit("_", 1)[-1].isdigit():
            head, tail = name.rsplit("_", 1)
            return f"{head}.{tail}"
        return name.replace("_", ".", 1) if "_" in name else name

    def evaluate(self, qrels: Qrels, results: dict[str, dict[str, float]]) -> EvaluationMetrics:
        """Run pytrec_eval and return the aggregate metrics.

        Args:
            qrels: A Qrels instance.
            results: Mapping {query_id: {doc_id: score}}.

        Returns:
            EvaluationMetrics with the per-measure macro averages.
        """
        param_measures = {self.to_measure_name(m) for m in self.measures}
        ev = pytrec_eval.RelevanceEvaluator(qrels.to_pytrec(), param_measures)
        per_query = ev.evaluate(results)
        if not per_query:
            return EvaluationMetrics()

        aggregated = self._aggregate(per_query)
        extras = {
            self.from_measure_name(k): v
            for k, v in aggregated.items()
            if self.from_measure_name(k) not in self._core_fields()
        }

        return EvaluationMetrics(
            ndcg_at_10=aggregated.get("ndcg_cut_10", 0.0),
            ndcg_at_100=aggregated.get("ndcg_cut_100", 0.0),
            recall_at_10=aggregated.get("recall_10", 0.0),
            recall_at_100=aggregated.get("recall_100", 0.0),
            map=aggregated.get("map", 0.0),
            reciprocal_rank=aggregated.get("recip_rank", 0.0),
            p5=aggregated.get("P_5", 0.0),
            p10=aggregated.get("P_10", 0.0),
            extras=extras,
        )

    @staticmethod
    def _aggregate(per_query: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        for measure in sorted({k for v in per_query.values() for k in v}):
            vals = [v[measure] for v in per_query.values() if v.get(measure) is not None]
            if vals:
                out[measure] = round(sum(vals) / len(per_query), 5)
        return out

    @staticmethod
    def _core_fields() -> set[str]:
        return {
            "ndcg_cut.10",
            "ndcg_cut.100",
            "recall.10",
            "recall.100",
            "map",
            "recip_rank",
            "P.5",
            "P.10",
        }


def hits_to_results(per_query_hits: list[list[tuple[str, float]]], queries: list) -> dict[str, dict[str, float]]:
    """Convert per-query hit lists into the {qid: {did: score}} shape.

    Public helper (replaces the legacy _hit_lists_to_results). Lives
    in trivium.evaluation because evaluation is its only consumer
    and it composes results into a structure the Evaluator consumes.
    """
    return {q.query_id: dict(pairs) for q, pairs in zip(queries, per_query_hits)}
