"""Evaluation APIs with lazy imports for lightweight package consumers."""

__all__ = ["EvaluationMetrics", "Evaluator", "ExactSearch", "LatencyProbe", "LatencyStats"]


def __getattr__(name: str):
    if name in {"Evaluator", "EvaluationMetrics"}:
        from trivium.evaluation.metrics import EvaluationMetrics, Evaluator

        return {"Evaluator": Evaluator, "EvaluationMetrics": EvaluationMetrics}[name]
    if name in {"LatencyProbe", "LatencyStats"}:
        from trivium.evaluation.latency import LatencyProbe, LatencyStats

        return {"LatencyProbe": LatencyProbe, "LatencyStats": LatencyStats}[name]
    if name == "ExactSearch":
        from trivium.evaluation.ground_truth import ExactSearch

        return ExactSearch
    raise AttributeError(name)
