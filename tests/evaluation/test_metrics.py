"""Unit tests for trivium.evaluation."""
from __future__ import annotations

import numpy as np

from trivium.domain.qrels import Qrels
from trivium.domain.result import SearchResult
from trivium.evaluation.latency import LatencyProbe, LatencyStats, drop_os_page_cache, probe_lambda
from trivium.evaluation.metrics import Evaluator, hits_to_results


class TestEvaluator:
    def test_empty_returns_zero_metrics(self):
        ev = Evaluator(k_values=(10,))
        m = ev.evaluate(Qrels(), {})
        assert m.ndcg_at_10 == 0.0
        assert m.recall_at_10 == 0.0

    def test_basic_ndcg_is_in_unit_range(self):
        rows = [{"qid": "q1", "did": "d1", "rel": 1}, {"qid": "q1", "did": "d2", "rel": 0}]
        qrels = Qrels.from_rows(rows)
        results = {"q1": {"d1": 1.0, "d2": 0.5}}
        m = Evaluator(k_values=(10,)).evaluate(qrels, results)
        assert 0.0 <= m.ndcg_at_10 <= 1.0

    def test_measure_name_roundtrip(self):
        assert Evaluator.from_measure_name(Evaluator.to_measure_name("ndcg_cut.10")) == "ndcg_cut.10"

    def test_hits_to_results(self):
        queries = [type("Q", (), {"query_id": "q1"})()]
        per_query = [[("d1", 0.9)]]
        out = hits_to_results(per_query, queries)
        assert out == {"q1": {"d1": 0.9}}


class TestLatency:
    def test_probe_runs_n_times(self):
        counter = [0]

        def f():
            counter[0] += 1
            return 1 + 1

        probe = LatencyProbe(fn=f, n=10, warmup=2, rotate=None)
        stats = probe.run()
        assert stats.n == 10
        assert counter[0] == 12  # 2 warmup + 10 measured

    def test_probe_with_rotate(self):
        rotate = ["a", "b", "c"]
        probe = LatencyProbe(fn=lambda x: x, n=6, warmup=0, rotate=rotate)
        probe.run()  # should not raise

    def test_drop_os_page_cache_is_safe(self):
        # just calling it should not raise; the underlying
        # implementation handles permission errors.
        drop_os_page_cache()

    def test_probe_lambda_convenience(self):
        stats = probe_lambda(lambda: None, n=5, warmup=1)
        assert stats.n == 5


class TestLatencyStats:
    def test_from_samples(self):
        samples = np.array([1_000_000] * 100)
        s = LatencyStats.from_samples(samples)
        assert s.n == 100
        assert s.mean_ms == pytest.approx(1.0, abs=1e-3)


import pytest
