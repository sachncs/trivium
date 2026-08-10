"""Unit tests for trivium.config."""
from __future__ import annotations

import pytest

from trivium.config.loader import load_config
from trivium.config.schema import (
    BenchmarkConfig,
    Bm25Config,
    Config,
    CorpusConfig,
    EncoderEntry,
    HybridConfig,
    PreflightConfig,
    RerankConfig,
    RuntimeConfig,
    VectorConfig,
)


def test_loader_round_trips_default_yaml(config_path):
    """The trivial default.yaml should load with all expected fields."""
    cfg = load_config(config_path)
    assert isinstance(cfg, Config)
    assert cfg.bm25.method == "lucene"
    assert cfg.bm25.k1 == pytest.approx(0.9)
    assert cfg.bm25.b == pytest.approx(0.4)
    assert 100_000 in cfg.vector.nlist_by_scale
    assert 8 in cfg.vector.nprobe_sweep
    assert cfg.preflight.target_ndcg10 == pytest.approx(0.6789)
    assert cfg.preflight.tolerance == pytest.approx(0.03)


def test_vector_min_scale_for_ivfpq_is_configurable():
    cfg = Config(vector=VectorConfig(min_scale_for_ivfpq=10_000, nlist_by_scale={100_000: 4096}))
    assert cfg.vector.min_scale_for_ivfpq == 10_000


def test_nlist_for_scale_unconfigured_raises_key_error():
    cfg = Config(vector=VectorConfig(nlist_by_scale={100_000: 4096}))
    with pytest.raises(KeyError):
        cfg.vector.nlist_for_scale(50_000)


def test_pydantic_forbids_extra_fields():
    with pytest.raises(Exception):
        EncoderEntry(slug="x", unknown="bad")  # type: ignore[call-arg]


def test_bm25_default_stopwords_and_stemmer():
    cfg = Bm25Config()
    assert cfg.stopwords == "en"
    assert cfg.stemmer is None


def test_hybrid_weight_sweep_default():
    cfg = HybridConfig()
    assert (0.5, 0.5) in cfg.rrf_weight_sweep


def test_preflight_min_observed_default():
    cfg = PreflightConfig()
    assert cfg.min_observed == pytest.approx(0.62)


def test_runtime_default_seeds():
    cfg = RuntimeConfig()
    assert cfg.random_seed == 42
    assert cfg.python_hash_seed == 42
    assert cfg.faiss_omp_threads == 1


def test_corpus_default_distractors():
    cfg = CorpusConfig()
    assert "scidocs" in cfg.distractors


def test_rerank_default_model():
    cfg = RerankConfig()
    assert cfg.model_id == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.max_length == 256


def test_benchmark_default_top_k():
    cfg = BenchmarkConfig()
    assert cfg.top_k_eval == [10, 100]
