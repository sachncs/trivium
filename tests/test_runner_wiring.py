"""Runner wiring tests for trivium.cli.run_benchmark.

We exercise the loop construction without spinning up any real
embedder or reranker model by stubbing the heavy collaborators.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import trivium.cli.run_benchmark as rb
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
from trivium.domain.document import Document


def _make_config(encoder_slugs: list[str]) -> Config:
    return Config(
        corpus=CorpusConfig(),
        scales=[5],
        encoders=[
            EncoderEntry(
                slug=slug,
                model_id="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
            )
            for slug in encoder_slugs
        ],
        bm25=Bm25Config(),
        vector=VectorConfig(nlist_by_scale={5: 4}),
        hybrid=HybridConfig(),
        rerank=RerankConfig(),
        benchmark=BenchmarkConfig(),
        preflight=PreflightConfig(),
        runtime=RuntimeConfig(),
    )


@pytest.fixture
def tiny_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Populate data/cache with a 5-doc corpus and 2 queries."""
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    docs = [Document(doc_id=f"d{i}", title=f"t{i}", text=f"doc {i}") for i in range(5)]
    (cache / "scifact_seed.jsonl").write_text(
        "\n".join(f'{{"id":"{d.doc_id}","title":"{d.title}","text":"{d.text}"}}' for d in docs)
        + "\n"
    )
    (cache / "corpus.jsonl").write_text(
        "\n".join(f'{{"id":"{d.doc_id}","title":"{d.title}","text":"{d.text}"}}' for d in docs)
        + "\n"
    )
    (cache / "queries.jsonl").write_text(
        "\n".join(f'{{"id":"q{i}","text":"query {i}"}}' for i in range(2)) + "\n"
    )
    (cache / "qrels.jsonl").write_text(
        "\n".join(f'{{"qid":"q{i}","did":"d{i}","rel":1}}' for i in range(2)) + "\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


class _StubReranker:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    slug = "encoder"
    model_id = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class _StubEmbedder:
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Accept the registry call signature: get_embedder(slug, slug=slug, model_id=..., ...)
        self.slug_value = kwargs.get("slug", args[0] if args else "stub")

    @property
    def slug(self) -> str:
        return self.slug_value

    def warmup(self, _texts):  # type: ignore[no-untyped-def]
        return None

    def encode_queries(self, texts):
        return np.zeros((len(list(texts)), 8), dtype=np.float32)


class _StubPipeline:
    name = "stub"

    def __init__(self, encoders=("*",)) -> None:  # noqa: ANN001 - mirrors ABC
        self.encoders = encoders
        self.inputs: list[object] = []

    def run(self, inp, config):  # type: ignore[override]
        from trivium.domain.pipeline_result import PipelineResult
        from trivium.evaluation.latency import LatencyStats

        self.inputs.append(inp)
        return [
            PipelineResult(
                name=self.name,
                scale=len(inp.documents),
                encoder=inp.encoder_slug,
                reranker=getattr(inp.reranker, "slug", "none") if inp.reranker else "none",
                metrics=MagicMock(),
                latency=LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                extras={"sentinel": True},
            )
        ]


def test_runner_passes_reranker_to_hybrid_rerank_pipeline(
    tiny_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI must hand a real reranker instance to hybrid_rerank mode."""
    config = _make_config(["minilm-l6"])
    bm25_pipe = _StubPipeline(encoders=("none",))
    hybrid_pipe = _StubPipeline()
    monkeypatch.setattr(rb, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(rb, "default_config_path", lambda: Path("/tmp/_unused.yaml"))
    monkeypatch.setattr(
        rb, "get_pipeline", lambda name: bm25_pipe if name == "bm25" else hybrid_pipe
    )
    monkeypatch.setattr(rb, "get_reranker", lambda *_a, **_k: _StubReranker())

    embedder_mock = MagicMock(
        side_effect=lambda *_args, **_k: _StubEmbedder(
            slug=_k.get("slug", _args[0] if _args else "stub")
        )
    )
    monkeypatch.setattr(rb, "get_embedder", embedder_mock)
    monkeypatch.setattr(rb, "preflight_check", lambda *_a, **_k: True)
    monkeypatch.setattr(rb, "CsvResultWriter", MagicMock())

    sys.argv = [
        "run_benchmark",
        "--modes", "hybrid_rerank",
        "--rerankers", "encoder",
        "--scales", "5",
        "--skip-preflight",
        "--output", str(tiny_corpus / "results" / "benchmark.csv"),
    ]
    rb.main()

    assert hybrid_pipe.inputs, "hybrid_rerank pipeline never received a PipelineInput"
    inp = hybrid_pipe.inputs[0]
    assert inp.reranker is not None, "hybrid_rerank PipelineInput.reranker is None"
    assert getattr(inp.reranker, "slug", None) == "encoder"


def test_runner_emits_one_row_per_encoder(
    tiny_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing --encoders a,b must produce rows for both a and b (issue #11)."""
    config = _make_config(["a", "b"])
    bm25_pipe = _StubPipeline(encoders=("none",))
    monkeypatch.setattr(rb, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(rb, "default_config_path", lambda: Path("/tmp/_unused.yaml"))
    monkeypatch.setattr(rb, "get_pipeline", lambda name: bm25_pipe)
    monkeypatch.setattr(rb, "get_reranker", lambda *_a, **_k: _StubReranker())

    embedder_mock = MagicMock(
        side_effect=lambda *_args, **_k: _StubEmbedder(
            slug=_k.get("slug", _args[0] if _args else "stub")
        )
    )
    monkeypatch.setattr(rb, "get_embedder", embedder_mock)
    monkeypatch.setattr(rb, "preflight_check", lambda *_a, **_k: True)
    monkeypatch.setattr(rb, "CsvResultWriter", MagicMock())

    sys.argv = [
        "run_benchmark",
        "--modes", "bm25",
        "--encoders", "a,b",
        "--scales", "5",
        "--skip-preflight",
        "--output", str(tiny_corpus / "results" / "benchmark.csv"),
    ]
    rb.main()

    encoder_slugs = {inp.encoder_slug for inp in bm25_pipe.inputs}
    assert encoder_slugs == {"a", "b"}, f"Expected both encoders to run; got {encoder_slugs}"


def test_runner_drops_unconfigured_encoder(
    tiny_corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--encoders not in the YAML config must be ignored, not crash."""
    config = _make_config(["a"])
    bm25_pipe = _StubPipeline(encoders=("none",))
    monkeypatch.setattr(rb, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(rb, "default_config_path", lambda: Path("/tmp/_unused.yaml"))
    monkeypatch.setattr(rb, "get_pipeline", lambda name: bm25_pipe)
    monkeypatch.setattr(rb, "get_reranker", lambda *_a, **_k: _StubReranker())

    embedder_mock = MagicMock(
        side_effect=lambda *_args, **_k: _StubEmbedder(
            slug=_k.get("slug", _args[0] if _args else "stub")
        )
    )
    monkeypatch.setattr(rb, "get_embedder", embedder_mock)
    monkeypatch.setattr(rb, "preflight_check", lambda *_a, **_k: True)
    monkeypatch.setattr(rb, "CsvResultWriter", MagicMock())

    sys.argv = [
        "run_benchmark",
        "--modes", "bm25",
        "--encoders", "missing-one",
        "--scales", "5",
        "--skip-preflight",
        "--output", str(tiny_corpus / "results" / "benchmark.csv"),
    ]
    rb.main()  # should not raise
    assert bm25_pipe.inputs == [], "Pipeline received input despite missing encoder"


def test_runner_hybrid_rerank_without_rerankers_prints_and_skips(
    tiny_corpus: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """If --rerankers is not given AND none can be instantiated, skip hybrid_rerank."""
    config = _make_config(["minilm-l6"])
    bm25_pipe = _StubPipeline(encoders=("none",))
    hybrid_pipe = _StubPipeline()
    monkeypatch.setattr(rb, "load_config", lambda *_a, **_k: config)
    monkeypatch.setattr(rb, "default_config_path", lambda: Path("/tmp/_unused.yaml"))
    monkeypatch.setattr(
        rb, "get_pipeline", lambda name: bm25_pipe if name == "bm25" else hybrid_pipe
    )
    monkeypatch.setattr(rb, "get_reranker", MagicMock(side_effect=KeyError("nope")))

    embedder_mock = MagicMock(
        side_effect=lambda *_args, **_k: _StubEmbedder(
            slug=_k.get("slug", _args[0] if _args else "stub")
        )
    )
    monkeypatch.setattr(rb, "get_embedder", embedder_mock)
    monkeypatch.setattr(rb, "preflight_check", lambda *_a, **_k: True)
    monkeypatch.setattr(rb, "CsvResultWriter", MagicMock())

    sys.argv = [
        "run_benchmark",
        "--modes", "bm25,hybrid_rerank",
        "--scales", "5",
        "--skip-preflight",
        "--output", str(tiny_corpus / "results" / "benchmark.csv"),
    ]
    rb.main()
    out = capsys.readouterr().out
    assert "requires --rerankers" in out
    assert hybrid_pipe.inputs == []
