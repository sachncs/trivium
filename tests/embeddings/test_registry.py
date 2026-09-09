"""Tests for trivium.embeddings.registry.

We stub out the heavy Sentence and E5 classes via sys.modules so
the real sentence_transformers (which spawns a tqdm monitor thread
that races with faiss on macOS) is never imported during tests.
"""
from __future__ import annotations

import sys
import types

import pytest

from trivium.embeddings import registry as embedder_registry
from trivium.embeddings.base import Embedder


class _StubEmbedder(Embedder):
    """Test double that records constructor kwargs for assertions."""

    def __init__(
        self,
        stub_slug: str = "stub",
        stub_model_id: str = "stub-id",
        stub_dimension: int = 0,
        **_: object,
    ) -> None:
        self.stub_slug = stub_slug
        self.stub_model_id = stub_model_id
        self.stub_dimension = stub_dimension

    @property
    def slug(self) -> str:
        return self.stub_slug

    @property
    def model_id(self) -> str:
        return self.stub_model_id

    @property
    def dimension(self) -> int:
        return self.stub_dimension

    def encode_documents(self, texts):  # type: ignore[override]
        raise NotImplementedError

    def encode_queries(self, texts):  # type: ignore[override]
        raise NotImplementedError

    def warmup(self, sample_texts=("warmup",)):  # type: ignore[override]
        return None


@pytest.fixture(autouse=True)
def _stub_heavy_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject lightweight stand-ins for trivium.embeddings.sentence and trivium.embeddings.e5."""
    sentence_stub = types.ModuleType("trivium.embeddings.sentence")
    e5_stub = types.ModuleType("trivium.embeddings.e5")

    class _StubSentence(_StubEmbedder):
        def __init__(self, slug="sentence-stub", model_id="m", dimension=0, **extra: object) -> None:
            super().__init__(stub_slug=slug, stub_model_id=model_id, stub_dimension=dimension)

    class _StubE5(_StubEmbedder):
        def __init__(self, slug="e5-stub", **extra: object) -> None:
            super().__init__(
                stub_slug=slug,
                stub_model_id="intfloat/e5-mistral-7b-instruct",
                stub_dimension=4096,
            )

        @property
        def dimension(self) -> int:
            return 4096

    sentence_stub.Sentence = _StubSentence  # type: ignore[attr-defined]
    e5_stub.E5 = _StubE5  # type: ignore[attr-defined]

    # Delete any previously-cached real modules so the lazy
    # `from trivium.embeddings import e5` resolves to the stub.
    for mod in (
        "trivium.embeddings.e5",
        "trivium.embeddings.sentence",
        "sentence_transformers",
        "torch",
    ):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    monkeypatch.setitem(sys.modules, "trivium.embeddings.sentence", sentence_stub)
    monkeypatch.setitem(sys.modules, "trivium.embeddings.e5", e5_stub)

    # Patch the registry's module-level reference too, in case any code
    # path cached the import.
    import trivium.embeddings.registry as _reg

    if hasattr(_reg, "_e5_module"):
        monkeypatch.setattr(_reg, "_e5_module", e5_stub)

    # Drop the cached attribute on the package so the next
    # `from trivium.embeddings import e5` re-imports and finds the stub.
    import trivium.embeddings as _pkg

    monkeypatch.delitem(_pkg.__dict__, "e5", raising=False)
    monkeypatch.delitem(_pkg.__dict__, "sentence", raising=False)


def test_minilm_l6_factory_returns_stub() -> None:
    emb = embedder_registry.get_embedder("minilm-l6")
    assert emb.slug == "minilm-l6"
    assert emb.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb.dimension == 384


def test_bge_small_factory() -> None:
    emb = embedder_registry.get_embedder("bge-small")
    assert emb.slug == "bge-small"
    assert emb.model_id == "BAAI/bge-small-en-v1.5"
    assert emb.dimension == 384


def test_bge_base_factory() -> None:
    emb = embedder_registry.get_embedder("bge-base")
    assert emb.slug == "bge-base"
    assert emb.model_id == "BAAI/bge-base-en-v1.5"
    assert emb.dimension == 768


def test_bge_large_factory() -> None:
    emb = embedder_registry.get_embedder("bge-large")
    assert emb.slug == "bge-large"
    assert emb.model_id == "BAAI/bge-large-en-v1.5"
    assert emb.dimension == 1024


def test_mpnet_base_factory() -> None:
    emb = embedder_registry.get_embedder("mpnet-base")
    assert emb.slug == "mpnet-base"
    assert emb.model_id == "sentence-transformers/all-mpnet-base-v2"
    assert emb.dimension == 768


def test_e5_mistral_7b_factory() -> None:
    emb = embedder_registry.get_embedder("e5-mistral-7b")
    assert emb.slug == "e5-mistral-7b"
    assert emb.model_id == "intfloat/e5-mistral-7b-instruct"
    assert emb.dimension == 4096


def test_unknown_slug_raises_key_error() -> None:
    with pytest.raises(KeyError):
        embedder_registry.get_embedder("not-a-real-embedder")


def test_registry_slugs_includes_all() -> None:
    slugs = embedder_registry.EmbedderRegistry.slugs()
    for expected in (
        "minilm-l6",
        "bge-small",
        "bge-base",
        "bge-large",
        "mpnet-base",
        "e5-mistral-7b",
    ):
        assert expected in slugs, f"{expected!r} missing from registered slugs: {slugs}"
