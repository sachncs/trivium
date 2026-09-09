"""Tests for trivium.reranking.registry.

We mock the heavy CrossEncoder / T5 model loads so the tests run on
CI without downloading 2-11 GB checkpoints.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from trivium.reranking import registry as reranker_registry
from trivium.reranking.base import Reranker


class _StubReranker(Reranker):
    """Test double: returns candidates in input order, no model load."""

    def __init__(self, slug: str = "stub", model_id: str = "stub-id", **extra: object) -> None:
        self.slug_value = slug
        self.model_id_value = model_id
        self.kwargs_received: dict[str, object] = {"slug": slug, "model_id": model_id, **extra}

    @property
    def slug(self) -> str:
        return self.slug_value

    @property
    def model_id(self) -> str:
        return self.model_id_value

    def rerank(self, query, candidates, top_k):  # type: ignore[override]
        from trivium.domain.document import Document
        from trivium.domain.result import SearchResult

        if not candidates:
            return SearchResult.empty()
        keep = list(candidates)[:top_k] if top_k else list(candidates)
        return SearchResult(
            hits=[
                (doc, float(len(keep) - i))
                for i, doc in enumerate(keep)
                if isinstance(doc, Document)
            ]
        )


@pytest.fixture(autouse=True)
def _patch_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy reranker classes with light stubs."""
    encoder_module = MagicMock()
    encoder_module.Encoder = _StubReranker
    bge_module = MagicMock()
    bge_module.Bge = _StubReranker
    monot5_module = MagicMock()
    monot5_module.Monot5 = _StubReranker

    monkeypatch.setitem(
        __import__("sys").modules,
        "trivium.reranking.encoder",
        encoder_module,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "trivium.reranking.bge",
        bge_module,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "trivium.reranking.monot5",
        monot5_module,
    )


def test_encoder_factory_returns_stub() -> None:
    rr = reranker_registry.get_reranker("encoder")
    assert isinstance(rr, Reranker)
    assert rr.slug == "encoder"


def test_bge_large_factory_returns_stub() -> None:
    rr = reranker_registry.get_reranker("bge-large")
    assert isinstance(rr, Reranker)
    assert rr.slug == "bge-large"


def test_bge_base_factory_returns_stub() -> None:
    rr = reranker_registry.get_reranker("bge-base")
    assert isinstance(rr, Reranker)
    assert rr.slug == "bge-base"


def test_bge_large_passes_model_id_through() -> None:
    """The bge-large factory must pin the BAAI/bge-reranker-large model_id."""
    rr = reranker_registry.get_reranker("bge-large")
    assert rr.model_id == "BAAI/bge-reranker-large"


def test_bge_base_passes_model_id_through() -> None:
    """The bge-base factory must pin the BAAI/bge-reranker-base model_id."""
    rr = reranker_registry.get_reranker("bge-base")
    assert rr.model_id == "BAAI/bge-reranker-base"


def test_monot5_3b_factory_returns_stub() -> None:
    rr = reranker_registry.get_reranker("monot5-3b")
    assert isinstance(rr, Reranker)
    assert rr.slug == "monot5-3b"


def test_unknown_slug_raises_key_error() -> None:
    with pytest.raises(KeyError):
        reranker_registry.get_reranker("not-a-real-reranker")


def test_registry_fa_exposes_slugs() -> None:
    slugs = reranker_registry.RerankerRegistry.slugs()
    assert "encoder" in slugs
    assert "bge-large" in slugs
    assert "bge-base" in slugs
    assert "monot5-3b" in slugs
