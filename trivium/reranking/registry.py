"""RerankerRegistry: lookup by slug and per-slug factory functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trivium.reranking.base import Reranker

FACTORIES: dict[str, Callable[..., Reranker]] = {}


def register_reranker(slug: str) -> Callable[[Callable[..., Reranker]], Callable[..., Reranker]]:
    def deco(factory: Callable[..., Reranker]) -> Callable[..., Reranker]:
        FACTORIES[slug] = factory
        return factory

    return deco


def get_reranker(slug: str, **kwargs: Any) -> Reranker:
    if slug not in FACTORIES:
        raise KeyError(f"unknown reranker slug: {slug!r}; registered: {sorted(FACTORIES)}")
    return FACTORIES[slug](**kwargs)


class RerankerRegistry:
    @staticmethod
    def register(slug: str, factory: Callable[..., Reranker]) -> None:
        FACTORIES[slug] = factory

    @staticmethod
    def get(slug: str, **kwargs: Any) -> Reranker:
        return get_reranker(slug, **kwargs)

    @staticmethod
    def slugs() -> list[str]:
        return sorted(FACTORIES)


@register_reranker("encoder")
def _make_encoder(**kwargs: Any) -> Reranker:
    """cross-encoder/ms-marco-MiniLM-L-6-v2 (default)."""
    from trivium.reranking.encoder import Encoder

    defaults: dict[str, Any] = {"slug": "encoder"}
    defaults.update(kwargs)
    return Encoder(**defaults)


@register_reranker("bge-large")
def _make_bge_large(**kwargs: Any) -> Reranker:
    """BAAI/bge-reranker-large (2.3 GB)."""
    from trivium.reranking.bge import Bge

    defaults: dict[str, Any] = {
        "slug": "bge-large",
        "model_id": "BAAI/bge-reranker-large",
    }
    defaults.update(kwargs)
    return Bge(**defaults)


@register_reranker("bge-base")
def _make_bge_base(**kwargs: Any) -> Reranker:
    """BAAI/bge-reranker-base (450 MB, faster than bge-large)."""
    from trivium.reranking.bge import Bge

    defaults: dict[str, Any] = {
        "slug": "bge-base",
        "model_id": "BAAI/bge-reranker-base",
    }
    defaults.update(kwargs)
    return Bge(**defaults)


try:
    from trivium.reranking.monot5 import Monot5  # noqa: F401

    @register_reranker("monot5-3b")
    def _make_monot5_3b(**kwargs: Any) -> Reranker:
        """castorini/monot5-3b-msmarco-10k (T5-3B, 11 GB; fp16 on GPU/MPS)."""
        defaults: dict[str, Any] = {"slug": "monot5-3b"}
        defaults.update(kwargs)
        return Monot5(**defaults)
except ImportError:
    # transformers / torch missing; MonoT5 not registered. Surface only at
    # first call to get_reranker("monot5-3b"), not at import time.
    pass
