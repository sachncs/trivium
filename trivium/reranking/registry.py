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
