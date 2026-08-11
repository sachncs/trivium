"""RetrieverRegistry: lookup by slug and per-slug factory functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trivium.retrieval.base import Retriever

FACTORIES: dict[str, Callable[..., Retriever]] = {}


def register_retriever(slug: str) -> Callable[[Callable[..., Retriever]], Callable[..., Retriever]]:
    def deco(factory: Callable[..., Retriever]) -> Callable[..., Retriever]:
        FACTORIES[slug] = factory
        return factory

    return deco


def get_retriever(slug: str, **kwargs: Any) -> Retriever:
    if slug not in FACTORIES:
        raise KeyError(f"unknown retriever slug: {slug!r}; registered: {sorted(FACTORIES)}")
    return FACTORIES[slug](**kwargs)


class RetrieverRegistry:
    @staticmethod
    def register(slug: str, factory: Callable[..., Retriever]) -> None:
        FACTORIES[slug] = factory

    @staticmethod
    def get(slug: str, **kwargs: Any) -> Retriever:
        return get_retriever(slug, **kwargs)

    @staticmethod
    def slugs() -> list[str]:
        return sorted(FACTORIES)
