"""FusionRegistry: factory for FusionStrategy classes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from trivium.fusion.base import FusionStrategy

FACTORIES: dict[str, Callable[..., FusionStrategy]] = {}


def register_fusion(
    slug: str,
) -> Callable[[Callable[..., FusionStrategy]], Callable[..., FusionStrategy]]:
    def deco(factory: Callable[..., FusionStrategy]) -> Callable[..., FusionStrategy]:
        FACTORIES[slug] = factory
        return factory

    return deco


def get_fusion(slug: str, **kwargs: Any) -> FusionStrategy:
    if slug not in FACTORIES:
        raise KeyError(f"unknown fusion slug: {slug!r}; registered: {sorted(FACTORIES)}")
    return FACTORIES[slug](**kwargs)


class FusionRegistry:
    @staticmethod
    def register(slug: str, factory: Callable[..., FusionStrategy]) -> None:
        FACTORIES[slug] = factory

    @staticmethod
    def get(slug: str, **kwargs: Any) -> FusionStrategy:
        return get_fusion(slug, **kwargs)

    @staticmethod
    def slugs() -> list[str]:
        return sorted(FACTORIES)
