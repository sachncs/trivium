"""PipelineRegistry: registry-driven dispatcher.

Adding a new benchmark mode = adding a class + one decorator.
No elif chains anywhere in trivium.
"""

from __future__ import annotations

from collections.abc import Callable

from trivium.pipelines.base import BenchmarkPipeline

FACTORIES: dict[str, type[BenchmarkPipeline]] = {}


def register_pipeline(name: str) -> Callable[[type[BenchmarkPipeline]], type[BenchmarkPipeline]]:
    def deco(cls: type[BenchmarkPipeline]) -> type[BenchmarkPipeline]:
        FACTORIES[name] = cls
        return cls

    return deco


def get_pipeline(name: str) -> BenchmarkPipeline:
    if name not in FACTORIES:
        raise KeyError(f"unknown pipeline name: {name!r}; registered: {sorted(FACTORIES)}")
    return FACTORIES[name]()


class PipelineRegistry:
    @staticmethod
    def register(name: str, cls: type[BenchmarkPipeline]) -> None:
        FACTORIES[name] = cls

    @staticmethod
    def get(name: str) -> BenchmarkPipeline:
        return get_pipeline(name)

    @staticmethod
    def names() -> list[str]:
        return sorted(FACTORIES)
