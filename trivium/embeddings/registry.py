"""EmbedderRegistry: lookup by slug and per-slug factory functions."""

from __future__ import annotations

from collections.abc import Callable

from trivium.embeddings.base import Embedder

_FACTORIES: dict[str, Callable[..., Embedder]] = {}


def register_embedder(slug: str) -> Callable[[Callable[..., Embedder]], Callable[..., Embedder]]:
    """Decorator: register an Embedder factory under `slug`."""

    def deco(factory: Callable[..., Embedder]) -> Callable[..., Embedder]:
        _FACTORIES[slug] = factory
        return factory

    return deco


def get_embedder(slug: str, **kwargs) -> Embedder:
    """Instantiate the embedder registered for `slug` with kwargs."""
    if slug not in _FACTORIES:
        raise KeyError(f"unknown embedder slug: {slug!r}; registered: {sorted(_FACTORIES)}")
    return _FACTORIES[slug](**kwargs)


class EmbedderRegistry:
    """Convenience façade over the module-level factory dict."""

    @staticmethod
    def register(slug: str, factory: Callable[..., Embedder]) -> None:
        _FACTORIES[slug] = factory

    @staticmethod
    def get(slug: str, **kwargs) -> Embedder:
        return get_embedder(slug, **kwargs)

    @staticmethod
    def slugs() -> list[str]:
        return sorted(_FACTORIES)


# Built-in registrations. Adding a new encoder is one decorator call.
try:
    from trivium.embeddings.sentence import Sentence

    @register_embedder("minilm-l6")
    def _make_minilm(**kwargs) -> Embedder:
        defaults = dict(
            slug="minilm-l6",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
        )
        defaults.update(kwargs)
        return Sentence(**defaults)
except ImportError:
    pass
