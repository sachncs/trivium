"""EmbedderRegistry: lookup by slug and per-slug factory functions."""

from __future__ import annotations

from collections.abc import Callable

from trivium.embeddings.base import Embedder

FACTORIES: dict[str, Callable[..., Embedder]] = {}


def register_embedder(slug: str) -> Callable[[Callable[..., Embedder]], Callable[..., Embedder]]:
    """Decorator: register an Embedder factory under `slug`."""

    def deco(factory: Callable[..., Embedder]) -> Callable[..., Embedder]:
        FACTORIES[slug] = factory
        return factory

    return deco


def get_embedder(slug: str, **kwargs) -> Embedder:
    """Instantiate the embedder registered for `slug` with kwargs."""
    if slug not in FACTORIES:
        raise KeyError(f"unknown embedder slug: {slug!r}; registered: {sorted(FACTORIES)}")
    return FACTORIES[slug](**kwargs)


class EmbedderRegistry:
    """Convenience façade over the module-level factory dict."""

    @staticmethod
    def register(slug: str, factory: Callable[..., Embedder]) -> None:
        FACTORIES[slug] = factory

    @staticmethod
    def get(slug: str, **kwargs) -> Embedder:
        return get_embedder(slug, **kwargs)

    @staticmethod
    def slugs() -> list[str]:
        return sorted(FACTORIES)


# ----- Built-in registrations -----
# Each registration is wrapped in try/except ImportError so the wheel
# still installs without the heavy model dependencies (transformers /
# torch for E5). The registry simply lacks the slug if the dependency
# is missing; the runner surfaces the missing slug as a printed warning.


@register_embedder("minilm-l6")
def make_minilm(**kwargs) -> Embedder:
    """sentence-transformers/all-MiniLM-L6-v2 (90 MB, 384d, l2-normalised)."""
    try:
        import torch  # noqa: F401

        torch.set_num_threads(1)
    except ImportError:
        pass
    from trivium.embeddings import sentence as _sentence_module

    defaults = {
        "slug": "minilm-l6",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
    }
    defaults.update(kwargs)
    return _sentence_module.Sentence(**defaults)


@register_embedder("bge-small")
def make_bge_small(**kwargs) -> Embedder:
    """BAAI/bge-small-en-v1.5 (130 MB, 384d)."""
    from trivium.embeddings import sentence as _sentence_module

    defaults = {
        "slug": "bge-small",
        "model_id": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "max_seq_length": 512,
    }
    defaults.update(kwargs)
    return _sentence_module.Sentence(**defaults)


@register_embedder("bge-base")
def make_bge_base(**kwargs) -> Embedder:
    """BAAI/bge-base-en-v1.5 (440 MB, 768d)."""
    from trivium.embeddings import sentence as _sentence_module

    defaults = {
        "slug": "bge-base",
        "model_id": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
        "max_seq_length": 512,
    }
    defaults.update(kwargs)
    return _sentence_module.Sentence(**defaults)


@register_embedder("bge-large")
def make_bge_large(**kwargs) -> Embedder:
    """BAAI/bge-large-en-v1.5 (1.34 GB, 1024d)."""
    from trivium.embeddings import sentence as _sentence_module

    defaults = {
        "slug": "bge-large",
        "model_id": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "max_seq_length": 512,
    }
    defaults.update(kwargs)
    return _sentence_module.Sentence(**defaults)


@register_embedder("mpnet-base")
def make_mpnet_base(**kwargs) -> Embedder:
    """sentence-transformers/all-mpnet-base-v2 (440 MB, 768d)."""
    from trivium.embeddings import sentence as _sentence_module

    defaults = {
        "slug": "mpnet-base",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "max_seq_length": 384,
    }
    defaults.update(kwargs)
    return _sentence_module.Sentence(**defaults)


try:
    import trivium.embeddings.e5 as _e5_module  # noqa: F401

    @register_embedder("e5-mistral-7b")
    def make_e5_mistral_7b(**kwargs) -> Embedder:
        """intfloat/e5-mistral-7b-instruct (14 GB, 4096d; GPU/MPS strongly preferred)."""
        defaults = {
            "slug": "e5-mistral-7b",
        }
        defaults.update(kwargs)
        # Resolve E5 through the module each call so test-time monkey-patching
        # of trivium.embeddings.e5.E5 takes effect without an import reload.
        return _e5_module.E5(**defaults)

except ImportError:
    # transformers / torch not installed; the E5 slug is simply missing.
    pass
