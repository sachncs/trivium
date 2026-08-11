"""trivium.embeddings: text vectorizers."""

from trivium.embeddings.registry import EmbedderRegistry, get_embedder, register_embedder

__all__ = ["EmbedderRegistry", "get_embedder", "register_embedder"]
