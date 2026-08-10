"""Pydantic v2 config schema + loader."""
from trivium.config.loader import load_config
from trivium.config.schema import (
    BenchmarkConfig,
    Bm25Config,
    Config,
    CorpusConfig,
    EncoderEntry,
    HybridConfig,
    PreflightConfig,
    RerankConfig,
    RuntimeConfig,
    VectorConfig,
)

__all__ = [
    "BenchmarkConfig",
    "Bm25Config",
    "Config",
    "CorpusConfig",
    "EncoderEntry",
    "HybridConfig",
    "PreflightConfig",
    "RerankConfig",
    "RuntimeConfig",
    "VectorConfig",
    "load_config",
]
