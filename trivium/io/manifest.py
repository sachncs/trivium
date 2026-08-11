"""Manifest: pydantic-persisted build manifest for corpora and embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ManifestScales(BaseModel):
    model_config = ConfigDict(extra="forbid")
    counts: dict[str, int] = Field(default_factory=dict)


class ManifestEncoders(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entries: dict[str, str] = Field(default_factory=dict)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format_version: int = 2
    scales: ManifestScales = Field(default_factory=ManifestScales)
    encoders: ManifestEncoders = Field(default_factory=ManifestEncoders)
    corpus_sha256: str = ""
    n_queries: int = 0
    n_qrels: int = 0
    extras: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @staticmethod
    def hash_corpus(path: Path) -> str:
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
