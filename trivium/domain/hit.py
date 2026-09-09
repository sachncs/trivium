"""Hit: a single (doc_id, score) pair."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Hit:
    """A single ranked hit."""

    doc_id: str
    score: float

    def to_pair(self) -> tuple[str, float]:
        return (self.doc_id, float(self.score))
