"""Query type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Query:
    """A retrieval query (BEIR test split row)."""

    query_id: str
    text: str

    @classmethod
    def from_row(cls, row: dict) -> Query:
        return cls(query_id=row["id"], text=row["text"])

    @classmethod
    def many(cls, rows: list[dict]) -> list[Query]:
        return [cls.from_row(r) for r in rows]
