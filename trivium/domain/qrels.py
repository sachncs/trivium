"""Qrels: relevance judgements, BEIR nested-dict + pytrec_eval helpers."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Qrels:
    """Relevance judgements in BEIR format: {qid: {did: rel}}.

    Frozen because qrels are immutable after construction. Iteration
    over (qid, did, rel) tuples is the primary access pattern.
    """

    matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "Qrels":
        out: dict[str, dict[str, int]] = {}
        for r in rows:
            out.setdefault(r["qid"], {})[r["did"]] = int(r["rel"])
        return cls(matrix=out)

    def to_pytrec(self) -> dict[str, dict[str, int]]:
        """Convert to the nested-dict shape pytrec_eval expects."""
        return {qid: dict(dmap) for qid, dmap in self.matrix.items()}

    def relevant_for(self, query_id: str) -> dict[str, int]:
        """Return the {did: rel} dict for one query."""
        return self.matrix.get(query_id, {})

    def all_query_ids(self) -> list[str]:
        return list(self.matrix.keys())

    def __len__(self) -> int:
        return len(self.matrix)
