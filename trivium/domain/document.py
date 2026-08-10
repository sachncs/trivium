"""Document type. Single source of truth for title+text concatenation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    """A corpus document.

    The `body` property holds the canonical BEIR-style
    'title + newline + text' concatenation used everywhere
    (BM25 tokenisation, cross-encoder reranking input, etc.).
    """

    doc_id: str
    title: str = ""
    text: str = ""

    @property
    def body(self) -> str:
        """Return 'title\ntext' or whichever is non-empty."""
        title = self.title.strip()
        text = self.text.strip()
        if title and text:
            return f"{title}\n{text}"
        return title or text

    @classmethod
    def from_row(cls, row: dict) -> "Document":
        """Build from a dict produced by the legacy loaders."""
        return cls(
            doc_id=row["id"],
            title=row.get("title", "") or "",
            text=row.get("text", "") or "",
        )
