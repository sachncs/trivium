"""SearchResult: the unified return type for all retrievers/fusion/rerank."""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from trivium.domain.hit import Hit


class SearchResult:
    """An ordered list of Hits with a stable top_k view.

    Returned by every Retriever.search, every FusionStrategy.fuse,
    and every Reranker.rerank. Replaces the three inconsistent
    shapes that lived in the legacy code (tuple[list[int], list[float]],
    tuple[np.ndarray, np.ndarray], list[tuple[str, float]]).
    """

    __slots__ = ("hits",)

    def __init__(self, hits: Iterable[Hit] | None = None) -> None:
        items = list(hits) if hits is not None else []
        self.hits: tuple[Hit, ...] = tuple(items)

    def __iter__(self) -> Iterator[Hit]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, idx):
        return self.hits[idx]

    def top_k(self, k: int) -> "SearchResult":
        """Return a new SearchResult with at most k hits."""
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        return SearchResult(self.hits[:k])

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[str, float]]) -> "SearchResult":
        return cls(Hit(did, float(s)) for did, s in pairs)

    def to_pairs(self) -> list[tuple[str, float]]:
        return [(h.doc_id, h.score) for h in self.hits]

    @classmethod
    def empty(cls) -> "SearchResult":
        return cls(())
