"""FusionStrategy ABC: combines result lists from multiple retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from trivium.domain.result import SearchResult


class FusionStrategy(ABC):
    """Combine SearchResult from N retrievers into one ranked list.

    Implementations:
        - trivium.fusion.rrf.Rrf        (Reciprocal Rank Fusion)
        - trivium.fusion.weighted.Weighted  (linear score combination)
    """

    @property
    @abstractmethod
    def slug(self) -> str:
        """Short identifier used in CSV rows and registries."""

    @abstractmethod
    def fuse(
        self,
        results: Sequence[SearchResult],
        top_k: int,
    ) -> SearchResult:
        """Combine N per-query result lists into a single SearchResult.

        Args:
            results: One SearchResult per input retriever.
            top_k: Limit the output to this many hits.
        """
