"""Domain value types: Document, Query, Qrels, Hit, SearchResult, PipelineResult."""

from trivium.domain.document import Document
from trivium.domain.hit import Hit
from trivium.domain.pipeline_result import PipelineResult
from trivium.domain.qrels import Qrels
from trivium.domain.query import Query
from trivium.domain.result import SearchResult

__all__ = [
    "Document",
    "Hit",
    "PipelineResult",
    "Query",
    "Qrels",
    "SearchResult",
]
