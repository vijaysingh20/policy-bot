"""Pydantic models shared across the pipeline.

Grouped by domain, but re-exported here so callers can keep using
``from backend.schemas import ChunkRecord``.
"""

from backend.schemas.answers import (
    AnswerDraft,
    ContextSource,
    QueryPlan,
    RetrievalContext,
)
from backend.schemas.documents import ChunkRecord, PageMetaData, PageRecord
from backend.schemas.evaluation import EvaluationInput, EvaluationScores, EvaluationRecord
from backend.schemas.indexing import EmbeddedChunk, IndexManifest, Vector
from backend.schemas.search import RerankedResult, SearchRequest, SearchResult

__all__ = [
    "AnswerDraft",
    "ChunkRecord",
    "ContextSource",
    "EmbeddedChunk",
    "EvaluationInput",
    "EvaluationScores",
    "IndexManifest",
    "PageMetaData",
    "PageRecord",
    "QueryPlan",
    "RerankedResult",
    "RetrievalContext",
    "SearchRequest",
    "SearchResult",
    "Vector",
]
