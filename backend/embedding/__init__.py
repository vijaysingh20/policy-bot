"""Embedding model loading and chunk encoding, shared by ingestion and retrieval."""

from backend.embedding.encoder import embed_document_chunks
from backend.embedding.model import (
    count_tokens,
    load_model_for_ingestion,
    load_model_for_query,
)

__all__ = [
    "count_tokens",
    "embed_document_chunks",
    "load_model_for_ingestion",
    "load_model_for_query",
]
