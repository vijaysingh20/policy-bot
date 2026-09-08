"""Search the persisted index and narrow the candidates down to context."""

from backend.retrieval.assets import (
    load_retrieval_assets,
    save_manifest,
    save_retrieval_assets,
)
from backend.retrieval.reranking import load_reranker, rerank_candidate
from backend.retrieval.vector_index import build_faiss_index, search_faiss
from backend.retrieval.workflow import retrieve_for_plan

__all__ = [
    "build_faiss_index",
    "load_reranker",
    "load_retrieval_assets",
    "rerank_candidate",
    "retrieve_for_plan",
    "save_manifest",
    "save_retrieval_assets",
    "search_faiss",
]
