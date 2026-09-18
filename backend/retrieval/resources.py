from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer

from backend.schemas import ChunkRecord, IndexManifest


@dataclass(frozen=True)
class DocumentResources:
    index: faiss.Index
    chunks: list[ChunkRecord]
    manifest: IndexManifest
    embedding_model: SentenceTransformer