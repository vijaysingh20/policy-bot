from dataclasses import dataclass
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from backend.embedding import load_model_for_query
from backend.retrieval.assets import load_retrieval_assets
from backend.schemas import ChunkRecord, IndexManifest


@dataclass(frozen=True)
class DocumentResources:
    index: faiss.Index
    chunks: list[ChunkRecord]
    manifest: IndexManifest
    embedding_model: SentenceTransformer


def load_document_resources(
    assets_dir: Path,
    *,
    shared_model: SentenceTransformer,
    shared_model_name: str,
    shared_revision: str
) -> DocumentResources:
    index, chunks, manifest = load_retrieval_assets(assets_dir)

    same_model = (
        manifest.embedding_model == shared_model_name
        and manifest.embedding_revision == shared_revision
    )
    embedding_model = shared_model if same_model else load_model_for_query(manifest)

    return DocumentResources(
        index=index, chunks=chunks, manifest=manifest, embedding_model=embedding_model
    )