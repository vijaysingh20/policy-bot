from pathlib import Path

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    HANDBOOK_PDF,
    INDEX_DIR,
)
from backend.embedding import (
    count_tokens,
    embed_document_chunks,
    load_model_for_ingestion,
)
from backend.ingestion.chunking import build_document_chunks
from backend.ingestion.pdf_loader import load_pdf_pages
from backend.retrieval.assets import save_manifest, save_retrieval_assets
from backend.retrieval.vector_index import build_faiss_index
from backend.schemas import ChunkRecord, EmbeddedChunk, IndexManifest


def build_token_splitter(
    model: SentenceTransformer
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=model.tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False
    )

def verify_chunk_sizes(
    chunks: list[ChunkRecord],
    model: SentenceTransformer
) -> None:
    largest_token_count = 0
    input_limit = model.max_seq_length
    oversized_chunks: list[ChunkRecord] = []

    for chunk in chunks:
        token_count = count_tokens(chunk.text, model)
        largest_token_count = max(largest_token_count, token_count)

        if token_count > input_limit:
            oversized_chunks.append(chunk)

    print("Total document chunks:", len(chunks))
    print("Model input limit:", input_limit)
    print("Largest chunk token count:", largest_token_count)
    print("Oversized chunks:", len(oversized_chunks))

    if oversized_chunks:
        raise ValueError("Resize oversized chunks before embedding")

def build_index(
    pdf_path: Path = HANDBOOK_PDF,
    output_dir: Path = INDEX_DIR,
    model_name: str = EMBEDDING_MODEL,
    *,
    source_name: str | None = None,
    model: SentenceTransformer | None = None,
    model_revision: str | None = None,
) -> tuple[
    faiss.Index,
    list[EmbeddedChunk],
    IndexManifest,
    SentenceTransformer,
]:
    """Load the PDF, chunk it, embed it, and persist the retrieval assets."""

    if model is None or model_revision is None:
        model, model_revision = load_model_for_ingestion(model_name)
        
    splitter = build_token_splitter(model)

    pages = load_pdf_pages(pdf_path, source_name=source_name)

    document_chunks = build_document_chunks(
        pages=pages,
        splitter=splitter,
    )

    if not document_chunks:
        raise ValueError(
            "No text chunks could be extracted from this PDF."
        )

    verify_chunk_sizes(document_chunks, model)

    embedded_chunks = embed_document_chunks(
        chunks=document_chunks,
        model=model,
    )

    print("Embedded chunks:", len(embedded_chunks))

    index = build_faiss_index(embedded_chunks)

    print("Indexed vectors:", index.ntotal)
    print("Index dimensions:", index.d)

    save_retrieval_assets(
        index=index,
        embedded_chunks=embedded_chunks,
        output_dir=output_dir,
    )

    manifest = IndexManifest(
        embedding_model=model_name,
        embedding_revision=model_revision,
        dimensions=index.d,
        chunk_count=index.ntotal
    )

    save_manifest(
        manifest=manifest,
        output_dir=output_dir
    )

    return index, embedded_chunks, manifest, model
