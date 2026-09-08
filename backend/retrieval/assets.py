import faiss
from pathlib import Path

from backend.schemas import ChunkRecord, EmbeddedChunk, IndexManifest


def save_retrieval_assets(
    index: faiss.IndexFlatIP,
    embedded_chunks: list[EmbeddedChunk],
    output_dir: Path
) -> None:
    if index.ntotal != len(embedded_chunks):
        raise ValueError("Index and chunks length do not match")

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.jsonl"

    faiss.write_index(index, str(index_path))

    with chunks_path.open("w", encoding="utf-8") as file:
        for item in embedded_chunks:
            serialized_chunk = item.chunk.model_dump_json()
            file.write(serialized_chunk + "\n")

def save_manifest(
    manifest: IndexManifest,
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    serialized_manifest = manifest.model_dump_json(indent=2)

    with manifest_path.open("w", encoding="utf-8") as file:
        file.write(serialized_manifest)

def load_retrieval_assets(
    output_dir: Path
) -> tuple[faiss.Index, list[ChunkRecord], IndexManifest]:
    manifest_path = output_dir / "manifest.json"
    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.jsonl"

    manifest_json = manifest_path.read_text(encoding="utf-8")
    manifest = IndexManifest.model_validate_json(manifest_json)

    chunks: list[ChunkRecord] = []

    with chunks_path.open("r", encoding="utf-8") as file:
        for line in file:
            result = ChunkRecord.model_validate_json(line)
            chunks.append(result)

    index = faiss.read_index(str(index_path))

    if(
        not isinstance(index, faiss.IndexFlat)
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise ValueError("Expected a flat inner product index")

    if index.d != manifest.dimensions:
        raise ValueError("dimensions do not match")

    if not (index.ntotal == len(chunks) == manifest.chunk_count):
        raise ValueError(
            f"Count mismatch: index={index.ntotal}, "
            f"chunks={len(chunks)}, manifest={manifest.chunk_count}"
        )

    return index, chunks, manifest