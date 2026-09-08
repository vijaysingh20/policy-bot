from sentence_transformers import SentenceTransformer

from backend.schemas import ChunkRecord, EmbeddedChunk, Vector


def embed_document_chunks(
    chunks: list[ChunkRecord],
    model: SentenceTransformer
) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("Document embedding shape: ", embeddings.shape)
    embedded_chunks = []

    for chunk, row in zip(chunks, embeddings, strict=True):
        vector = Vector(values=row.tolist())
        embedded_chunk = EmbeddedChunk(
            chunk=chunk,
            vector=vector
        )

        embedded_chunks.append(embedded_chunk)
    return embedded_chunks
