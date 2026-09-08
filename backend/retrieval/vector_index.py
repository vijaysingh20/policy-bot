import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.schemas import (
    ChunkRecord,
    EmbeddedChunk,
    SearchRequest,
    SearchResult,
)


def build_faiss_index(
    embedded_chunks: list[EmbeddedChunk]
) -> faiss.IndexFlatIP:
    if not embedded_chunks:
        raise ValueError("Cannot build an index without chunks")

    matrix = np.array(
        [item.vector.values for item in embedded_chunks],
        dtype=np.float32
    )

    faiss.normalize_L2(matrix)
    print("Vector dimension: ", matrix.shape)

    dim = matrix.shape
    index = faiss.IndexFlatIP(dim[1])

    index.add(matrix)
    return index

def search_faiss(
    request: SearchRequest,
    index: faiss.Index,
    chunks: list[ChunkRecord],
    model: SentenceTransformer
) -> list[SearchResult]:
    query = request.query.strip()

    if not query:
        raise ValueError("Query must contain non-whitespace text")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk counts doesn't match")

    if index.ntotal == 0:
        return []

    query_matrix = np.array(
        model.encode([query]),
        dtype=np.float32
    )

    if query_matrix.shape[1] != index.d:
        raise ValueError("Query vector dimensions do not match the index")

    faiss.normalize_L2(query_matrix)
    k = min(request.top_k, index.ntotal)

    scores, positions = index.search(query_matrix, k)

    results = []
    for position, score in zip(
        positions[0],
        scores[0],
        strict=True
    ):
        chunk = chunks[int(position)]

        result = SearchResult(
            chunk=chunk,
            score=float(score)
        )
        results.append(result)

    return results
