"""Reference implementation of cosine search over in-memory vectors.

The faiss index in :mod:`backend.retrieval.vector_index` is what the
application uses; these helpers stay as the readable baseline.
"""

import math

from sentence_transformers import SentenceTransformer

from backend.schemas import EmbeddedChunk, SearchRequest, SearchResult, Vector


def dot_product(a:Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a.values, b.values))

def magnitude(v: Vector) -> float:
    return math.sqrt(sum(x ** 2 for x in v.values))

def cosine_similarity(a: Vector, b: Vector) -> float:
    if len(a.values) != len(b.values):
        raise ValueError("Both vectors must have the same number of dimensions")
    mag_a = magnitude(a)
    mag_b = magnitude(b)

    if mag_a == 0 or mag_b == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return dot_product(a, b) / (mag_a * mag_b)

def search_chunks(
    request: SearchRequest,
    embedded_chunks: list[EmbeddedChunk],
    model: SentenceTransformer
) -> list[SearchResult]:
    query = request.query.strip()

    if not query:
        raise ValueError("Query must contain non-whitespace text")

    if not embedded_chunks:
        return []

    query_embeddings = model.encode([query])
    query_vector = Vector(
        values=query_embeddings[0].tolist()
    )

    results = []

    for item in embedded_chunks:
        similarity = cosine_similarity(item.vector, query_vector)
        result = SearchResult(
            chunk=item.chunk,
            score=similarity
        )

        results.append(result)

    ranked_results = sorted(
        results,
        key=lambda item: item.score,
        reverse=True
    )
    return ranked_results[:request.top_k]
