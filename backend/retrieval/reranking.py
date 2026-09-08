from sentence_transformers import CrossEncoder

from backend.config import DEVICE, RERANKER_MODEL
from backend.schemas import RerankedResult, SearchRequest, SearchResult

def load_reranker() -> CrossEncoder:
    return CrossEncoder(
        RERANKER_MODEL,
        device=DEVICE,
    )

def rerank_candidate(
    request: SearchRequest,
    candidates: list[SearchResult],
    reranker: CrossEncoder
) -> list[RerankedResult]:
    query = request.query.strip()

    if not query:
        raise ValueError("Query must contain non-whitespace text")

    if not candidates:
        return []

    pairs = [
        (query, candidate.chunk.text)
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    reranked: list[RerankedResult] = []

    for retrieval_rank, (candidate, score) in enumerate(
        zip(candidates, scores, strict=True),
        start=1
    ):
        chunk = RerankedResult(
            candidate=candidate,
            retrieval_rank=retrieval_rank,
            rerank_score=float(score)
        )

        reranked.append(chunk)
    reranked_results = sorted(
        reranked,
        key = lambda item: item.rerank_score,
        reverse=True
    )
    return reranked_results[:request.top_k]
