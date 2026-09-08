import faiss
from langsmith import traceable
from sentence_transformers import CrossEncoder, SentenceTransformer

from backend.config import CANDIDATE_TOP_K, FINAL_TOP_K
from backend.retrieval.reranking import rerank_candidate
from backend.retrieval.vector_index import search_faiss
from backend.schemas import ChunkRecord, QueryPlan, SearchRequest, SearchResult

@traceable(
    name="retrieve_for_plan",
    run_type="chain",
    process_inputs=lambda inputs: {
        "plan": inputs["plan"].model_dump(),
    },
)
def retrieve_for_plan(
    plan: QueryPlan,
    index: faiss.Index,
    chunks: list[ChunkRecord],
    embedding_model: SentenceTransformer,
    reranker: CrossEncoder
) -> list[SearchResult]:
    merged_result: list[SearchResult] = []
    seen: set[tuple[str, int, int]] = set()

    for subquery in plan.queries:
        candidate_request = SearchRequest(
            query=subquery,
            top_k=CANDIDATE_TOP_K
        )

        final_request = SearchRequest(
            query=subquery,
            top_k=FINAL_TOP_K
        )

        candidates = search_faiss(
            request=candidate_request,
            index=index,
            chunks=chunks,
            model=embedding_model,
        )

        reranked = rerank_candidate(
            final_request,
            candidates,
            reranker
        )

        for item in reranked:
            result = item.candidate
            chunk = result.chunk

            key = (
                chunk.metadata.source,
                chunk.metadata.page_number,
                chunk.chunk_number
            )

            if key not in seen:
                seen.add(key)
                merged_result.append(result)

    return merged_result