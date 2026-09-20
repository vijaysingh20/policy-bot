import numpy as np
import pytest

from backend.embedding.encoder import embed_document_chunks
from backend.retrieval.reranking import rerank_candidate
from backend.retrieval.vector_index import build_faiss_index, search_faiss
from backend.retrieval.workflow import retrieve_for_plan
from backend.schemas import ChunkRecord, PageMetaData, QueryPlan, SearchRequest, SearchResult

VOCABULARY = ["annual", "sick", "leave", "salary", "payroll", "remote", "work", "policy"]


class FakeEncoder:
    def encode(self, texts, **kwargs):
        return np.array(
            [[text.lower().split().count(word) for word in VOCABULARY] for text in texts],
            dtype=np.float32,
        )


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return np.array(
            [len(set(query.lower().split()) & set(text.lower().split())) for query, text in pairs],
            dtype=np.float32,
        )


def chunk(text: str, page: int) -> ChunkRecord:
    return ChunkRecord(text=text, metadata=PageMetaData(source="handbook.pdf", page_number=page),
                       chunk_number=1)


CHUNKS = [
    chunk("annual leave policy", 1),
    chunk("sick leave policy", 2),
    chunk("salary payroll policy", 3),
    chunk("remote work policy", 4),
]


@pytest.fixture(scope="module")
def index():
    return build_faiss_index(embed_document_chunks(CHUNKS, FakeEncoder()))


def pages(results):
    return [r.chunk.metadata.page_number for r in results]


# --------------- FAISS search --------------------

def test_search_returns_the_most_similar_chunk_first(index):
    results = search_faiss(SearchRequest(query="sick leave", top_k=2), index, CHUNKS, FakeEncoder())
    assert pages(results)[0] == 2
    assert len(results) == 2


def test_search_rejects_an_index_that_does_not_match_its_chunks(index):
    with pytest.raises(ValueError, match="counts"):
        search_faiss(SearchRequest(query="leave", top_k=2), index, CHUNKS[:2], FakeEncoder())


# --------------- Re-ranking  --------------------     

def test_rerank_reorders_by_cross_encoder_score_and_keeps_original_rank():
    candidates = [SearchResult(chunk=c, score=0.5) for c in CHUNKS]      # FAISS order: 1,2,3,4
    reranked = rerank_candidate(SearchRequest(query="remote work", top_k=2),
                                candidates, FakeCrossEncoder())

    assert [r.candidate.chunk.metadata.page_number for r in reranked] == [4, 1]
    assert reranked[0].retrieval_rank == 4
    assert len(reranked) == 2  


def test_rerank_of_no_candidates_is_empty():
    assert rerank_candidate(SearchRequest(query="leave", top_k=3), [], FakeCrossEncoder()) == []


# --------------- Full plan: search -> rerank -> merge sub-queries  --------------------

def test_decomposed_plan_merges_sub_query_results_without_duplicates(index):
    plan = QueryPlan(route="decompose", queries=["annual leave", "sick leave"])
    results = retrieve_for_plan(plan=plan, index=index, chunks=CHUNKS,
                                embedding_model=FakeEncoder(), reranker=FakeCrossEncoder())

    assert pages(results)[0] == 1
    assert 2 in pages(results)
    assert len(pages(results)) == len(set(pages(results)))