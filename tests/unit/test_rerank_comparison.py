"""The before/after harness, driven by fakes so the expected numbers are known exactly."""

import numpy as np
import pytest

from backend.embedding.encoder import embed_document_chunks
from backend.evaluation.rerank_comparison import (
    QuestionResult,
    evaluate_question,
    load_golden_set,
    reciprocal_rank,
    render_markdown,
    summarize,
)
from backend.retrieval.vector_index import build_faiss_index
from backend.schemas import ChunkRecord, PageMetaData


class LeaveObsessedEncoder:
    """A poor retriever: it only notices the word "leave" (plus a small constant)."""

    def encode(self, texts, **kwargs):
        return np.array([[t.lower().split().count("leave"), 0.1] for t in texts],
                        dtype=np.float32)


class KeywordCrossEncoder:
    """A sensible re-ranker: scores by shared words between question and chunk."""

    def predict(self, pairs, **kwargs):
        return np.array([len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs],
                        dtype=np.float32)


CHUNKS = [
    ChunkRecord(text=text, metadata=PageMetaData(source="handbook.pdf", page_number=page),
                chunk_number=1)
    for page, text in [(1, "annual leave policy"), (2, "sick leave policy"),
                       (3, "salary payroll policy"), (4, "remote work policy")]
]


@pytest.fixture(scope="module")
def index():
    return build_faiss_index(embed_document_chunks(CHUNKS, LeaveObsessedEncoder()))


@pytest.mark.parametrize("pages, relevant, expected", [
    ([4, 1, 2], {4}, 1.0),
    ([1, 4, 2], {4}, 0.5),
    ([1, 2, 4], {4}, 1 / 3),
    ([1, 2, 3], {4}, 0.0),
    ([1, 2, 3], {2, 3}, 0.5),          # first relevant page counts
])
def test_reciprocal_rank(pages, relevant, expected):
    assert reciprocal_rank(pages, relevant) == pytest.approx(expected)


def test_reranker_rescues_a_chunk_the_retriever_ranked_low(index):
    item = {"id": "q1", "category": "paraphrase", "question": "remote work leave rules",
            "relevant_pages": [4]}
    result = evaluate_question(item, index=index, chunks=CHUNKS,
                               embedding_model=LeaveObsessedEncoder(),
                               reranker=KeywordCrossEncoder(), top_k=2, candidate_k=4)

    assert set(result.candidate_pages) == {1, 2, 3, 4}  # one shared pool for both conditions
    assert set(result.before_pages) == {1, 2}          # FAISS alone: the two "leave" chunks
    assert result.after_pages[0] == 4                  # re-ranker moves it to the top
    assert (result.before_rr, result.after_rr) == (0.0, 1.0)


def result(category, before_rr, after_rr, pool_hit=True):
    return QuestionResult(id="x", category=category, question="q", relevant_pages=[9],
                          candidate_pages=[9] if pool_hit else [1], before_pages=[],
                          after_pages=[], before_rr=before_rr, after_rr=after_rr)


def test_summary_is_per_category_plus_all():
    summary = summarize([result("simple", 1.0, 1.0), result("paraphrase", 0.0, 0.5),
                         result("paraphrase", 0.5, 1.0, pool_hit=False)])

    assert summary["all"]["questions"] == 3
    assert summary["paraphrase"]["before_hit"] == pytest.approx(0.5)
    assert summary["paraphrase"]["after_hit"] == pytest.approx(1.0)
    assert summary["paraphrase"]["after_mrr"] == pytest.approx(0.75)
    assert summary["paraphrase"]["pool_recall"] == pytest.approx(0.5)


def test_markdown_report_lists_all_last_and_shows_changed_questions():
    results = [result("simple", 1.0, 1.0), result("paraphrase", 0.0, 0.5)]
    report = render_markdown(summarize(results), results, top_k=3, pool=20)

    lines = report.splitlines()
    assert "Hit@3 before" in lines[0]
    assert lines[4].startswith("| **all** |")              # after simple, paraphrase
    assert lines[-1].startswith("| q |")                   # the one question that changed


def test_golden_set_is_well_formed():
    items = load_golden_set()
    assert len({item["id"] for item in items}) == len(items)
    for item in items:
        assert item["question"].strip()
        assert item["expected_route"] in {"single", "decompose", "out_of_scope"}
        has_pages = bool(item["relevant_pages"])
        assert has_pages == (item["category"] not in {"out_of_scope", "not_in_document"}), item["id"]
        assert all(isinstance(p, int) and 1 <= p <= 43 for p in item["relevant_pages"])