from fastapi import BackgroundTasks

import backend.app.query as query_module
from backend.evaluation import get_evaluation_state, initialize_evaluation_store
from backend.schemas import AnswerDraft, IndexManifest, QueryPlan, SearchResult


class FakeRunnable:
    """Stands in for an LLM chain: .invoke() returns a fixed value."""

    def __init__(self, value):
        self.value = value

    def invoke(self, _inputs):
        return self.value


def test_out_of_scope_question_returns_a_pollable_evaluation_id(tmp_path, monkeypatch):
    db_path = tmp_path / "state.sqlite3"
    initialize_evaluation_store(db_path)

    out_of_scope = QueryPlan(route="out_of_scope", queries=[])
    monkeypatch.setattr(query_module, "build_query_planner", lambda: FakeRunnable(out_of_scope))
    logged = []
    monkeypatch.setattr(query_module, "save_evaluation_record", lambda **kw: logged.append(kw))

    response = query_module.run_question(
        "How do I bake a chocolate cake?",
        index=None,
        chunks=[],
        manifest=None,
        embedding_model=None,
        reranker=None,
        background_tasks=BackgroundTasks(),
        db_path=db_path,
    )

    assert response.evaluation_status == "skipped"
    assert response.sources == []
    state = get_evaluation_state(response.evaluation_id, db_path)
    assert state is not None and state.status == "skipped"
    assert logged[0]["status"] == "skipped"


def test_in_scope_question_answers_with_cited_pages_and_queues_evaluation(
    tmp_path, monkeypatch, make_context
):
    db_path = tmp_path / "state.sqlite3"
    initialize_evaluation_store(db_path)

    retrieved = [SearchResult(chunk=s.chunk, score=0.9) for s in make_context(2).sources]
    monkeypatch.setattr(query_module, "build_query_planner",
                        lambda: FakeRunnable(QueryPlan(route="single", queries=["annual leave days"])))
    monkeypatch.setattr(query_module, "retrieve_for_plan", lambda **kwargs: retrieved)
    monkeypatch.setattr(query_module, "build_answer_chain", lambda: FakeRunnable(
        AnswerDraft(answer="You get 20 days of leave [S2] [S7].", source_ids=["S2", "S7"])
    ))

    background_tasks = BackgroundTasks()
    response = query_module.run_question(
        "How many days of annual leave do I get?",
        index=None,
        chunks=[],
        manifest=IndexManifest(embedding_model="test-model", embedding_revision="0" * 40,
                               dimensions=384, chunk_count=2),
        embedding_model=None,
        reranker=None,
        background_tasks=background_tasks,
        db_path=db_path,
    )

    assert response.answer == "You get 20 days of leave [S2]."
    assert [s.chunk.metadata.page_number for s in response.sources] == [2]
    assert response.evaluation_status == "pending"
    assert get_evaluation_state(response.evaluation_id, db_path).status == "pending"
    assert len(background_tasks.tasks) == 1 