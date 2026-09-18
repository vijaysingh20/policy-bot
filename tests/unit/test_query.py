from fastapi import BackgroundTasks

import backend.app.query as query_module
from backend.evaluation import get_evaluation_state, initialize_evaluation_store
from backend.schemas import QueryPlan


class FakePlanner:
    def __init__(self, plan: QueryPlan):
        self.plan = plan

    def invoke(self, _inputs):
        return self.plan


def test_out_of_scope_question_returns_a_pollable_evaluation_id(tmp_path, monkeypatch):
    db_path = tmp_path / "state.sqlite3"
    initialize_evaluation_store(db_path)

    out_of_scope = QueryPlan(route="out_of_scope", queries=[])
    monkeypatch.setattr(query_module, "build_query_planner", lambda: FakePlanner(out_of_scope))
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
        db_path=db_path
    )

    assert response.evaluation_status == "skipped"
    assert response.sources == []
    state = get_evaluation_state(response.evaluation_id, db_path)
    assert state is not None and state.status == "skipped"
    assert logged[0]["status"] == "skipped"