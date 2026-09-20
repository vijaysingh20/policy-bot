from uuid import uuid4

from backend.evaluation import save_evaluation_record
from backend.schemas import AnswerDraft, EvaluationRecord, EvaluationScores, QueryPlan

SCORES = EvaluationScores(faithfulness=0.9, answer_relevancy=0.8, context_precision=0.7)


def save(log_path, context, **overrides):
    fields = dict(
        question="How many days of annual leave do I get?",
        plan=QueryPlan(route="single", queries=["annual leave days"]),
        draft=AnswerDraft(answer="20 days [S1].", source_ids=["S1"]),
        context=context,
        scores=SCORES,
        log_path=log_path,
    )
    return save_evaluation_record(**{**fields, **overrides})


def test_each_query_appends_one_json_line(tmp_path, make_context):
    log_path = tmp_path / "logs" / "evaluations.jsonl"     # parent dir does not exist yet
    save(log_path, make_context(1))
    save(log_path, make_context(1), status="skipped", scores=None, reason="Out of scope")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [EvaluationRecord.model_validate_json(line) for line in lines]
    assert [r.status for r in records] == ["completed", "skipped"]


def test_logged_record_keeps_scores_trace_and_cited_pages(tmp_path, make_context):
    log_path = tmp_path / "evaluations.jsonl"
    trace_id = uuid4()
    save(log_path, make_context(2), trace_id=trace_id)

    record = EvaluationRecord.model_validate_json(log_path.read_text(encoding="utf-8"))
    assert record.trace_id == trace_id
    assert record.scores == SCORES
    assert [s.chunk.metadata.page_number for s in record.context.sources] == [1, 2]