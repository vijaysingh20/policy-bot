from pathlib import Path
from uuid import UUID

from backend.config import CHAT_MODEL, EVALUATION_LOG
from backend.evaluation.prompts import CONTEXT_PRECISION_PROMPT_VERSION
from backend.schemas import (
    AnswerDraft,
    EvaluationRecord,
    EvaluationScores,
    QueryPlan,
    RetrievalContext
)
from backend.schemas.evaluation import EvaluationStatus

def append_evaluation_record(
    record: EvaluationRecord,
    log_path: Path
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    serialized_record = record.model_dump_json()
    with log_path.open("a", encoding="utf-8") as file:
        file.write(serialized_record + "\n")


def save_evaluation_record(
    *,
    trace_id: UUID | None = None,
    question: str,
    plan: QueryPlan,
    draft: AnswerDraft,
    context: RetrievalContext,
    scores: EvaluationScores | None = None,
    status: EvaluationStatus = "completed",
    reason: str | None = None,
    log_path: Path = EVALUATION_LOG
) -> EvaluationRecord:
    record = EvaluationRecord(
        trace_id=trace_id,
        question=question,
        query_plan=plan,
        draft=draft,
        context=context,
        scores=scores,
        status=status,
        reason=reason,
        evaluator_model=CHAT_MODEL,
        context_precision_prompt_version=CONTEXT_PRECISION_PROMPT_VERSION
    )
    append_evaluation_record(record, log_path)
    return record