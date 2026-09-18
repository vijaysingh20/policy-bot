import logging
from pathlib import Path

from backend.evaluation.scoring import score_answer
from backend.evaluation.state_store import finish_evaluation_state
from backend.evaluation.storage import save_evaluation_record
from backend.schemas import EvaluationJob, EvaluationState

logger = logging.getLogger(__name__)


def run_evaluation_job(job: EvaluationJob, db_path: Path) -> None:
    try:
        scores = score_answer(
            question=job.question,
            draft=job.draft,
            context=job.context,
            manifest=job.manifest
        )
        final_state = EvaluationState(
            evaluation_id=job.evaluation_id, status="completed", scores=scores
        )
    except Exception as exc:
        logger.exception("Evaluation failed for job %s", job.evaluation_id)
        final_state = EvaluationState(
            evaluation_id=job.evaluation_id,
            status="failed",
            reason=f"Evaluation failed with {type(exc).__name__}."
        )

    finish_evaluation_state(state=final_state, db_path=db_path)

    try:
        save_evaluation_record(
            question=job.question,
            plan=job.query_plan,
            draft=job.draft,
            context=job.context,
            scores=final_state.scores,
            status=final_state.status,
            reason=final_state.reason
        )
    except Exception:
        logger.exception("Could not write the evaluation log for job %s", job.evaluation_id)