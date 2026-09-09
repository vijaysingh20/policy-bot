import logging
from pathlib import Path

from backend.observability import log_evaluation_feedback
from backend.evaluation import finish_evaluation_state
from backend.schemas import EvaluationState, EvaluationJob
from backend.helper import score_answer, save_evaluation_records

logger = logging.getLogger(__name__)

def run_evaluation_job(
    job: EvaluationJob,
    db_path: Path
)-> None:
    try:
        scores = score_answer(
            question=job.question,
            draft=job.draft,
            context=job.context,
            manifest=job.manifest
        )

        final_state = EvaluationState(
            evaluation_id=job.evaluation_id,
            status="completed",
            scores=scores
        )
    except Exception as exc:
        logger.exception(
            "Evaluation failed for job %s",
            job.evaluation_id,
        )

        final_state = EvaluationState(
            evaluation_id=job.evaluation_id,
            status="failed",
            reason=f"Evaluation failed with {type(exc).__name__}."
        )

    finish_evaluation_state(
        state=final_state,
        db_path=db_path
    )

    try:
        save_evaluation_records(
            trace_id=job.trace_id,
            question=job.question,
            plan=job.query_plan,
            draft=job.draft,
            context=job.context,
            scores=final_state.scores,
            status=final_state.status,
            reason=final_state.reason,
        )
    except Exception:
        logger.exception(
            "Could not write the evaluation log for job %s",
            job.evaluation_id,
        )

    if (
        final_state.scores is not None
        and job.trace_id is not None
        and job.project_name is not None
    ):
        try:
            log_evaluation_feedback(
                run_id=job.trace_id,
                project_name=job.project_name,
                scores=final_state.scores,
            )
        except Exception:
            logger.exception(
                "Could not attach feedback for job %s",
                job.evaluation_id,
            )
