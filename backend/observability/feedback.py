from uuid import UUID

from langsmith import Client

from backend.config import CHAT_MODEL
from backend.evaluation.prompts import CONTEXT_PRECISION_PROMPT_VERSION
from backend.schemas import EvaluationScores

def log_evaluation_feedback(
    run_id: UUID,
    project_name: str,
    scores: EvaluationScores
) -> None:
    client = Client()

    project = client.read_project(project_name=project_name)

    for metric_name, score in scores.model_dump().items():
        comment = f"RAGAS evaluation using {CHAT_MODEL}"

        if metric_name == "context_precision":
            comment += (
                "; variant=without_reference"
                f"; prompt_version={CONTEXT_PRECISION_PROMPT_VERSION}"
            )

        client.create_feedback(
            run_id=run_id,
            session_id=project.id,
            key=metric_name,
            score=score,
            comment=comment
        )