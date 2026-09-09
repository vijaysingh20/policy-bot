from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, model_validator
from backend.schemas.answers import ContextSource
from backend.schemas.evaluation import EvaluationScores

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = Field(min_length=1)

class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=2000
    )

class QuestionResponse(BaseModel):
    answer: str = Field(min_length=1)
    sources: list[ContextSource]
    evaluation_id: UUID
    evaluation_status: Literal[
        "pending", "completed", "skipped", "failed"
    ]
    scores: EvaluationScores | None = None
    trace_id: UUID | None = None

class EvaluationState(BaseModel):
    evaluation_id: UUID

    status: Literal[
        "pending",
        "completed",
        "skipped",
        "failed"
    ] = "pending"

    scores: EvaluationScores | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_evaluation_state(self) -> "EvaluationState":
        if self.status == "pending":
            if self.scores is not None or self.reason is not None:
                raise ValueError(
                    "Pending evaluations must have no scores or reason"
                )

        elif self.status == "completed":
            if self.scores is None:
                raise ValueError(
                    "Completed evaluations require scores"
                )

            if self.reason is not None:
                raise ValueError(
                    "Completed evaluations must have no reason"
                )

        else:
            if self.scores is not None:
                raise ValueError(
                    "Skipped or failed evaluations must have no scores"
                )

            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    "Skipped or failed evaluations require a non-blank reason"
                )

        return self