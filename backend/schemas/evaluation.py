from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Literal
from pydantic import model_validator
from backend.schemas.answers import QueryPlan, AnswerDraft, RetrievalContext


class EvaluationInput(BaseModel):
    user_input: str = Field(min_length=1)
    response: str = Field(min_length=1)
    retrieved_contexts: list[str]

class EvaluationScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1, allow_inf_nan=False)
    answer_relevancy: float = Field(ge=-1, le=1, allow_inf_nan=False)
    context_precision: float = Field(ge=0, le=1, allow_inf_nan=False)

class EvaluationRecord(BaseModel):
    record_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    trace_id: UUID | None = None
    question: str = Field(min_length=1)
    query_plan: QueryPlan
    draft: AnswerDraft
    context: RetrievalContext
    scores: EvaluationScores | None = None
    status: Literal["completed", "skipped", "failed"] = "completed"
    reason: str | None = None
    evaluator_model: str = Field(min_length=1)
    context_precision_prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evaluation_status(self) -> "EvaluationRecord":
        if self.status == "completed":
            if self.scores is None:
                raise ValueError("Score must be there for completed evaluations")
            elif self.reason is not None:
                raise ValueError("Reason must be none for completed evaluations")
        elif self.status in ["skipped", "failed"]:
            if self.scores is not None:
                raise ValueError("Scores must be none for skipped or failed evaluations")
            elif self.reason is None or not self.reason.strip():
                raise ValueError(
                    "A non-blank reason is required for skipped or failed evaluations"
                )
        
        return self