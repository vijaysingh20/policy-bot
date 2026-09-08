from pydantic import BaseModel, Field


class EvaluationInput(BaseModel):
    user_input: str = Field(min_length=1)
    response: str = Field(min_length=1)
    retrieved_contexts: list[str]

class EvaluationScores(BaseModel):
    faithfulness: float = Field(ge=0, le=1, allow_inf_nan=False)
    answer_relevancy: float = Field(ge=-1, le=1, allow_inf_nan=False)
    context_precision: float = Field(ge=0, le=1, allow_inf_nan=False)
