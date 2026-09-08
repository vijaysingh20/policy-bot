from typing import Literal

from pydantic import BaseModel, Field, model_validator

from backend.schemas.documents import ChunkRecord


class ContextSource(BaseModel):
    source_id: str = Field(min_length=1)
    chunk: ChunkRecord

class RetrievalContext(BaseModel):
    context_text: str
    sources: list[ContextSource]

class AnswerDraft(BaseModel):
    answer: str = Field(
        description=(
            "The user-facing answer. Include inline citations "
            "such as [S1] beside supported policy claims."
        )
    )
    source_ids: list[str] = Field(
        description=(
            "Unique source IDs actually cited in the answer, "
            "without brackets. Use an empty list when no "
            "supported answer can be given."
        )
    )

class QueryPlan(BaseModel):
    route: Literal["single", "decompose", "out_of_scope"] = Field(
        description=(
            "Choose single for one focused retrieval question, "
            "decompose for multiple retrieval questions, "
            "or out_of_scope for questions unrelated to HR policies."
        )
    )

    queries: list[str] = Field(
        description=(
            "Exactly one retrieval question for single; "
            "two or more focused retrieval questions for decompose; "
            "an empty list for out_of_scope."
        )
    )

    @model_validator(mode="after")
    def validate_route_queries(self) -> "QueryPlan":
        if any(not query.strip() for query in self.queries):
            raise ValueError("Retrieval questions must not be blank")

        query_count = len(self.queries)

        if self.route == "single" and query_count != 1:
            raise ValueError("Route single requires exactly one query")

        elif self.route == "decompose" and query_count < 2:
            raise ValueError("Route decompose requires at least two queries")

        elif self.route == "out_of_scope" and query_count != 0:
            raise ValueError("Route out_of_scope requires an empty query list")
        return self
