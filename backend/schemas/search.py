from pydantic import BaseModel, Field

from backend.schemas.documents import ChunkRecord


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1)

class SearchResult(BaseModel):
    chunk: ChunkRecord
    score: float

class RerankedResult(BaseModel):
    candidate: SearchResult
    retrieval_rank: int = Field(ge=1)
    rerank_score: float
