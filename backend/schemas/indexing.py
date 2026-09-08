from typing import Literal

from pydantic import BaseModel, Field

from backend.schemas.documents import ChunkRecord


class Vector(BaseModel):
    values: list[float] = Field(min_length=1)

class EmbeddedChunk(BaseModel):
    chunk: ChunkRecord
    vector: Vector

class IndexManifest(BaseModel):
    schema_version: Literal[1] = 1

    embedding_model: str = Field(min_length=1)
    embedding_revision: str = Field(
        pattern=r"^[0-9a-f]{40}$"
    )

    dimensions: int = Field(ge=1)
    chunk_count: int = Field(ge=1)

    index_type: Literal["IndexFlatIP"] = "IndexFlatIP"
    similarity: Literal["cosine"] = "cosine"
    normalized: Literal[True] = True
