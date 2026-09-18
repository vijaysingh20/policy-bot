from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str | None
    content_type: str | None
    page_count: int = Field(ge=1)
    size_bytes: int = Field(gt=0)
    status: Literal["ready", "uploaded"] = "uploaded"