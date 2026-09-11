from pydantic import BaseModel, Field
from typing import Literal
from uuid import UUID

class UploadInfo(BaseModel):
    filename: str | None
    content_type: str | None
    page_count: int = Field(ge=1)

class DocumentUploadResponse(UploadInfo):
    document_id: UUID
    size_bytes: int = Field(gt=0)
    status: Literal["uploaded"] = "uploaded"