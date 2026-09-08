from pydantic import BaseModel, Field


class PageMetaData(BaseModel):
    source: str = Field(min_length=1)
    page_number: int = Field(ge=1)

class PageRecord(BaseModel):
    text: str
    metadata: PageMetaData

class ChunkRecord(BaseModel):
    text: str = Field(min_length=1)
    metadata: PageMetaData
    chunk_number: int = Field(ge=1)
