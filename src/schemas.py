from pydantic import BaseModel, Field
from typing import List


class UploadResponse(BaseModel):
    message: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]
