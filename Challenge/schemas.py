from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# ========== REQUEST ==========

class DocumentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)

class EmbeddingRequest(BaseModel):
    document_id: str = Field(..., min_length=1)

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1)

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1)

# ========== RESPONSE ==========

class DocumentResponse(BaseModel):
    message: str
    document_id: str

class EmbeddingResponse(BaseModel):
    message: str

class SearchResultItem(BaseModel):
    document_id: str
    title: str
    content_snippet: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)

class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class AnswerContext(BaseModel):
    document_id: str
    content_snippet: str
    similarity_score: float

class AnswerResponse(BaseModel):
    question: str
    answer: str
    context_used: Optional[List[AnswerContext]] = None
    grounded: bool = Field(..., description="Indica si la respuesta está basada en contexto real")

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    code: int