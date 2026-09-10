"""向量召回调试接口。"""

from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..services.retrieval_service import search_by_cosine


router = APIRouter(prefix="/retrieval", tags=["retrieval"])


class CosineSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    kb_ids: list[int] = Field(min_length=1, max_length=50)
    metric: Literal["cosine"] = "cosine"
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空")
        return normalized

    @field_validator("kb_ids")
    @classmethod
    def validate_kb_ids(cls, value: list[int]) -> list[int]:
        if any(kb_id <= 0 for kb_id in value):
            raise ValueError("知识库 ID 必须大于 0")
        # 去重但保持用户给出的顺序，便于日志和调试。
        return list(dict.fromkeys(value))


class RetrievedChunkResponse(BaseModel):
    rank: int
    chunk_id: int
    doc_id: int
    document: str
    chunk_index: int
    section: str | None
    page: int | None
    content: str
    token_count: int
    score: float


class CosineSearchResponse(BaseModel):
    query: str
    metric: Literal["cosine"] = "cosine"
    embedding_model: str
    dimensions: int
    latency_ms: int
    result_count: int
    results: list[RetrievedChunkResponse]


@router.post("/search", response_model=CosineSearchResponse)
async def cosine_search(
    request: CosineSearchRequest,
    session: AsyncSession = Depends(get_db),
) -> CosineSearchResponse:
    try:
        result = await search_by_cosine(
            session,
            query=request.query,
            kb_ids=request.kb_ids,
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response_results = [
        RetrievedChunkResponse(**asdict(item))
        for item in result.results
    ]
    return CosineSearchResponse(
        query=result.query,
        embedding_model=result.embedding_model,
        dimensions=result.dimensions,
        latency_ms=result.latency_ms,
        result_count=len(response_results),
        results=response_results,
    )
