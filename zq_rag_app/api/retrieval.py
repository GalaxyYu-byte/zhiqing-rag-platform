"""向量、图谱及混合召回调试接口。"""

import logging
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.config import settings
from ..core.security import CurrentUser, UserContext
from ..services.bm25_retrieval_service import search_by_bm25
from ..services.graph_retrieval_service import (
    GraphRetrievalResult,
    search_by_graph,
)
from ..services.hybrid_retrieval_service import fuse_dense_bm25_graph_rrf
from ..services.permission_service import (
    PermissionLevel,
    has_knowledge_base_permission,
    list_accessible_document_ids,
)
from ..services.reranker_service import RerankerService
from ..services.retrieval_service import search_by_cosine


logger = logging.getLogger(__name__)
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


class GraphSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    kb_ids: list[int] = Field(min_length=1, max_length=50)
    doc_ids: list[int] | None = Field(default=None, min_length=1, max_length=200)
    top_k: int = Field(default=10, ge=1, le=100)
    max_hops: int = Field(default=2, ge=1, le=2)
    entity_limit: int = Field(default=20, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空")
        return normalized

    @field_validator("kb_ids", "doc_ids")
    @classmethod
    def validate_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(item <= 0 for item in value):
            raise ValueError("知识库和文档 ID 必须大于 0")
        return list(dict.fromkeys(value))


class GraphClaimMatchResponse(BaseModel):
    claim_uid: str
    subject_uid: str
    subject: str
    predicate: str
    object_uid: str
    object: str
    polarity: str
    valid_from: str | None
    valid_to: str | None
    evidence_quote: str
    confidence: float
    chunk_uid: str
    kb_id: int
    doc_id: int
    doc_version: int
    chunk_index: int
    page: int | None
    section: str | None
    hop: int
    score: float


class GraphSearchResponse(BaseModel):
    query: str
    engine: str
    latency_ms: int
    result_count: int
    claim_count: int
    results: list[RetrievedChunkResponse]
    claims: list[GraphClaimMatchResponse]


class HybridGraphSearchRequest(GraphSearchRequest):
    candidate_k: int = Field(default=30, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=100)
    min_dense_score: float = Field(default=0.3, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.4, ge=0.0)
    bm25_weight: float = Field(default=0.3, ge=0.0)
    graph_weight: float = Field(default=0.3, ge=0.0)
    rrf_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    rerank: bool = True


class HybridCandidateResponse(BaseModel):
    rank: int
    chunk_id: int
    fusion_rank: int
    fusion_score: float
    reranker_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    graph_rank: int | None
    dense_score: float | None
    bm25_score: float | None
    graph_score: float | None


class HybridGraphSearchResponse(BaseModel):
    query: str
    engine: str
    latency_ms: int
    reranked: bool
    result_count: int
    graph_claim_count: int
    results: list[RetrievedChunkResponse]
    candidates: list[HybridCandidateResponse]
    graph_claims: list[GraphClaimMatchResponse]


async def _resolve_accessible_doc_ids(
    session: AsyncSession,
    *,
    user: UserContext,
    kb_ids: list[int],
    requested_doc_ids: list[int] | None = None,
) -> list[int]:
    """在任何检索发生前统一收敛知识库和文档访问范围。"""

    for kb_id in kb_ids:
        allowed = await has_knowledge_base_permission(
            session,
            user=user,
            kb_id=kb_id,
            required=PermissionLevel.READ,
        )
        if not allowed:
            raise HTTPException(status_code=403, detail=f"无权读取知识库 {kb_id}")

    accessible_doc_ids = await list_accessible_document_ids(
        session,
        user=user,
        kb_ids=kb_ids,
        requested_doc_ids=requested_doc_ids,
    )
    if requested_doc_ids is not None and not set(requested_doc_ids).issubset(
        accessible_doc_ids
    ):
        raise HTTPException(status_code=403, detail="请求包含无权读取的文档")
    if not accessible_doc_ids:
        raise HTTPException(status_code=403, detail="当前范围内没有可读取文档")
    return accessible_doc_ids


@router.post("/search", response_model=CosineSearchResponse)
async def cosine_search(
    request: CosineSearchRequest,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> CosineSearchResponse:
    doc_ids = await _resolve_accessible_doc_ids(
        session,
        user=current_user,
        kb_ids=request.kb_ids,
    )
    try:
        result = await search_by_cosine(
            session,
            query=request.query,
            kb_ids=request.kb_ids,
            top_k=request.top_k,
            min_score=request.min_score,
            doc_ids=doc_ids,
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


@router.post("/graph-search", response_model=GraphSearchResponse)
async def graph_search(
    request: GraphSearchRequest,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> GraphSearchResponse:
    doc_ids = await _resolve_accessible_doc_ids(
        session,
        user=current_user,
        kb_ids=request.kb_ids,
        requested_doc_ids=request.doc_ids,
    )
    try:
        result = await search_by_graph(
            session,
            query=request.query,
            kb_ids=request.kb_ids,
            doc_ids=doc_ids,
            top_k=request.top_k,
            max_hops=request.max_hops,
            entity_limit=request.entity_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Graph 检索失败")
        raise HTTPException(status_code=503, detail="Graph 检索暂时不可用") from exc

    return GraphSearchResponse(
        query=result.query,
        engine=result.engine,
        latency_ms=result.latency_ms,
        result_count=len(result.results),
        claim_count=len(result.matches),
        results=[RetrievedChunkResponse(**asdict(item)) for item in result.results],
        claims=[GraphClaimMatchResponse(**asdict(item)) for item in result.matches],
    )


@router.post("/hybrid-graph-search", response_model=HybridGraphSearchResponse)
async def hybrid_graph_search(
    request: HybridGraphSearchRequest,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> HybridGraphSearchResponse:
    doc_ids = await _resolve_accessible_doc_ids(
        session,
        user=current_user,
        kb_ids=request.kb_ids,
        requested_doc_ids=request.doc_ids,
    )
    try:
        dense = await search_by_cosine(
            session,
            query=request.query,
            kb_ids=request.kb_ids,
            top_k=request.candidate_k,
            min_score=request.min_dense_score,
            doc_ids=doc_ids,
        )
        bm25 = await search_by_bm25(
            session,
            query=request.query,
            kb_ids=request.kb_ids,
            doc_ids=doc_ids,
            top_k=request.candidate_k,
        )
        try:
            graph = await search_by_graph(
                session,
                query=request.query,
                kb_ids=request.kb_ids,
                doc_ids=doc_ids,
                top_k=request.candidate_k,
                max_hops=request.max_hops,
                entity_limit=request.entity_limit,
            )
        except ValueError:
            raise
        except Exception:
            logger.exception("Graph 分支不可用，混合检索降级到 Dense + BM25")
            graph = GraphRetrievalResult(
                query=request.query,
                engine="neo4j_unavailable",
                latency_ms=0,
                results=[],
                matches=(),
            )
        hybrid = fuse_dense_bm25_graph_rrf(
            dense,
            bm25,
            graph,
            top_k=request.candidate_k,
            dense_weight=request.dense_weight,
            bm25_weight=request.bm25_weight,
            graph_weight=request.graph_weight,
            rrf_weight=request.rrf_weight,
            rrf_k=request.rrf_k,
        )

        reranked = request.rerank and bool(hybrid.results)
        if reranked:
            reranker = RerankerService()
            try:
                ranked = await reranker.rerank(
                    hybrid,
                    top_n=min(request.top_k, len(hybrid.results)),
                )
            finally:
                await reranker.aclose()
            results = ranked.results
            engine = ranked.engine
            latency_ms = ranked.latency_ms
            candidates = [
                HybridCandidateResponse(
                    rank=item.rank,
                    chunk_id=item.chunk_id,
                    fusion_rank=item.fusion_rank,
                    fusion_score=item.fusion_score,
                    reranker_score=item.reranker_score,
                    dense_rank=item.dense_rank,
                    bm25_rank=item.bm25_rank,
                    graph_rank=item.graph_rank,
                    dense_score=item.dense_score,
                    bm25_score=item.bm25_score,
                    graph_score=item.graph_score,
                )
                for item in ranked.candidate_scores
            ]
        else:
            results = hybrid.results[: request.top_k]
            engine = hybrid.engine
            latency_ms = hybrid.latency_ms
            candidates = [
                HybridCandidateResponse(
                    rank=item.rank,
                    chunk_id=item.chunk_id,
                    fusion_rank=item.rank,
                    fusion_score=item.final_score,
                    reranker_score=None,
                    dense_rank=item.dense_rank,
                    bm25_rank=item.bm25_rank,
                    graph_rank=item.graph_rank,
                    dense_score=item.dense_score,
                    bm25_score=item.bm25_score,
                    graph_score=item.graph_score,
                )
                for item in hybrid.candidate_scores[: request.top_k]
            ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Graph 混合检索失败")
        raise HTTPException(status_code=503, detail="混合检索暂时不可用") from exc

    return HybridGraphSearchResponse(
        query=request.query,
        engine=engine,
        latency_ms=latency_ms,
        reranked=reranked,
        result_count=len(results),
        graph_claim_count=len(graph.matches),
        results=[RetrievedChunkResponse(**asdict(item)) for item in results],
        candidates=candidates,
        graph_claims=[GraphClaimMatchResponse(**asdict(item)) for item in graph.matches],
    )
