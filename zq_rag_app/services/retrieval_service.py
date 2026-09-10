"""基于 pgvector HNSW 索引的余弦相似度召回服务。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.document import DocChunk, Document
from .embedding_service import EmbeddingService, get_embedding_service


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
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


@dataclass(slots=True, frozen=True)
class CosineRetrievalResult:
    query: str
    embedding_model: str
    dimensions: int
    latency_ms: int
    results: list[RetrievedChunk]


async def search_by_cosine(
    session: AsyncSession,
    *,
    query: str,
    kb_ids: list[int],
    top_k: int,
    min_score: float,
    embedding_service: EmbeddingService | None = None,
) -> CosineRetrievalResult:
    """生成 Query 向量并按余弦相似度返回当前版本的文档分块。

    ``vector_cosine_ops`` HNSW 索引与 ``cosine_distance`` 使用同一距离定义。
    返回给前端的 score 统一换算为 ``1 - cosine_distance``，越大越相关。
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    if not kb_ids or any(kb_id <= 0 for kb_id in kb_ids):
        raise ValueError("kb_ids 必须包含至少一个有效知识库 ID")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1 到 100 之间")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score 必须在 0 到 1 之间")

    started_at = time.perf_counter()
    service = embedding_service or get_embedding_service()
    embedding_result = await service.embed_many([normalized_query])
    query_vector = embedding_result.vectors[0]

    distance = DocChunk.embedding.cosine_distance(query_vector)
    score = (1.0 - distance).label("score")
    statement = (
        select(
            DocChunk.id.label("chunk_id"),
            DocChunk.doc_id,
            Document.file_name.label("document"),
            DocChunk.chunk_index,
            DocChunk.section_title.label("section"),
            DocChunk.page_num.label("page"),
            DocChunk.content,
            DocChunk.token_count,
            score,
        )
        .join(Document, Document.id == DocChunk.doc_id)
        .where(
            DocChunk.kb_id.in_(sorted(set(kb_ids))),
            Document.is_deleted.is_(False),
            Document.status == "DONE",
            # 重建索引期间旧版本仍可用；完成后只返回文档当前版本。
            DocChunk.doc_version == Document.version,
            distance <= 1.0 - min_score,
        )
        .order_by(distance.asc(), DocChunk.id.asc())
        .limit(top_k)
    )
    rows = (await session.execute(statement)).all()
    results = [
        RetrievedChunk(
            rank=rank,
            chunk_id=int(row.chunk_id),
            doc_id=int(row.doc_id),
            document=str(row.document),
            chunk_index=int(row.chunk_index),
            section=row.section,
            page=row.page,
            content=str(row.content),
            token_count=int(row.token_count),
            # pgvector/驱动可能返回 Decimal，这里统一成 JSON 友好的 float。
            score=float(row.score),
        )
        for rank, row in enumerate(rows, start=1)
    ]
    latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
    return CosineRetrievalResult(
        query=normalized_query,
        embedding_model=service.model,
        dimensions=service.dimensions,
        latency_ms=latency_ms,
        results=results,
    )
