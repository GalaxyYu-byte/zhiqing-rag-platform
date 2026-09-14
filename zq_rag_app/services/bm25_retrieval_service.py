"""基于 ParadeDB pg_search 的 BM25 关键词召回服务。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .retrieval_service import RetrievedChunk


@dataclass(slots=True, frozen=True)
class BM25RetrievalResult:
    query: str
    engine: str
    latency_ms: int
    results: list[RetrievedChunk]


async def search_by_bm25(
    session: AsyncSession,
    *,
    query: str,
    kb_ids: list[int],
    top_k: int,
    doc_ids: list[int] | None = None,
) -> BM25RetrievalResult:
    """使用 ``pg_search`` 的 ``|||`` 操作符和 ``pdb.score`` 排序。"""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    normalized_kb_ids = sorted(set(kb_ids))
    if not normalized_kb_ids or any(kb_id <= 0 for kb_id in normalized_kb_ids):
        raise ValueError("kb_ids 必须包含至少一个有效知识库 ID")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1 到 100 之间")
    normalized_doc_ids: list[int] | None = None
    if doc_ids is not None:
        normalized_doc_ids = sorted(set(doc_ids))
        if (
            not normalized_doc_ids
            or any(doc_id <= 0 for doc_id in normalized_doc_ids)
        ):
            raise ValueError("doc_ids 必须包含至少一个有效文档 ID")

    doc_filter = ""
    parameters: dict[str, object] = {
        "query": normalized_query,
        "kb_ids": normalized_kb_ids,
        "top_k": top_k,
    }
    if normalized_doc_ids is not None:
        doc_filter = "AND c.doc_id = ANY(:doc_ids)"
        parameters["doc_ids"] = normalized_doc_ids

    # 参数只承载数据；动态片段是固定 SQL，不拼接用户输入。
    statement = text(
        f"""
        SELECT
            c.id AS chunk_id,
            c.doc_id,
            d.file_name AS document,
            c.chunk_index,
            c.section_title AS section,
            c.page_num AS page,
            c.content,
            c.token_count,
            pdb.score(c.id) AS score
        FROM kb_doc_chunk AS c
        JOIN kb_document AS d ON d.id = c.doc_id
        WHERE c.content ||| :query
          AND c.kb_id = ANY(:kb_ids)
          AND d.is_deleted = FALSE
          AND d.status = 'DONE'
          AND c.doc_version = d.version
          {doc_filter}
        ORDER BY score DESC, c.id ASC
        LIMIT :top_k
        """
    )
    started_at = time.perf_counter()
    rows = (await session.execute(statement, parameters)).all()
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
            score=float(row.score),
        )
        for rank, row in enumerate(rows, start=1)
    ]
    latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
    return BM25RetrievalResult(
        query=normalized_query,
        engine="pg_search",
        latency_ms=latency_ms,
        results=results,
    )
