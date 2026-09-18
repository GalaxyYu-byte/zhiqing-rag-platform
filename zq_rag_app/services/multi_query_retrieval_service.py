"""固定授权范围内串行扩展召回；先融合每个通道，再进行既有通道融合。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from ..query_analysis.multi_query import MultiQuerySummary
from .bm25_retrieval_service import BM25RetrievalResult
from .retrieval_service import CosineRetrievalResult, RetrievedChunk


logger = logging.getLogger(__name__)


def fuse_query_rankings(
    rankings: list[list[RetrievedChunk]], *, top_k: int, rrf_k: int = 60,
    original_weight: float = 0.6,
) -> list[RetrievedChunk]:
    """原问题保留较高权重，每张榜单中的同一 chunk 只投一次票。"""
    if not rankings or not 1 <= top_k <= 100 or rrf_k <= 0 or not 0 < original_weight < 1:
        raise ValueError("多查询融合参数无效")
    if len(rankings) == 1:
        return rankings[0][:top_k]
    weights = [original_weight, *[(1 - original_weight) / (len(rankings) - 1)] * (len(rankings) - 1)]
    scores: dict[int, float] = {}
    sources: dict[int, RetrievedChunk] = {}
    for chunks, weight in zip(rankings, weights, strict=True):
        seen = set()
        for chunk in chunks:
            if chunk.rank <= 0:
                raise ValueError("召回排名必须为正数")
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            sources.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (rrf_k + chunk.rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [replace(sources[chunk_id], rank=rank, score=min(1.0, scores[chunk_id] * (rrf_k + 1)))
            for rank, chunk_id in enumerate(ordered, start=1)]


@asynccontextmanager
async def _read_scope(session: Any):
    # SAVEPOINT 恢复可选分支的 SQL 失败/超时，避免回滚整个会话并过期已校验的 ORM 对象。
    begin_nested = getattr(session, "begin_nested", None)
    if begin_nested is None:
        yield
    else:
        async with begin_nested():
            yield


async def retrieve_expansions(
    session: Any, *, dense: CosineRetrievalResult, bm25: BM25RetrievalResult,
    expansion: MultiQuerySummary, dense_search: Callable[..., Any], bm25_search: Callable[..., Any],
    kb_ids: list[int], doc_ids: list[int] | None, candidate_k: int, timeout_seconds: float,
) -> tuple[CosineRetrievalResult, BM25RetrievalResult, MultiQuerySummary]:
    if dense.query != bm25.query or not expansion.queries or expansion.queries[0] != dense.query:
        raise ValueError("扩展召回必须属于同一原问题")
    if not 0 < timeout_seconds <= 30 or len(expansion.queries) > 3:
        raise ValueError("扩展召回预算或数量无效")
    if expansion.status != "expanded" or len(expansion.queries) == 1:
        return dense, bm25, replace(expansion, applied_queries=(dense.query,))
    dense_rankings, bm25_rankings = [dense.results], [bm25.results]
    applied, failed = [dense.query], []
    dense_latency, bm25_latency = dense.latency_ms, bm25.latency_ms
    failure = None
    pending = list(expansion.queries[1:])
    try:
        async with asyncio.timeout(timeout_seconds):
            for query in expansion.queries[1:]:
                try:
                    async with _read_scope(session):
                        new_dense = await dense_search(session, query=query, kb_ids=kb_ids, doc_ids=doc_ids,
                                                       top_k=candidate_k, min_score=0.3)
                        new_bm25 = await bm25_search(session, query=query, kb_ids=kb_ids, doc_ids=doc_ids,
                                                     top_k=candidate_k)
                        if new_dense.query != query or new_bm25.query != query:
                            raise ValueError("扩展服务返回了不同问题的候选")
                        if doc_ids is not None and any(c.doc_id not in doc_ids for c in [*new_dense.results, *new_bm25.results]):
                            raise ValueError("扩展结果超出授权文档范围")
                    # 两个通道成功且读事务正常退出后才接纳该查询的结果。
                    dense_rankings.append(new_dense.results)
                    bm25_rankings.append(new_bm25.results)
                    dense_latency += new_dense.latency_ms
                    bm25_latency += new_bm25.latency_ms
                    applied.append(query)
                except Exception:
                    failure = "retrieval_error"
                    failed.append(query)
                    logger.warning("Multi Query 可选召回失败: category=%s", failure)
                pending.remove(query)
    except TimeoutError:
        failure = "retrieval_timeout"
        failed.extend(pending)
    if len(applied) > 1:
        dense = replace(dense, latency_ms=dense_latency, results=fuse_query_rankings(dense_rankings, top_k=candidate_k))
        bm25 = replace(bm25, latency_ms=bm25_latency, engine=f"{bm25.engine}+multi_query_rrf",
                       results=fuse_query_rankings(bm25_rankings, top_k=candidate_k))
    summary = replace(
        expansion, status="degraded" if failure else "expanded", applied_queries=tuple(applied),
        failed_queries=tuple(failed), degradation_reason=failure,
        warnings=(*expansion.warnings, *(('部分扩展召回失败，保留原问题及已成功扩展的结果。',) if failure else ())),
    )
    return dense, bm25, summary
