"""Dense 与 BM25 的归一化加权及 RRF 融合。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .bm25_retrieval_service import BM25RetrievalResult
from .graph_retrieval_service import GraphRetrievalResult
from .retrieval_service import CosineRetrievalResult, RetrievedChunk


@dataclass(slots=True, frozen=True)
class HybridCandidateScore:
    rank: int
    chunk_id: int
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float | None
    bm25_score: float | None
    dense_normalized: float
    bm25_normalized: float
    normalized_weighted_score: float
    rrf_normalized: float
    final_score: float
    graph_rank: int | None = None
    graph_score: float | None = None
    graph_normalized: float = 0.0


@dataclass(slots=True, frozen=True)
class HybridRetrievalResult:
    query: str
    engine: str
    latency_ms: int
    results: list[RetrievedChunk]
    candidate_scores: tuple[HybridCandidateScore, ...]


def _min_max_scores(chunks: list[RetrievedChunk]) -> dict[int, float]:
    if not chunks:
        return {}
    scores = [float(chunk.score) for chunk in chunks]
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
        return {chunk.chunk_id: 1.0 for chunk in chunks}
    scale = high - low
    return {
        chunk.chunk_id: (float(chunk.score) - low) / scale
        for chunk in chunks
    }


def fuse_dense_bm25_rrf(
    dense: CosineRetrievalResult,
    bm25: BM25RetrievalResult,
    *,
    top_k: int,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    rrf_weight: float = 0.5,
    rrf_k: int = 60,
) -> HybridRetrievalResult:
    """融合两路候选，最终分数由归一化加权分和加权 RRF 分组成。"""

    if dense.query != bm25.query:
        raise ValueError("Dense 与 BM25 的 query 必须一致")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1 到 100 之间")
    if dense_weight < 0 or bm25_weight < 0:
        raise ValueError("Dense/BM25 权重不能小于 0")
    branch_weight_sum = dense_weight + bm25_weight
    if branch_weight_sum <= 0:
        raise ValueError("Dense/BM25 权重之和必须大于 0")
    if not 0.0 <= rrf_weight <= 1.0:
        raise ValueError("rrf_weight 必须在 0 到 1 之间")
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    started_at = time.perf_counter()
    normalized_dense_weight = dense_weight / branch_weight_sum
    normalized_bm25_weight = bm25_weight / branch_weight_sum
    dense_by_id = {chunk.chunk_id: chunk for chunk in dense.results}
    bm25_by_id = {chunk.chunk_id: chunk for chunk in bm25.results}
    dense_norm = _min_max_scores(dense.results)
    bm25_norm = _min_max_scores(bm25.results)
    dense_rank = {chunk.chunk_id: chunk.rank for chunk in dense.results}
    bm25_rank = {chunk.chunk_id: chunk.rank for chunk in bm25.results}

    scored: list[tuple[int, float, float, float]] = []
    for chunk_id in dense_by_id.keys() | bm25_by_id.keys():
        normalized_weighted_score = (
            normalized_dense_weight * dense_norm.get(chunk_id, 0.0)
            + normalized_bm25_weight * bm25_norm.get(chunk_id, 0.0)
        )
        rrf_score = 0.0
        if chunk_id in dense_rank:
            rrf_score += normalized_dense_weight / (
                rrf_k + dense_rank[chunk_id]
            )
        if chunk_id in bm25_rank:
            rrf_score += normalized_bm25_weight / (
                rrf_k + bm25_rank[chunk_id]
            )
        # 理论最大值为两路都排第 1，即 1 / (rrf_k + 1)。
        rrf_normalized = min(1.0, rrf_score * (rrf_k + 1))
        final_score = (
            (1.0 - rrf_weight) * normalized_weighted_score
            + rrf_weight * rrf_normalized
        )
        scored.append(
            (
                chunk_id,
                final_score,
                normalized_weighted_score,
                rrf_normalized,
            )
        )

    scored.sort(key=lambda row: (-row[1], -row[2], -row[3], row[0]))
    fused_results: list[RetrievedChunk] = []
    diagnostics: list[HybridCandidateScore] = []
    for rank, (
        chunk_id,
        final_score,
        normalized_weighted_score,
        rrf_normalized,
    ) in enumerate(scored[:top_k], start=1):
        source = dense_by_id.get(chunk_id) or bm25_by_id[chunk_id]
        fused_results.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                document=source.document,
                chunk_index=source.chunk_index,
                section=source.section,
                page=source.page,
                content=source.content,
                token_count=source.token_count,
                score=final_score,
            )
        )
        dense_chunk = dense_by_id.get(chunk_id)
        bm25_chunk = bm25_by_id.get(chunk_id)
        diagnostics.append(
            HybridCandidateScore(
                rank=rank,
                chunk_id=chunk_id,
                dense_rank=dense_rank.get(chunk_id),
                bm25_rank=bm25_rank.get(chunk_id),
                dense_score=dense_chunk.score if dense_chunk else None,
                bm25_score=bm25_chunk.score if bm25_chunk else None,
                dense_normalized=dense_norm.get(chunk_id, 0.0),
                bm25_normalized=bm25_norm.get(chunk_id, 0.0),
                normalized_weighted_score=normalized_weighted_score,
                rrf_normalized=rrf_normalized,
                final_score=final_score,
            )
        )

    fusion_latency_ms = max(
        0,
        round((time.perf_counter() - started_at) * 1000),
    )
    return HybridRetrievalResult(
        query=dense.query,
        engine="dense_bm25_normalized_weighted_rrf",
        latency_ms=dense.latency_ms + bm25.latency_ms + fusion_latency_ms,
        results=fused_results,
        candidate_scores=tuple(diagnostics),
    )


def fuse_dense_bm25_graph_rrf(
    dense: CosineRetrievalResult,
    bm25: BM25RetrievalResult,
    graph: GraphRetrievalResult,
    *,
    top_k: int,
    dense_weight: float = 0.4,
    bm25_weight: float = 0.3,
    graph_weight: float = 0.3,
    rrf_weight: float = 0.5,
    rrf_k: int = 60,
) -> HybridRetrievalResult:
    """融合 Dense、BM25 和 Graph 三路 Chunk 候选。"""

    if dense.query != bm25.query or dense.query != graph.query:
        raise ValueError("Dense、BM25 与 Graph 的 query 必须一致")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1 到 100 之间")
    if min(dense_weight, bm25_weight, graph_weight) < 0:
        raise ValueError("Dense/BM25/Graph 权重不能小于 0")
    branch_weight_sum = dense_weight + bm25_weight + graph_weight
    if branch_weight_sum <= 0:
        raise ValueError("Dense/BM25/Graph 权重之和必须大于 0")
    if not 0.0 <= rrf_weight <= 1.0:
        raise ValueError("rrf_weight 必须在 0 到 1 之间")
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    if not graph.results:
        if dense_weight + bm25_weight <= 0:
            return HybridRetrievalResult(
                query=dense.query,
                engine="graph_empty",
                latency_ms=dense.latency_ms + bm25.latency_ms + graph.latency_ms,
                results=[],
                candidate_scores=(),
            )
        fallback = fuse_dense_bm25_rrf(
            dense,
            bm25,
            top_k=top_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
            rrf_weight=rrf_weight,
            rrf_k=rrf_k,
        )
        return HybridRetrievalResult(
            query=fallback.query,
            engine=f"{fallback.engine}+graph_fallback",
            latency_ms=fallback.latency_ms + graph.latency_ms,
            results=fallback.results,
            candidate_scores=fallback.candidate_scores,
        )

    started_at = time.perf_counter()
    weights = {
        "dense": dense_weight / branch_weight_sum,
        "bm25": bm25_weight / branch_weight_sum,
        "graph": graph_weight / branch_weight_sum,
    }
    dense_by_id = {chunk.chunk_id: chunk for chunk in dense.results}
    bm25_by_id = {chunk.chunk_id: chunk for chunk in bm25.results}
    graph_by_id = {chunk.chunk_id: chunk for chunk in graph.results}
    dense_norm = _min_max_scores(dense.results)
    bm25_norm = _min_max_scores(bm25.results)
    graph_norm = _min_max_scores(graph.results)
    dense_rank = {chunk.chunk_id: chunk.rank for chunk in dense.results}
    bm25_rank = {chunk.chunk_id: chunk.rank for chunk in bm25.results}
    graph_rank = {chunk.chunk_id: chunk.rank for chunk in graph.results}

    scored: list[tuple[int, float, float, float]] = []
    chunk_ids = dense_by_id.keys() | bm25_by_id.keys() | graph_by_id.keys()
    for chunk_id in chunk_ids:
        normalized_weighted_score = (
            weights["dense"] * dense_norm.get(chunk_id, 0.0)
            + weights["bm25"] * bm25_norm.get(chunk_id, 0.0)
            + weights["graph"] * graph_norm.get(chunk_id, 0.0)
        )
        rrf_score = 0.0
        if chunk_id in dense_rank:
            rrf_score += weights["dense"] / (rrf_k + dense_rank[chunk_id])
        if chunk_id in bm25_rank:
            rrf_score += weights["bm25"] / (rrf_k + bm25_rank[chunk_id])
        if chunk_id in graph_rank:
            rrf_score += weights["graph"] / (rrf_k + graph_rank[chunk_id])
        rrf_normalized = min(1.0, rrf_score * (rrf_k + 1))
        final_score = (
            (1.0 - rrf_weight) * normalized_weighted_score
            + rrf_weight * rrf_normalized
        )
        scored.append(
            (chunk_id, final_score, normalized_weighted_score, rrf_normalized)
        )

    scored.sort(key=lambda row: (-row[1], -row[2], -row[3], row[0]))
    fused_results: list[RetrievedChunk] = []
    diagnostics: list[HybridCandidateScore] = []
    for rank, (
        chunk_id,
        final_score,
        normalized_weighted_score,
        rrf_normalized,
    ) in enumerate(scored[:top_k], start=1):
        source = (
            dense_by_id.get(chunk_id)
            or bm25_by_id.get(chunk_id)
            or graph_by_id[chunk_id]
        )
        fused_results.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                document=source.document,
                chunk_index=source.chunk_index,
                section=source.section,
                page=source.page,
                content=source.content,
                token_count=source.token_count,
                score=final_score,
            )
        )
        dense_chunk = dense_by_id.get(chunk_id)
        bm25_chunk = bm25_by_id.get(chunk_id)
        graph_chunk = graph_by_id.get(chunk_id)
        diagnostics.append(
            HybridCandidateScore(
                rank=rank,
                chunk_id=chunk_id,
                dense_rank=dense_rank.get(chunk_id),
                bm25_rank=bm25_rank.get(chunk_id),
                graph_rank=graph_rank.get(chunk_id),
                dense_score=dense_chunk.score if dense_chunk else None,
                bm25_score=bm25_chunk.score if bm25_chunk else None,
                graph_score=graph_chunk.score if graph_chunk else None,
                dense_normalized=dense_norm.get(chunk_id, 0.0),
                bm25_normalized=bm25_norm.get(chunk_id, 0.0),
                graph_normalized=graph_norm.get(chunk_id, 0.0),
                normalized_weighted_score=normalized_weighted_score,
                rrf_normalized=rrf_normalized,
                final_score=final_score,
            )
        )

    fusion_latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
    return HybridRetrievalResult(
        query=dense.query,
        engine="dense_bm25_graph_normalized_weighted_rrf",
        latency_ms=(
            dense.latency_ms + bm25.latency_ms + graph.latency_ms + fusion_latency_ms
        ),
        results=fused_results,
        candidate_scores=tuple(diagnostics),
    )
