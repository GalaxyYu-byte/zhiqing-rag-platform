import json

import httpx
import pytest

from zq_rag_app.services.hybrid_retrieval_service import (
    HybridCandidateScore,
    HybridRetrievalResult,
)
from zq_rag_app.services.reranker_service import RerankerService
from zq_rag_app.services.retrieval_service import RetrievedChunk


def _hybrid_result() -> HybridRetrievalResult:
    chunks = [
        RetrievedChunk(
            rank=index,
            chunk_id=index,
            doc_id=index,
            document=f"{index}.pdf",
            chunk_index=0,
            section=None,
            page=None,
            content=f"候选 {index}",
            token_count=3,
            score=1.0 - index / 10,
        )
        for index in (1, 2, 3)
    ]
    diagnostics = tuple(
        HybridCandidateScore(
            rank=chunk.rank,
            chunk_id=chunk.chunk_id,
            dense_rank=chunk.rank,
            bm25_rank=chunk.rank,
            dense_score=chunk.score,
            bm25_score=chunk.score * 10,
            dense_normalized=chunk.score,
            bm25_normalized=chunk.score,
            normalized_weighted_score=chunk.score,
            rrf_normalized=chunk.score,
            final_score=chunk.score,
        )
        for chunk in chunks
    )
    return HybridRetrievalResult(
        query="问题",
        engine="hybrid",
        latency_ms=30,
        results=chunks,
        candidate_scores=diagnostics,
    )


@pytest.mark.asyncio
async def test_reranker_maps_response_indexes_back_to_chunks():
    captured = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 2, "relevance_score": 0.95},
                        {"index": 0, "relevance_score": 0.8},
                    ]
                },
                "usage": {"total_tokens": 42},
                "request_id": "req-1",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = RerankerService(
        endpoint="https://example.test/rerank",
        api_key="secret",
        model="gte-rerank-v2",
        timeout_ms=1000,
        client=client,
    )
    try:
        result = await service.rerank(_hybrid_result(), top_n=2)
    finally:
        await client.aclose()

    assert captured["input"]["query"] == "问题"
    assert captured["input"]["documents"] == ["候选 1", "候选 2", "候选 3"]
    assert captured["parameters"] == {
        "return_documents": False,
        "top_n": 2,
    }
    assert [chunk.chunk_id for chunk in result.results] == [3, 1]
    assert result.reranker_latency_ms >= 0
    assert result.latency_ms >= 30
    assert result.total_tokens == 42
    assert result.candidate_scores[0].fusion_rank == 3


@pytest.mark.asyncio
async def test_reranker_rejects_incomplete_response():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"output": {"results": [{"index": 0, "relevance_score": 0.8}]}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = RerankerService(
        endpoint="https://example.test/rerank",
        api_key="secret",
        client=client,
    )
    try:
        with pytest.raises(RuntimeError, match="预期 2"):
            await service.rerank(_hybrid_result(), top_n=2)
    finally:
        await client.aclose()
