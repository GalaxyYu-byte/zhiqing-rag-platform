"""DashScope 文本 Reranker 客户端。"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from ..core.config import settings
from .hybrid_retrieval_service import HybridRetrievalResult
from .retrieval_service import RetrievedChunk


@dataclass(slots=True, frozen=True)
class RerankedCandidateScore:
    rank: int
    chunk_id: int
    fusion_rank: int
    fusion_score: float
    reranker_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float | None
    bm25_score: float | None
    dense_normalized: float
    bm25_normalized: float
    normalized_weighted_score: float
    rrf_normalized: float
    graph_rank: int | None = None
    graph_score: float | None = None
    graph_normalized: float = 0.0


@dataclass(slots=True, frozen=True)
class RerankerResult:
    query: str
    engine: str
    model: str
    latency_ms: int
    reranker_latency_ms: int
    total_tokens: int | None
    request_id: str | None
    results: list[RetrievedChunk]
    candidate_scores: tuple[RerankedCandidateScore, ...]


class RerankerService:
    def __init__(
        self,
        *,
        endpoint: str = settings.reranker_endpoint,
        api_key: str = settings.dashscope_api_key,
        model: str = settings.reranker_model,
        timeout_ms: int = settings.reranker_timeout_ms,
        retry_attempts: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not endpoint.strip() or not api_key.strip() or not model.strip():
            raise ValueError("Reranker endpoint/api_key/model 不能为空")
        if timeout_ms <= 0 or retry_attempts <= 0:
            raise ValueError("Reranker timeout/retry_attempts 必须大于 0")
        self.endpoint = endpoint.strip()
        self.model = model.strip()
        self.timeout_ms = timeout_ms
        self.retry_attempts = retry_attempts
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_ms / 1000),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _request(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: BaseException | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = await self.client.post(self.endpoint, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt == self.retry_attempts:
                    raise
                await asyncio.sleep(0.25 * attempt)
        assert last_error is not None
        raise last_error

    async def rerank(
        self,
        hybrid: HybridRetrievalResult,
        *,
        top_n: int,
    ) -> RerankerResult:
        if not hybrid.query.strip():
            raise ValueError("query 不能为空")
        if not hybrid.results:
            raise ValueError("精排候选不能为空")
        if not 1 <= top_n <= len(hybrid.results):
            raise ValueError("top_n 必须在候选数量范围内")

        payload = {
            "model": self.model,
            "input": {
                "query": hybrid.query,
                "documents": [chunk.content for chunk in hybrid.results],
            },
            "parameters": {
                "return_documents": False,
                "top_n": top_n,
            },
        }
        started_at = time.perf_counter()
        response = await self._request(payload)
        reranker_latency_ms = max(
            0,
            round((time.perf_counter() - started_at) * 1000),
        )
        response.raise_for_status()
        body = response.json()
        if body.get("code"):
            raise RuntimeError(
                f"Reranker API 返回错误 {body['code']}: "
                f"{body.get('message', '')}"
            )
        raw_results = body.get("output", {}).get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError("Reranker 响应缺少 output.results")

        parsed: list[tuple[int, float]] = []
        seen_indexes: set[int] = set()
        for item in raw_results:
            index = int(item["index"])
            score = float(item["relevance_score"])
            if not 0 <= index < len(hybrid.results):
                raise RuntimeError("Reranker 返回了越界的文档索引")
            if index in seen_indexes:
                raise RuntimeError("Reranker 返回了重复的文档索引")
            if not math.isfinite(score):
                raise RuntimeError("Reranker 返回了非有限分数")
            seen_indexes.add(index)
            parsed.append((index, score))
        if len(parsed) != top_n:
            raise RuntimeError(
                f"Reranker 返回 {len(parsed)} 条结果，预期 {top_n} 条"
            )
        parsed.sort(key=lambda value: (-value[1], value[0]))

        upstream_diagnostics = {
            candidate.chunk_id: asdict(candidate)
            for candidate in hybrid.candidate_scores
        }
        results: list[RetrievedChunk] = []
        diagnostics: list[RerankedCandidateScore] = []
        for rank, (index, reranker_score) in enumerate(parsed, start=1):
            source = hybrid.results[index]
            upstream = upstream_diagnostics[source.chunk_id]
            results.append(
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
                    score=reranker_score,
                )
            )
            diagnostics.append(
                RerankedCandidateScore(
                    rank=rank,
                    chunk_id=source.chunk_id,
                    fusion_rank=source.rank,
                    fusion_score=source.score,
                    reranker_score=reranker_score,
                    dense_rank=upstream["dense_rank"],
                    bm25_rank=upstream["bm25_rank"],
                    dense_score=upstream["dense_score"],
                    bm25_score=upstream["bm25_score"],
                    dense_normalized=upstream["dense_normalized"],
                    bm25_normalized=upstream["bm25_normalized"],
                    graph_rank=upstream.get("graph_rank"),
                    graph_score=upstream.get("graph_score"),
                    graph_normalized=upstream.get("graph_normalized", 0.0),
                    normalized_weighted_score=upstream[
                        "normalized_weighted_score"
                    ],
                    rrf_normalized=upstream["rrf_normalized"],
                )
            )

        usage = body.get("usage") or {}
        total_tokens = usage.get("total_tokens")
        return RerankerResult(
            query=hybrid.query,
            engine=f"{hybrid.engine}+{self.model}",
            model=self.model,
            latency_ms=hybrid.latency_ms + reranker_latency_ms,
            reranker_latency_ms=reranker_latency_ms,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            request_id=body.get("request_id"),
            results=results,
            candidate_scores=tuple(diagnostics),
        )
