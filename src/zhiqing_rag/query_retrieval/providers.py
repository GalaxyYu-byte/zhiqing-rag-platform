"""查询向量与重排客户端：限时、无重试，严格校验供应商索引。"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from openai import AsyncOpenAI

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing.embedding import (
    EMBEDDING_DIMENSIONS,
    EmbeddingValidationError,
    validate_embedding,
)

logger = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    """供应商未配置或能力关闭；code 可供执行器生成安全的诊断。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class ProviderResponseError(ValueError):
    """供应商响应不满足完整性要求，禁止使用其中的部分结果。"""


def _validate_query(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")


async def _close_owned(close: Callable[[], Awaitable[Any]], timeout: float) -> None:
    # 调用已超时/取消后，清理仍需有独立上限；不吞掉调用方的取消。
    try:
        async with asyncio.timeout(min(timeout, 1.0)):
            await close()
    except Exception as exc:
        logger.warning("检索供应商客户端关闭失败: %s", type(exc).__name__)


class QueryEmbedder:
    """注入客户端由调用方管理，自建客户端仅在当前请求内使用。"""

    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None):
        self.settings = settings
        self.client = client

    async def embed(self, query: str) -> tuple[float, ...]:
        _validate_query(query)
        if self.settings.embedding_dimensions != EMBEDDING_DIMENSIONS:
            raise ProviderUnavailable("EMBEDDING_DIMENSIONS_UNSUPPORTED")
        key = self.settings.dashscope_api_key.get_secret_value()
        if (
            (self.client is None and not key.strip())
            or not self.settings.openai_base_url.strip()
            or not self.settings.embedding_model.strip()
        ):
            raise ProviderUnavailable("EMBEDDING_NOT_CONFIGURED")
        timeout = self.settings.query_embedding_timeout_seconds
        async with asyncio.timeout(timeout):
            client = self.client
            owned = client is None
            if client is None:
                client = AsyncOpenAI(
                    api_key=key,
                    base_url=self.settings.openai_base_url,
                    timeout=timeout,
                    max_retries=0,
                )
            try:
                response = await client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=[query],
                    dimensions=EMBEDDING_DIMENSIONS,
                    encoding_format="float",
                )
                data = getattr(response, "data", None)
                if not isinstance(data, (list, tuple)) or len(data) != 1:
                    raise ProviderResponseError("查询向量返回数量不匹配")
                item = data[0]
                index = getattr(item, "index", None)
                if type(index) is not int or index != 0:
                    raise ProviderResponseError("查询向量响应索引无效")
                vector = getattr(item, "embedding", None)
                if not isinstance(vector, (list, tuple)):
                    raise ProviderResponseError("查询向量格式无效")
                try:
                    validate_embedding(vector)
                except (EmbeddingValidationError, OverflowError):
                    raise ProviderResponseError("查询向量校验失败") from None
                return tuple(float(value) for value in vector)
            finally:
                if owned:
                    await _close_owned(client.close, timeout)


class QueryReranker:
    """只采纳候选索引与分数，供应商返回的文档正文不进入结果。"""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.client = client

    async def rerank(self, query: str, documents: tuple[str, ...]) -> tuple[tuple[int, float], ...]:
        _validate_query(query)
        if not isinstance(documents, tuple) or any(
            not isinstance(document, str) or not document.strip() for document in documents
        ):
            raise ValueError("documents 必须是非空原文字符串组成的 tuple")
        if not documents:
            return ()
        if not self.settings.query_reranker_enabled:
            raise ProviderUnavailable("RERANKER_DISABLED")
        key = self.settings.dashscope_api_key.get_secret_value()
        if (
            (self.client is None and not key.strip())
            or not self.settings.reranker_endpoint.strip()
            or not self.settings.reranker_model.strip()
        ):
            raise ProviderUnavailable("RERANKER_NOT_CONFIGURED")
        payload = {
            "model": self.settings.reranker_model,
            "input": {"query": query, "documents": list(documents)},
            "parameters": {"return_documents": False, "top_n": len(documents)},
        }
        headers = {"Content-Type": "application/json"}
        if key.strip():
            headers["Authorization"] = f"Bearer {key}"
        timeout = self.settings.query_reranker_timeout_seconds
        async with asyncio.timeout(timeout):
            client = self.client
            owned = client is None
            if client is None:
                # httpx 默认不重试连接，且不跟随重定向转发候选正文。
                client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)
            try:
                response = await client.post(
                    self.settings.reranker_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                return _parse_rerank(response.json(), len(documents))
            finally:
                if owned:
                    await _close_owned(client.aclose, timeout)


def _parse_rerank(body: Any, count: int) -> tuple[tuple[int, float], ...]:
    if not isinstance(body, dict) or body.get("code"):
        raise ProviderResponseError("重排供应商返回失败响应")
    output = body.get("output")
    results = output.get("results") if isinstance(output, dict) else None
    if not isinstance(results, list) or len(results) != count:
        raise ProviderResponseError("重排候选返回数量不匹配")
    parsed = []
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            raise ProviderResponseError("重排候选格式无效")
        index = result.get("index")
        score = result.get("relevance_score")
        if type(index) is not int or not 0 <= index < count or index in seen:
            raise ProviderResponseError("重排候选索引重复或无效")
        if type(score) not in (int, float):
            raise ProviderResponseError("重排候选分数无效")
        try:
            score = float(score)
        except OverflowError:
            raise ProviderResponseError("重排候选分数无效") from None
        if not math.isfinite(score):
            raise ProviderResponseError("重排候选分数必须有限")
        parsed.append((index, score))
        seen.add(index)
    return tuple(sorted(parsed, key=lambda item: (-item[1], item[0])))
