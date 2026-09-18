"""DeepSeek 在线问题分析：JSON 模式、严格校验、有限重试与整体超时降级。"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from openai import (
    APIConnectionError, APIResponseValidationError, APIStatusError,
    APITimeoutError, AsyncOpenAI,
)
from pydantic import ValidationError

from ..core.config import settings
from ..core.query_metrics import ANALYZER_CALLS, ANALYZER_DURATION, ANALYZER_REQUESTS, ANALYZER_RETRIES
from ..query_analysis.models import (
    QueryAnalysis, QueryAnalysisRequest, QueryAnalysisResult,
)
from ..query_analysis.prompts import build_messages


logger = logging.getLogger(__name__)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 包含重复字段")
        result[key] = value
    return result


def _parse_response(response: Any, request: QueryAnalysisRequest) -> QueryAnalysis:
    if not response.choices:
        raise ValueError("模型未返回候选")
    choice = response.choices[0]
    if choice.finish_reason != "stop":
        raise ValueError("模型输出不完整或被拦截")
    content = choice.message.content
    if not isinstance(content, str) or not content.strip() or len(content) > 40000:
        raise ValueError("模型内容为空或超出大小限制")
    payload = json.loads(content, object_pairs_hook=_unique_object)
    analysis = QueryAnalysis.model_validate(payload)
    analysis.validate_grounding(request)
    # 同类同名抽取项去重，保持模型给出的顺序。
    for field, key_fields in (
        ("entities", ("entity_type", "name", "graph_role", "qualifiers")),
        ("keywords", ("kind", "text")),
        ("metadata", ("field", "operator", "value")),
    ):
        seen: set[tuple[str, ...]] = set()
        unique = []
        for candidate in getattr(analysis, field):
            key = tuple(tuple(value) if isinstance(value, list) else value
                        for value in (getattr(candidate, key_field) for key_field in key_fields))
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        setattr(analysis, field, unique)
    return analysis


class QueryAnalyzerService:
    """不读取知识库，不执行检索。注入的客户端由调用方管理生命周期。"""

    def __init__(
        self, *, client: Any | None = None, api_key: str | None = None,
        base_url: str | None = None, model: str | None = None,
        timeout_seconds: float | None = None, retry_attempts: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = (model if model is not None else settings.query_analyzer_model).strip()
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None
            else settings.query_analyzer_timeout_seconds
        )
        self.retry_attempts = (
            retry_attempts if retry_attempts is not None
            else settings.query_analyzer_retry_attempts
        )
        self.max_tokens = max_tokens if max_tokens is not None else settings.query_analyzer_max_tokens
        if not self.model or not 0 < self.timeout_seconds <= 120:
            raise ValueError("model 不能为空，整体超时必须在 (0, 120] 秒之间")
        if not 1 <= self.retry_attempts <= 3 or not 512 <= self.max_tokens <= 8192:
            raise ValueError("retry_attempts 必须在 1..3，max_tokens 必须在 512..8192")
        key = settings.deepseek_api_key.get_secret_value() if api_key is None else api_key
        self.client = client
        self._owns_client = client is None
        if self.client is None and key.strip():
            self.client = AsyncOpenAI(
                api_key=key.strip(), base_url=base_url or settings.deepseek_base_url,
                timeout=self.timeout_seconds, max_retries=0,
            )

    async def aclose(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.close()

    async def analyze(
        self, request: QueryAnalysisRequest,
    ) -> QueryAnalysisResult:
        started = time.perf_counter()
        status, reason = "error", "unexpected_error"
        try:
            result = await self._analyze(request)
            status, reason = result.status, result.degradation_reason or "none"
            return result
        except asyncio.CancelledError:
            status, reason = "cancelled", "cancelled"
            raise
        finally:
            ANALYZER_REQUESTS.labels(status, reason).inc()
            ANALYZER_DURATION.observe(time.perf_counter() - started)

    async def _analyze(self, request: QueryAnalysisRequest) -> QueryAnalysisResult:
        started = time.perf_counter()
        reference_date = request.reference_date or datetime.now(
            timezone(timedelta(hours=8))
        ).date()
        if self.client is None:
            return self._fallback(request, reference_date, started, 0, "missing_api_key")
        attempts = 0
        failure = "invalid_output"
        correction = False
        try:
            # 覆盖全部尝试和退避，避免多次 SDK 超时叠加拖慢在线链路。
            async with asyncio.timeout(self.timeout_seconds):
                for attempts in range(1, self.retry_attempts + 1):
                    try:
                        ANALYZER_CALLS.inc()
                        if attempts > 1:
                            ANALYZER_RETRIES.labels(failure).inc()
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=build_messages(request, reference_date, correction=correction),
                            temperature=0.0, max_tokens=self.max_tokens,
                            response_format={"type": "json_object"},
                            extra_body={"thinking": {"type": "disabled"}},
                        )
                        analysis = _parse_response(response, request)
                        return QueryAnalysisResult(
                            **analysis.model_dump(),
                            model_suggestion=analysis.retrieval_strategy.model_copy(deep=True),
                            original_query=request.query,
                            reference_date=reference_date.isoformat(), model=self.model,
                            status="ok", degradation_reason=None, attempts=attempts,
                            latency_ms=round((time.perf_counter() - started) * 1000),
                            warnings=[],
                        )
                    except (
                        ValueError, ValidationError, APIResponseValidationError,
                        AttributeError, IndexError, TypeError,
                    ):
                        failure = "invalid_output"
                        correction = True
                    except (APITimeoutError, TimeoutError):
                        failure = "timeout"
                    except APIConnectionError:
                        failure = "provider_error"
                    except APIStatusError as exc:
                        failure = "provider_error"
                        # 401/403/余额不足/无效模型等配置问题不重复请求。
                        if exc.status_code not in {408, 409, 429} and exc.status_code < 500:
                            break
                    # 不记录原始问题、模型输出、密钥或异常响应体。
                    logger.warning("Query Analyzer 尝试失败: attempt=%s category=%s", attempts, failure)
                    if attempts < self.retry_attempts:
                        await asyncio.sleep(0.2 * attempts)
        except TimeoutError:
            failure = "timeout"
        return self._fallback(request, reference_date, started, attempts, failure)

    def _fallback(
        self, request: QueryAnalysisRequest, reference_date: date,
        started: float, attempts: int, reason: str,
    ) -> QueryAnalysisResult:
        # 未成功识别时不伪造实体和过滤条件，也不自动请求图谱。
        analysis = QueryAnalysis.model_validate({
            "intent": {"primary": "other", "secondary": [], "confidence": 0.0},
            "query_type": {
                "primary": "semantic", "confidence": 0.0,
                "has_context_reference": False, "relation_required": False,
                "potential_multi_hop": False, "requires_exhaustive": False,
                "needs_clarification": False, "clarification_question": None,
            },
            "entities": [], "keywords": [], "metadata": [],
            "retrieval_strategy": {
                "path": "hybrid", "use_query_rewrite": False,
                "use_multi_query": False, "use_hyde": False,
                "rewrite_method": "none", "rewrite_operations": [],
                "reason": "分析不可用，保留原问题进行基础混合检索。",
            },
        })
        warnings = ["问题分析未成功；抽取为空不代表问题没有实体或约束。"]
        if request.history:
            warnings.append("降级结果未解析会话历史；独立问题可保留原文检索，明显指代或追问需先补充完整问题。")
        return QueryAnalysisResult(
            **analysis.model_dump(), original_query=request.query,
            reference_date=reference_date.isoformat(), model=self.model,
            status="degraded", degradation_reason=reason, attempts=attempts,
            latency_ms=round((time.perf_counter() - started) * 1000), warnings=warnings,
        )
