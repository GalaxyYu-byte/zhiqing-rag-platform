"""限时调用模型、逐字校验与保守降级；不访问知识库或文档。"""

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx
from openai import APITimeoutError, AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from zhiqing_rag.core.config import Settings, get_settings

from .models import GroundingError, QueryAnalysisRequest, QueryAnalysisResult, QueryFeatures
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# 只用于模型故障时保护明显的指代/全量需求，不是成功分析的路由规则。
CONTEXT_REFERENCE = re.compile(
    r"它(?:们)?|他(?:们)?|她(?:们)?|这个|那个|这些|那些|这两|那两|上述|前者|后者|刚才|上面|继续|然后呢"
    r"|\b(?:it|they|them|this|that|those|these|former|latter)\b",
    re.IGNORECASE,
)
EXHAUSTIVE_REQUEST = re.compile(
    r"统计|汇总|总额|总数|完整清单|(?:所有|全部)(?:的)?(?:文档|文件|资料)"
    r"|(?:文档|文件|资料).*?(?:多少|几份|几篇)|(?:多少|几份|几篇).*?(?:文档|文件|资料)"
    r"|\b(?:count|total|all documents|all files)\b",
    re.IGNORECASE,
)


class InvalidResponse(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidResponse("JSON 字段重复")
        result[key] = value
    return result


def _safe_token_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


class QueryAnalyzerService:
    """外部注入的客户端由调用方关闭；内部客户端每次调用后自动关闭。"""

    def __init__(self, settings: Settings | None = None, *, client: AsyncOpenAI | None = None):
        self.settings = settings or get_settings()
        self.client = client

    async def analyze(self, request: QueryAnalysisRequest) -> QueryAnalysisResult:
        # 调用方错误不能被包装为模型故障。
        request = QueryAnalysisRequest.model_validate(request.model_dump())
        started = time.perf_counter()

        def result(**values):
            return QueryAnalysisResult(
                original_query=request.query,
                model=self.settings.query_analyzer_model,
                latency_ms=max(0, round((time.perf_counter() - started) * 1000)),
                **values,
            )

        def degraded(reason, error=None):
            if CONTEXT_REFERENCE.search(request.query):
                hint, message = "clarify", "暂时无法可靠理解指代，请补充对象名称和条件。"
            elif EXHAUSTIVE_REQUEST.search(request.query):
                hint, message = (
                    "unsupported",
                    "暂时无法可靠分析统计或完整清单，不能用片段检索代替。",
                )
            else:
                hint, message = "basic_retrieval", "分析暂不可用，建议保留当前原问题执行基础检索。"
            response = result(
                status="degraded",
                analysis=None,
                fallback_hint=hint,
                message=message,
                degradation_reason=reason,
            )
            # 故障类型和状态码足以定位调用失败；不记录异常正文、凭据、问题或模型输出。
            log = logger.info if reason == "disabled" else logger.warning
            log(
                "问题分析降级 reason=%s error_type=%s http_status=%s latency_ms=%s",
                reason,
                type(error).__name__ if error is not None else "none",
                getattr(error, "status_code", None),
                response.latency_ms,
            )
            return response

        if not self.settings.query_analyzer_enabled:
            return degraded("disabled")
        key = self.settings.deepseek_api_key.get_secret_value()
        if self.client is None and not key:
            return degraded("not_configured")
        try:
            async with asyncio.timeout(self.settings.query_analyzer_timeout_seconds):
                if self.client is not None:
                    analysis = await self._analyze(self.client, request)
                else:
                    async with AsyncOpenAI(
                        api_key=key,
                        base_url=self.settings.deepseek_base_url,
                        timeout=self.settings.query_analyzer_timeout_seconds,
                        max_retries=0,
                    ) as client:
                        analysis = await self._analyze(client, request)
        except (TimeoutError, httpx.TimeoutException, APITimeoutError) as error:
            return degraded("timeout", error)
        except GroundingError as error:
            return degraded("invalid_grounding", error)
        except InvalidResponse as error:
            return degraded(
                "incomplete_response" if str(error) == "incomplete" else "invalid_response",
                error,
            )
        except (ValidationError, json.JSONDecodeError, RecursionError) as error:
            return degraded("invalid_response", error)
        except (OpenAIError, httpx.HTTPError, OSError) as error:
            # 不返回/记录异常正文，其中可能包含凭据、模型响应或用户问题。
            return degraded("model_unavailable", error)
        return result(
            status="ok",
            analysis=analysis,
            fallback_hint=None,
            message=None,
            degradation_reason=None,
        )

    async def _analyze(self, client: AsyncOpenAI, request: QueryAnalysisRequest) -> QueryFeatures:
        response = await client.chat.completions.create(
            model=self.settings.query_analyzer_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.model_dump_json()},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=self.settings.query_analyzer_max_output_tokens,
            # DeepSeek 默认开启思考；特征抽取显式关闭，避免思考耗尽 JSON 的输出预算。
            extra_body={"thinking": {"type": "disabled"}},
        )
        if len(response.choices) != 1:
            raise InvalidResponse("必须返回一份分析")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            reason = choice.finish_reason
            safe_reason = reason if reason in (
                "length", "content_filter", "tool_calls", "insufficient_system_resource", "aborted"
            ) else "unknown"
            usage = getattr(response, "usage", None)
            details = getattr(usage, "completion_tokens_details", None)
            logger.warning(
                "分析模型输出未完整结束 finish_reason=%s completion_tokens=%s reasoning_tokens=%s",
                safe_reason,
                _safe_token_count(getattr(usage, "completion_tokens", None)),
                _safe_token_count(getattr(details, "reasoning_tokens", None)),
            )
            raise InvalidResponse("incomplete")
        content = choice.message.content
        if getattr(choice.message, "refusal", None) or not content or len(content) > 64000:
            raise InvalidResponse("模型没有返回有效 JSON")
        analysis = QueryFeatures.model_validate(
            json.loads(content, object_pairs_hook=_unique_object)
        )
        analysis.validate_grounding(request)
        return analysis
