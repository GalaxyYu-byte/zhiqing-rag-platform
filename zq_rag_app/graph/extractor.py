"""基于 Qwen JSON Object 输出的 Pydantic 图谱抽取器。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from ..core.config import settings
from .models import GraphChunkContext, GraphExtraction


logger = logging.getLogger(__name__)


class GraphExtractor(Protocol):
    async def extract(self, context: GraphChunkContext) -> GraphExtraction: ...


class QwenGraphExtractor:
    """调用 OpenAI 兼容接口，并对返回 JSON 做本地强校验。"""

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = settings.graph_extraction_model,
        temperature: float = settings.graph_extraction_temperature,
        max_tokens: int = settings.graph_extraction_max_tokens,
        retry_attempts: int = settings.graph_extraction_retry_attempts,
    ) -> None:
        if max_tokens <= 0 or retry_attempts <= 0:
            raise ValueError("max_tokens 和 retry_attempts 必须大于 0")
        self.client = client or AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.graph_extraction_timeout_seconds,
            max_retries=0,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_attempts = retry_attempts

    async def extract(self, context: GraphChunkContext) -> GraphExtraction:
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=self._build_messages(context, last_error),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    # 结构化抽取不使用思考模式，避免 JSON 输出约束失效。
                    extra_body={"enable_thinking": False},
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("图抽取模型返回了空内容")
                payload = _normalize_model_payload(content)
                extraction = GraphExtraction.model_validate(payload)
                removed_entities, removed_relations = (
                    extraction.discard_unsupported_evidence(context.content)
                )
                if removed_entities or removed_relations:
                    logger.warning(
                        "丢弃无原文证据的图候选: entities=%s relations=%s",
                        removed_entities,
                        removed_relations,
                    )
                extraction.validate_evidence(context.content)
                return extraction
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_attempts or not _is_retryable(exc):
                    raise
                delay = min(4.0, 0.5 * (2 ** (attempt - 1)))
                logger.warning(
                    "图抽取第 %s 次失败，将在 %.1f 秒后重试: %s",
                    attempt,
                    delay,
                    type(exc).__name__,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("图抽取重试循环异常结束") from last_error

    @staticmethod
    def _build_messages(
        context: GraphChunkContext,
        previous_error: Exception | None,
    ) -> list[dict[str, str]]:
        schema = json.dumps(
            GraphExtraction.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        correction = ""
        if previous_error is not None:
            correction = (
                "\n上一次输出未通过校验，请修正。错误摘要："
                f"{str(previous_error)[:800]}"
            )
        system = f"""你是企业知识图谱信息抽取器。只输出一个 JSON 对象，不要输出 Markdown。
输出必须符合下面的 JSON Schema：
{schema}

规则：
1. 只能抽取“原文开始”和“原文结束”之间明确表达的实体和关系，不得补充常识。
   文件名、章节和页码只是定位元数据，除非它们也逐字出现在原文中，否则禁止抽取。
2. local_id 按 e1、e2 顺序编号；关系只能引用 entities 中的 local_id。
3. evidence_quote 必须逐字来自输入原文，可以忽略原文中的连续空白差异。
4. 同一个现实实体在当前 Chunk 内只创建一次，其他名称写入 aliases。
5. 不确定是否为同一实体时保持分开；无法确认的关系不要输出。
6. predicate 只能使用 Schema 中的枚举值。表达“ A 拥有 B”时必须输出
   source=B、predicate=OWNED_BY、target=A，禁止输出 OWNS。
7. 没有可抽取内容时返回 schema_version 为 1.0 的空数组。{correction}"""
        user = (
            "请从以下文档 Chunk 中抽取知识图谱 JSON。\n"
            f"文件：{context.file_name}\n"
            f"章节：{context.section_title or '未知'}\n"
            f"页码：{context.page_num or '未知'}\n"
            "原文开始：\n"
            f"{context.content}\n"
            "原文结束。"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def _is_retryable(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            ValidationError,
            ValueError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def _normalize_model_payload(content: str) -> dict[str, Any]:
    """修复可无歧义转换的模型输出，再交给严格协议校验。

    这里仅做安全的机械转换：反向谓词交换端点；文档实体和未知类型候选
    被丢弃；重复关系去重。其他未知值仍由 Pydantic 拒绝。
    """

    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("图抽取模型必须返回 JSON 对象")

    entities = payload.get("entities")
    supported_entity_types = {
        "PERSON", "ORGANIZATION", "DEPARTMENT", "SYSTEM", "SERVICE",
        "PRODUCT", "TECHNOLOGY", "PROCESS", "POLICY", "EVENT", "LOCATION",
        "CONCEPT",
    }
    valid_entities: list[dict[str, Any]] = []
    valid_local_ids: set[str] = set()
    if isinstance(entities, list):
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            entity_type = entity.get("entity_type")
            if not isinstance(entity_type, str):
                continue
            entity_type = entity_type.strip().upper().replace("-", "_").replace(" ", "_")
            if entity_type not in supported_entity_types:
                continue
            local_id = entity.get("local_id")
            if not isinstance(local_id, str) or local_id in valid_local_ids:
                continue
            entity["entity_type"] = entity_type
            valid_entities.append(entity)
            valid_local_ids.add(local_id)
        payload["entities"] = valid_entities

    relations = payload.get("relations")
    if not isinstance(relations, list):
        return payload

    supported_predicates = {
        "PART_OF", "OWNED_BY", "DEPENDS_ON", "USES", "IMPACTS",
        "CAUSED_BY", "RESOLVED_BY", "LOCATED_IN", "RELATED_TO",
    }
    normalized_relations: list[dict[str, Any]] = []
    relation_keys: set[tuple[Any, ...]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        if (
            relation.get("source_local_id") not in valid_local_ids
            or relation.get("target_local_id") not in valid_local_ids
            or relation.get("source_local_id") == relation.get("target_local_id")
        ):
            continue
        predicate = relation.get("predicate")
        if not isinstance(predicate, str):
            continue

        normalized = predicate.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized in {"OWNS", "OWNER_OF", "USED_BY"}:
            source = relation.get("source_local_id")
            target = relation.get("target_local_id")
            relation["source_local_id"] = target
            relation["target_local_id"] = source
            normalized = "OWNED_BY" if normalized in {"OWNS", "OWNER_OF"} else "USES"
        elif normalized in {"BELONGS_TO", "CONTAINS"}:
            if normalized == "CONTAINS":
                source = relation.get("source_local_id")
                relation["source_local_id"] = relation.get("target_local_id")
                relation["target_local_id"] = source
            normalized = "PART_OF"
        elif normalized == "APPLIES_TO":
            normalized = "RELATED_TO"
        if normalized not in supported_predicates:
            continue
        relation["predicate"] = normalized
        key = (
            relation.get("source_local_id"),
            relation.get("predicate"),
            relation.get("target_local_id"),
            relation.get("polarity", "POSITIVE"),
            relation.get("valid_from"),
            relation.get("valid_to"),
        )
        if key in relation_keys:
            continue
        relation_keys.add(key)
        normalized_relations.append(relation)

    payload["relations"] = normalized_relations

    return payload
