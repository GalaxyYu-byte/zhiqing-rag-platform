"""DeepSeek 图谱抽取、严格验证和稳定实体标识。"""

import hashlib
import json
import logging
import re
import unicodedata
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from zhiqing_rag.core.config import Settings

EXTRACTOR_VERSION = "graph-extractor-v2"

EntityType = Literal["Actor", "Process", "Step", "Material", "Resource", "Rule", "State", "Field"]
RelationType = Literal[
    "HAS_STEP",
    "NEXT_STEP",
    "RESPONSIBLE_FOR",
    "REVIEWS",
    "CONFIRMS",
    "APPLIES_TO",
    "REQUIRES_MATERIAL",
    "PRODUCES_MATERIAL",
    "OPERATES_ON",
    "HAS_FIELD",
    "HAS_STATE",
    "GOVERNED_BY",
    "DEPENDS_ON",
    "ESCALATES_TO",
    "TRANSITIONS_TO",
]

ENDPOINTS: dict[str, tuple[set[str], set[str]]] = {
    "HAS_STEP": ({"Process"}, {"Step"}),
    "NEXT_STEP": ({"Step"}, {"Step"}),
    "RESPONSIBLE_FOR": ({"Actor"}, {"Process", "Step"}),
    "REVIEWS": ({"Actor"}, {"Step"}),
    "CONFIRMS": ({"Actor"}, {"Step"}),
    "APPLIES_TO": ({"Process", "Rule"}, {"Actor"}),
    "REQUIRES_MATERIAL": ({"Process", "Step"}, {"Material"}),
    "PRODUCES_MATERIAL": ({"Step"}, {"Material"}),
    "OPERATES_ON": ({"Step"}, {"Resource", "Material"}),
    "HAS_FIELD": ({"Material"}, {"Field"}),
    "HAS_STATE": ({"Material", "Resource"}, {"State"}),
    "GOVERNED_BY": ({"Process", "Step", "Material", "Resource", "State", "Field"}, {"Rule"}),
    "DEPENDS_ON": ({"Step"}, {"Step", "Resource"}),
    "ESCALATES_TO": ({"Process", "Step"}, {"Actor"}),
    "TRANSITIONS_TO": ({"State"}, {"State"}),
}

logger = logging.getLogger(__name__)


class GraphExtractionError(ValueError):
    """不含文档内容的安全错误码，供任务状态与日志使用。"""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(code)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=40)
    type: EntityType
    name: str = Field(min_length=1, max_length=150)
    canonical_name: str = Field(min_length=1, max_length=150)
    evidence_quote: str = Field(min_length=1, max_length=500)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    type: RelationType
    source_id: str
    target_id: str
    evidence_quote: str = Field(min_length=1, max_length=500)
    condition_text: str | None = Field(default=None, max_length=500)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[ExtractedEntity] = Field(max_length=80)
    relations: list[ExtractedRelation] = Field(max_length=120)

    @model_validator(mode="after")
    def check_endpoints(self):
        by_id = {entity.id: entity for entity in self.entities}
        if len(by_id) != len(self.entities):
            raise ValueError("实体局部 ID 重复")
        for relation in self.relations:
            source = by_id.get(relation.source_id)
            target = by_id.get(relation.target_id)
            allowed_source, allowed_target = ENDPOINTS[relation.type]
            if (
                source is None
                or target is None
                or source.type not in allowed_source
                or target.type not in allowed_target
            ):
                raise ValueError("关系端点不存在或类型不在白名单")
        return self


def normalize_name(value: str) -> str:
    """仅合并确定性的 Unicode/空格/大小写差异，不猜测语义同义词。"""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip(" \t\r\n，,。.")


def stable_key(*parts: object) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def validate_evidence(extraction: Extraction, content: str) -> None:
    for record in [*extraction.entities, *extraction.relations]:
        if record.evidence_quote not in content:
            raise ValueError("证据原文未出现在当前分块")
    for relation in extraction.relations:
        if relation.condition_text and relation.condition_text not in content:
            raise ValueError("关系条件未出现在当前分块")
    for entity in extraction.entities:
        if entity.name not in content:
            raise ValueError("实体名称未出现在当前分块")
        if entity.canonical_name not in content and normalize_name(
            entity.canonical_name
        ) != normalize_name(entity.name):
            raise ValueError("规范实体名称缺少原文依据")


def _entity_name_feedback(extraction: Extraction, content: str) -> str:
    """向模型指出名称错误；原文仅进入重试消息，不写入任务错误详情。"""
    details = []
    for index, entity in enumerate(extraction.entities):
        invalid_fields = []
        if entity.name not in content:
            invalid_fields.append(("name", entity.name))
        if entity.canonical_name not in content and normalize_name(
            entity.canonical_name
        ) != normalize_name(entity.name):
            invalid_fields.append(("canonical_name", entity.canonical_name))
        if not invalid_fields:
            continue
        fields = "；".join(
            f"entities[{index}].{field} = {json.dumps(value, ensure_ascii=False)} "
            "不是当前分块中的连续原文"
            for field, value in invalid_fields
        )
        details.append(
            fields + "。该实体的逐字证据："
            + json.dumps(entity.evidence_quote, ensure_ascii=False)
        )
        if len(details) == 3:
            break
    return (
        "\n名称错误详情（最多列出 3 个实体）：\n" + "\n".join(details)
        + "\n禁止自行补上说明、核对、审核等词，也不能拼接列头与单元格来构造名称。"
        "请依据原文重新判断名称和类型；不要机械删除前缀后沿用原类型。"
        "其余实体也须逐一检查。canonical_name 仅允许现有的形式归一化。"
    )


def validated_endpoint_subset(output: str, content: str) -> Extraction:
    """模型误配关系端点时，仅保留符合白名单且有逐字证据的事实。"""
    raw = json.loads(output)
    if not isinstance(raw, dict) or set(raw) != {"entities", "relations"}:
        raise ValueError("图谱顶层结构无效")
    if not isinstance(raw["entities"], list) or not isinstance(raw["relations"], list):
        raise ValueError("图谱列表结构无效")
    entities = []
    seen_ids = set()
    duplicate_ids = set()
    for item in raw["entities"]:
        try:
            entity = ExtractedEntity.model_validate(item)
            validate_evidence(Extraction(entities=[entity], relations=[]), content)
        except (ValueError, TypeError):
            continue
        if entity.id in seen_ids:
            duplicate_ids.add(entity.id)
        seen_ids.add(entity.id)
        entities.append(entity)
    entities = [entity for entity in entities if entity.id not in duplicate_ids]
    if raw["entities"] and not entities:
        raise ValueError("没有通过证据校验的实体")
    by_id = {entity.id: entity for entity in entities}
    valid_relations = []
    for item in raw["relations"]:
        try:
            relation = ExtractedRelation.model_validate(item)
        except (ValueError, TypeError):
            continue
        source = by_id.get(relation.source_id)
        target = by_id.get(relation.target_id)
        allowed_source, allowed_target = ENDPOINTS[relation.type]
        if (
            source is not None
            and target is not None
            and source.type in allowed_source
            and target.type in allowed_target
            and relation.evidence_quote in content
            and (not relation.condition_text or relation.condition_text in content)
        ):
            valid_relations.append(relation)
    if len(entities) == len(raw["entities"]) and len(valid_relations) == len(raw["relations"]):
        raise ValueError("无可修复的结构或证据错误")
    result = Extraction.model_validate({"entities": entities, "relations": valid_relations})
    validate_evidence(result, content)
    logger.info(
        "模型输出含无效事实，丢弃 %s 个实体和 %s 条关系",
        len(raw["entities"]) - len(entities),
        len(raw["relations"]) - len(valid_relations),
    )
    return result


SYSTEM_PROMPT = """你从企业制度/流程文档中抽取概念图谱。仅输出 JSON 对象，格式为
{"entities":[{"id":"e1","type":"Actor","name":"财务","canonical_name":"财务","evidence_quote":"财务审核报销申请"},
{"id":"e2","type":"Step","name":"审核报销申请","canonical_name":"审核报销申请","evidence_quote":"财务审核报销申请"}],
"relations":[{"type":"REVIEWS","source_id":"e1","target_id":"e2","evidence_quote":"财务审核报销申请","condition_text":null}]}。
实体类型仅限 Actor, Process, Step, Material, Resource, Rule, State, Field。
关系及端点类型必须严格遵守：
HAS_STEP Process->Step; NEXT_STEP Step->Step; RESPONSIBLE_FOR Actor->Process/Step;
REVIEWS Actor->Step; CONFIRMS Actor->Step; APPLIES_TO Process/Rule->Actor;
REQUIRES_MATERIAL Process/Step->Material; PRODUCES_MATERIAL Step->Material;
OPERATES_ON Step->Resource/Material; HAS_FIELD Material->Field;
HAS_STATE Material/Resource->State;
GOVERNED_BY Process/Step/Material/Resource/State/Field->Rule;
DEPENDS_ON Step->Step/Resource; ESCALATES_TO Process/Step->Actor;
TRANSITIONS_TO State->State。
每条关系的 source_id 和 target_id 必须引用 entities 中的实际 id，禁止使用未列出的 id。
只抽原文明确支持的事实；不要推断执行结果。name 必须是输入正文中的连续原文；
canonical_name 也应取自原文或仅做空格、大小写等形式归一化。
禁止为了组成动作名称自行补上“说明”“核对”“审核”等词，或拼接列头与单元格内容。
名称与类型都应依据原文判断，不能为符合预设类型而改写名称。
每条实体及关系的 evidence_quote 必须是输入正文中的连续逐字片段。
condition_text 非空时也必须是正文中的连续逐字片段。条件、例外、否定不能丢失。
每个 Chunk 最多提取 25 个有明确证据的实体和 40 条关系。
不能准确抽取时返回 {"entities":[],"relations":[]}。"""


async def extract_chunk(
    settings: Settings, content: str, *, client: AsyncOpenAI | None = None
) -> Extraction:
    if not settings.deepseek_api_key.get_secret_value():
        raise ValueError("DEEPSEEK_API_KEY 未配置")
    owns_client = client is None
    if client is None:
        client = AsyncOpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url=settings.deepseek_base_url,
            timeout=settings.graph_llm_timeout_seconds,
            max_retries=0,
        )
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        last_code = "OUTPUT_EMPTY"
        last_detail = ""
        rejected_candidate = False
        for attempt in range(3):
            response = await client.chat.completions.create(
                model=settings.graph_extraction_model,
                temperature=0,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=4096,
                messages=messages,
            )
            choice = response.choices[0]
            output = choice.message.content or ""
            if choice.finish_reason == "length":
                rejected_candidate = True
                last_code = "OUTPUT_TRUNCATED"
                last_detail = ""
                feedback = "上次 JSON 被截断。请减少实体和关系，仅保留证据最明确的内容。"
            elif not output.strip():
                last_code = "OUTPUT_EMPTY"
                last_detail = ""
                feedback = "上次返回为空。请直接输出完整 JSON 对象，不要留空。"
            else:
                try:
                    result = Extraction.model_validate_json(output)
                    validate_evidence(result, content)
                    if result.entities or result.relations or not rejected_candidate:
                        return result
                    last_code = "OUTPUT_EMPTY_AFTER_INVALID"
                    last_detail = ""
                    feedback = (
                        "先前的非空候选未通过校验，本次空图谱不能确认没有事实。"
                        "请仅保留有逐字证据的实体和关系。"
                    )
                except ValidationError as error:
                    rejected_candidate = True
                    try:
                        return validated_endpoint_subset(output, content)
                    except (ValueError, TypeError):
                        pass
                    last_code = "SCHEMA_INVALID"
                    last_detail = "; ".join(
                        f"{'.'.join(map(str, item['loc'])) or 'root'}:{item['type']}"
                        + (
                            f":{item['ctx']['error']}"
                            if item["type"] == "value_error" and item.get("ctx")
                            else ""
                        )
                        for item in error.errors(include_input=False)[:3]
                    )
                    details = [
                        f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
                        for item in error.errors(include_input=False)[:3]
                    ]
                    feedback = "上次 JSON 未通过结构或关系端点校验：" + "; ".join(details)
                except ValueError as error:
                    rejected_candidate = True
                    last_code = "EVIDENCE_INVALID"
                    last_detail = str(error)
                    feedback = "上次 JSON 未通过原文证据校验：" + str(error)
                    if str(error) in ("实体名称未出现在当前分块", "规范实体名称缺少原文依据"):
                        feedback += _entity_name_feedback(result, content)
            if attempt < 2:
                if output and choice.finish_reason != "length":
                    messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": feedback + "请重新输出完整 JSON。"})
        raise GraphExtractionError(last_code, last_detail)
    finally:
        if owns_client:
            await client.close()
