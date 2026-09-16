"""LLM 图谱抽取协议和可信写入上下文。"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntityType(StrEnum):
    """第一版图谱允许抽取的实体类型。"""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    DEPARTMENT = "DEPARTMENT"
    SYSTEM = "SYSTEM"
    SERVICE = "SERVICE"
    PRODUCT = "PRODUCT"
    TECHNOLOGY = "TECHNOLOGY"
    PROCESS = "PROCESS"
    POLICY = "POLICY"
    EVENT = "EVENT"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"


class Predicate(StrEnum):
    """第一版图谱允许抽取的事实谓词。"""

    PART_OF = "PART_OF"
    OWNED_BY = "OWNED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    USES = "USES"
    IMPACTS = "IMPACTS"
    CAUSED_BY = "CAUSED_BY"
    RESOLVED_BY = "RESOLVED_BY"
    LOCATED_IN = "LOCATED_IN"
    RELATED_TO = "RELATED_TO"


class Polarity(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class _ExtractionModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True,
    )


class ExtractedEntity(_ExtractionModel):
    """模型在单个 Chunk 内识别出的局部实体。"""

    local_id: str = Field(pattern=r"^e[1-9]\d*$")
    entity_type: EntityType
    name: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=10)
    description: str | None = Field(default=None, max_length=500)
    external_id_source: str | None = Field(default=None, max_length=100)
    external_id: str | None = Field(default=None, max_length=200)
    evidence_quote: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_external_identity(self) -> "ExtractedEntity":
        if (self.external_id_source is None) != (self.external_id is None):
            raise ValueError("external_id_source 和 external_id 必须同时提供")

        unique_aliases: list[str] = []
        seen = {self.name.casefold()}
        for alias in self.aliases:
            alias = alias.strip()
            key = alias.casefold()
            if alias and key not in seen:
                seen.add(key)
                unique_aliases.append(alias)
        self.aliases = unique_aliases
        return self


class ExtractedRelation(_ExtractionModel):
    """模型识别出的有向事实以及其原文证据。"""

    source_local_id: str = Field(pattern=r"^e[1-9]\d*$")
    target_local_id: str = Field(pattern=r"^e[1-9]\d*$")
    predicate: Predicate
    polarity: Polarity = Polarity.POSITIVE
    confidence: float = Field(ge=0, le=1)
    evidence_quote: str = Field(min_length=1, max_length=1000)
    valid_from: str | None = Field(default=None, max_length=50)
    valid_to: str | None = Field(default=None, max_length=50)

    @model_validator(mode="after")
    def reject_self_relation(self) -> "ExtractedRelation":
        if self.source_local_id == self.target_local_id:
            raise ValueError("关系起点和终点不能是同一个局部实体")
        return self


class GraphExtraction(_ExtractionModel):
    """一次 Chunk 抽取的完整、严格输出。"""

    schema_version: str = Field(pattern=r"^1\.0$")
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=100)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def validate_graph_references(self) -> "GraphExtraction":
        local_ids = [entity.local_id for entity in self.entities]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("entity.local_id 不能重复")

        known_ids = set(local_ids)
        relation_keys: set[tuple[str, ...]] = set()
        for relation in self.relations:
            if relation.source_local_id not in known_ids:
                raise ValueError(
                    f"找不到 source_local_id={relation.source_local_id}"
                )
            if relation.target_local_id not in known_ids:
                raise ValueError(
                    f"找不到 target_local_id={relation.target_local_id}"
                )
            key = (
                relation.source_local_id,
                relation.predicate,
                relation.target_local_id,
                relation.polarity,
                relation.valid_from or "",
                relation.valid_to or "",
            )
            if key in relation_keys:
                raise ValueError(f"关系重复: {key}")
            relation_keys.add(key)
        return self

    def validate_evidence(self, source_text: str) -> None:
        """确认所有证据来自当前 Chunk，并回填精确的原文切片。

        模型偶尔会把中文弯引号改成英文直引号，或折叠连续空白。这里仅把
        这些不改变语义的排版差异用于定位；定位成功后保存的仍是原文切片。
        任何无法安全定位的证据都会被拒绝。
        """

        for holder in [*self.entities, *self.relations]:
            quote = holder.evidence_quote
            span = _locate_evidence_span(source_text, quote)
            if span is None:
                raise ValueError(f"证据不在当前 Chunk 中: {quote!r}")
            holder.evidence_quote = source_text[span[0] : span[1]]

    def discard_unsupported_evidence(self, source_text: str) -> tuple[int, int]:
        """丢弃无原文证据的模型候选，并保留可逐字回填的事实。

        无证据实体被删除后，引用它的关系也必须删除。返回删除的实体数和
        关系数，供调用层记录可观测日志。该方法不会放宽证据匹配规则。
        """

        valid_entities: list[ExtractedEntity] = []
        removed_entity_ids: set[str] = set()
        for entity in self.entities:
            span = _locate_evidence_span(source_text, entity.evidence_quote)
            if span is None:
                removed_entity_ids.add(entity.local_id)
                continue
            entity.evidence_quote = source_text[span[0] : span[1]]
            valid_entities.append(entity)

        valid_relations: list[ExtractedRelation] = []
        for relation in self.relations:
            if (
                relation.source_local_id in removed_entity_ids
                or relation.target_local_id in removed_entity_ids
            ):
                continue
            span = _locate_evidence_span(source_text, relation.evidence_quote)
            if span is None:
                continue
            relation.evidence_quote = source_text[span[0] : span[1]]
            valid_relations.append(relation)

        removed_entities = len(self.entities) - len(valid_entities)
        removed_relations = len(self.relations) - len(valid_relations)
        self.entities = valid_entities
        self.relations = valid_relations
        return removed_entities, removed_relations


class GraphChunkContext(BaseModel):
    """由应用注入的可信 Chunk 元数据，不允许模型生成。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kb_id: int = Field(gt=0)
    doc_id: int = Field(gt=0)
    doc_version: int = Field(gt=0)
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    file_name: str = Field(min_length=1, max_length=255)
    page_num: int | None = Field(default=None, gt=0)
    section_title: str | None = Field(default=None, max_length=500)


class GraphImportSample(BaseModel):
    """固定 JSON 导入文件的顶层协议。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    extractor_version: str = Field(min_length=1, max_length=100)
    context: GraphChunkContext
    extraction: GraphExtraction


_EQUIVALENT_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "＂": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "＇": "'",
    }
)


def _normalize_evidence_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.translate(_EQUIVALENT_QUOTES)).strip()


def _indexed_evidence_text(value: str) -> tuple[str, list[int], list[int]]:
    """生成可定位的规范文本及每个字符对应的原文起止位置。"""

    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, char in enumerate(value):
        if char.isspace():
            if normalized and normalized[-1] == " ":
                ends[-1] = index + 1
                continue
            normalized.append(" ")
        else:
            normalized.append(char.translate(_EQUIVALENT_QUOTES))
        starts.append(index)
        ends.append(index + 1)
    return "".join(normalized), starts, ends


def _locate_evidence_span(source_text: str, quote: str) -> tuple[int, int] | None:
    direct_start = source_text.find(quote)
    if direct_start >= 0:
        return direct_start, direct_start + len(quote)

    needle = _normalize_evidence_text(quote)
    if not needle:
        return None
    haystack, starts, ends = _indexed_evidence_text(source_text)
    normalized_start = haystack.find(needle)
    if normalized_start < 0:
        return None
    normalized_end = normalized_start + len(needle) - 1
    return starts[normalized_start], ends[normalized_end]
