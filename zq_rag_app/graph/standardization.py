"""LLM 抽取结果的确定性实体标准化与局部去重。"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from .identity import build_entity_identity_key
from .models import ExtractedEntity, GraphExtraction


DEFAULT_CANONICAL_ALIASES = {
    "k8s": "Kubernetes",
    "redis 缓存": "Redis",
    "postgres": "PostgreSQL",
    "postgresql 数据库": "PostgreSQL",
}


def normalize_display_text(value: str) -> str:
    """统一全半角和空白，同时保留适合展示的大小写。"""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


class EntityStandardizer:
    """规范实体名称，并合并同一 Chunk 中身份完全相同的实体。"""

    def __init__(
        self,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        configured = aliases if aliases is not None else DEFAULT_CANONICAL_ALIASES
        self.aliases = {
            normalize_display_text(alias).casefold(): normalize_display_text(name)
            for alias, name in configured.items()
        }

    def standardize(self, extraction: GraphExtraction) -> GraphExtraction:
        extraction_by_key: dict[tuple[str, str], ExtractedEntity] = {}
        local_id_mapping: dict[str, str] = {}

        for source in extraction.entities:
            display_name = normalize_display_text(source.name)
            canonical_name = self.aliases.get(display_name.casefold(), display_name)
            aliases = [
                normalize_display_text(alias)
                for alias in source.aliases
                if normalize_display_text(alias)
            ]
            if canonical_name != display_name:
                aliases.append(display_name)

            standardized = source.model_copy(
                update={
                    "name": canonical_name,
                    "aliases": _deduplicate_names(canonical_name, aliases),
                    "description": (
                        normalize_display_text(source.description)
                        if source.description
                        else None
                    ),
                    "external_id_source": (
                        normalize_display_text(source.external_id_source)
                        if source.external_id_source
                        else None
                    ),
                    "external_id": (
                        normalize_display_text(source.external_id)
                        if source.external_id
                        else None
                    ),
                }
            )
            key = (
                str(standardized.entity_type),
                build_entity_identity_key(standardized),
            )
            existing = extraction_by_key.get(key)
            if existing is None:
                extraction_by_key[key] = standardized
                local_id_mapping[source.local_id] = standardized.local_id
                continue

            local_id_mapping[source.local_id] = existing.local_id
            existing.aliases = _deduplicate_names(
                existing.name,
                [
                    *existing.aliases,
                    standardized.name,
                    *standardized.aliases,
                ],
            )
            if existing.description is None:
                existing.description = standardized.description

        relations = []
        relation_keys: set[tuple[str, ...]] = set()
        for relation in extraction.relations:
            source_id = local_id_mapping[relation.source_local_id]
            target_id = local_id_mapping[relation.target_local_id]
            # 标准化后成为自环，说明模型把同一实体拆成了两份，丢弃该伪关系。
            if source_id == target_id:
                continue
            standardized_relation = relation.model_copy(
                update={
                    "source_local_id": source_id,
                    "target_local_id": target_id,
                }
            )
            key = (
                source_id,
                str(standardized_relation.predicate),
                target_id,
                str(standardized_relation.polarity),
                standardized_relation.valid_from or "",
                standardized_relation.valid_to or "",
            )
            if key not in relation_keys:
                relation_keys.add(key)
                relations.append(standardized_relation)

        return GraphExtraction(
            schema_version=extraction.schema_version,
            entities=list(extraction_by_key.values()),
            relations=relations,
        )


def _deduplicate_names(canonical_name: str, aliases: list[str]) -> list[str]:
    seen = {canonical_name.casefold()}
    result: list[str] = []
    for alias in aliases:
        key = alias.casefold()
        if key not in seen:
            seen.add(key)
            result.append(alias)
    return result[:10]
