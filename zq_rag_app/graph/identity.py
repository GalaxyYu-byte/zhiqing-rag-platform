"""图节点业务身份归一化与确定性 UID。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence

from .models import ExtractedEntity, ExtractedRelation, GraphChunkContext


IDENTITY_VERSION = "v1"


def normalize_identity_text(value: str) -> str:
    """生成仅用于匹配的规范文本，不修改面向用户展示的名称。"""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    if not normalized:
        raise ValueError("实体身份文本不能为空")
    return normalized


def stable_uid(kind: str, parts: Sequence[object]) -> str:
    """用无歧义 JSON 编码生成带类型前缀的稳定 SHA-256 UID。"""

    payload = json.dumps(
        [IDENTITY_VERSION, kind, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{kind}:{IDENTITY_VERSION}:{digest}"


def build_entity_identity_key(entity: ExtractedEntity) -> str:
    if entity.external_id is not None:
        source = normalize_identity_text(entity.external_id_source or "")
        external_id = normalize_identity_text(entity.external_id)
        return f"ext:{source}:{external_id}"
    return f"name:{normalize_identity_text(entity.name)}"


def build_entity_uid(kb_id: int, entity: ExtractedEntity) -> str:
    return stable_uid(
        "entity",
        [kb_id, entity.entity_type, build_entity_identity_key(entity)],
    )


def build_chunk_uid(context: GraphChunkContext) -> str:
    return stable_uid(
        "chunk",
        [
            context.kb_id,
            context.doc_id,
            context.doc_version,
            context.chunk_index,
        ],
    )


def build_mention_uid(chunk_uid: str, entity_uid: str) -> str:
    return stable_uid("mention", [chunk_uid, entity_uid])


def build_evidence_uid(claim_uid: str, chunk_uid: str) -> str:
    return stable_uid("evidence", [claim_uid, chunk_uid])


def build_claim_uid(
    *,
    kb_id: int,
    subject_uid: str,
    object_uid: str,
    relation: ExtractedRelation,
) -> str:
    return stable_uid(
        "claim",
        [
            kb_id,
            subject_uid,
            relation.predicate,
            object_uid,
            relation.polarity,
            relation.valid_from,
            relation.valid_to,
        ],
    )
