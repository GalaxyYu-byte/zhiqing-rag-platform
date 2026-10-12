"""来源支持的实体身份基础；规范名相同只用于找候选，不能单独证明共指。"""

import re
from dataclasses import dataclass
from pathlib import PurePath

from zhiqing_rag.workers.graph_extraction import normalize_name, stable_key

from .models import ChunkRecord, GraphBridgeEvidence, GraphSourceEvidence

IDENTITY_POLICY_VERSION = 1

_ALIAS_PATTERN = re.compile(
    r"([\w\u4e00-\u9fff]{2,80})\s*[（(]?\s*"
    r"(?:以下简称|简称|又称|别名为)\s*[“\"《]?([^，,。；;）)》”\"\s]{2,80})"
)


def explicit_aliases(name: str, quote: str) -> tuple[str, ...]:
    """只取当前名称后原句明确声明的别名，禁止把 LLM 规范名差异当别名。"""
    result = []
    for match in _ALIAS_PATTERN.finditer(quote):
        full, alias = match.groups()
        if normalize_name(name) in {normalize_name(full), normalize_name(alias)}:
            result.extend((full, alias))
    return tuple(dict.fromkeys(result))


def explicit_contexts(names, entity_name: str, quote: str) -> tuple[str, ...]:
    if entity_name not in quote:
        return ()
    contexts = set()
    for clause in re.split(r"[。；;\n]", quote):
        if entity_name not in clause:
            continue
        contexts.update(
            normalize_name(name)
            for name in names
            if re.search(re.escape(name) + r"(?:中|内|的)", clause)
        )
        contexts.update(
            normalize_name(match.group(1))
            for match in re.finditer(
                re.escape(entity_name) + r"(?:仅)?用于([^，,。；;\n]{2,80}流程)", clause
            )
        )
    return tuple(sorted(contexts))


@dataclass(frozen=True, slots=True)
class VerifiedMention:
    uid: str
    entity_type: str
    name: str
    names: frozenset[str]
    contexts: frozenset[str]
    source: GraphSourceEvidence
    chunk: ChunkRecord


def _identity_denied(mention: VerifiedMention) -> bool:
    """共指依据不能覆盖原文明确的非同一对象、版本隔离或禁止混用声明。"""
    for clause in re.split(r"[。；;\n]", normalize_name(mention.chunk.content)):
        if not any(name in clause for name in mention.names):
            continue
        if re.search(
            r"(?:不是|并非|不属于|不代表|不等同于|不能视为).{0,20}同一"
            r"|同名.{0,20}(?:不同|不代表|不等于)"
            r"|(?:禁止|不得|不能|不可|不应).{0,12}(?:混用|合并|共指|视为同一)"
            r"|(?:旧版|历史版|旧表).{0,40}(?:不适用|不得|不能|不可).{0,40}(?:新版|当前|本次)",
            clause,
        ):
            return True
        if re.search(r"(?:属于|是).{0,8}不同(?:材料|对象|表单|版本)", clause):
            return True
        repeated = any(clause.count(name) >= 2 for name in mention.names)
        if repeated and re.search(
            r"(?:并)?不(?:相同|等同)|属于不同(?:材料|对象|表单|版本)"
            r"|(?:不得|不能|不可|不应).{0,8}等同",
            clause,
        ):
            return True
    return False


def identity_conflict(left: VerifiedMention, right: VerifiedMention) -> bool:
    """同一个旧图 UID 也不能覆盖真实来源中的对象否定与明确上下文冲突。"""
    return (
        _identity_denied(left)
        or _identity_denied(right)
        or bool(left.contexts and right.contexts and not left.contexts & right.contexts)
    )


def identity_bridge(left: VerifiedMention, right: VerifiedMention) -> GraphBridgeEvidence | None:
    if identity_conflict(left, right):
        return None
    if left.entity_type != right.entity_type or left.entity_type not in {
        "Material",
        "Process",
        "Actor",
    }:
        return None
    if (left.chunk.tenant_id, left.chunk.knowledge_base_id) != (
        right.chunk.tenant_id,
        right.chunk.knowledge_base_id,
    ):
        return None
    common = left.names & right.names
    if not common:
        return None
    basis = None
    if left.contexts and right.contexts and not left.contexts & right.contexts:
        return None
    titles = {normalize_name(PurePath(item.chunk.document_name).stem) for item in (left, right)} | {
        normalize_name(item.chunk.document_title) for item in (left, right)
    }
    if left.contexts & right.contexts:
        basis = "shared_context"
    elif left.entity_type in {"Material", "Process"}:
        # 文档自身以这个业务对象命名，形成‘流程引用材料 -> 材料本体文档’的出处锚。
        if titles & common:
            basis = "explicit_document"
    if basis is None:
        return None
    if normalize_name(left.name) != normalize_name(right.name):
        basis = "explicit_alias"
    declared_full = {
        normalize_name(match.group(1))
        for item in (left, right)
        for match in _ALIAS_PATTERN.finditer(item.chunk.content)
    }
    # 原文标题/显式别名声明的全称优先，抽取器选择简称作 canonical 不改变派生身份。
    canonical = sorted(common & titles or common & declared_full or common)[0]
    context = sorted(left.contexts & right.contexts or left.contexts | right.contexts)
    identity_id = stable_key(
        "kb-entity-identity",
        IDENTITY_POLICY_VERSION,
        left.chunk.tenant_id,
        left.chunk.knowledge_base_id,
        left.entity_type,
        canonical,
        context,
    )
    return GraphBridgeEvidence(
        identity_id=identity_id,
        identity_version=IDENTITY_POLICY_VERSION,
        source_entity_uid=left.uid,
        target_entity_uid=right.uid,
        entity_type=left.entity_type,
        canonical_name=canonical,
        basis=basis,
        sources=(left.source, right.source),
    )
