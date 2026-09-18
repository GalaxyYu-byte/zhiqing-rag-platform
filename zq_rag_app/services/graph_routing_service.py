"""在当前授权文档版本中匹配图谱实体，检查是否有可用的活动关系证据。"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from ..core.neo4j import get_neo4j_session
from ..graph.identity import normalize_identity_text
from ..models.document import Document
from ..query_routing.models import GraphCoverage, GraphEntityCandidate


_ENTITY_COVERAGE = """
UNWIND $entity_candidates AS candidate
CALL (candidate) {
    MATCH (seed:Entity)
    WHERE seed.kb_id IN $kb_ids
      AND (candidate.entity_type IS NULL OR seed.entity_type = candidate.entity_type)
      AND (seed.canonical_name_normalized = candidate.name
           OR any(alias IN coalesce(seed.aliases, []) WHERE toLower(trim(alias)) = candidate.name))
    MATCH (seed)-[:SUBJECT_OF|OBJECT]-(claim:Claim)-[:SUPPORTED_BY]->(chunk:Chunk)
    MATCH (:Entity)-[:SUBJECT_OF]->(claim)-[:OBJECT]->(:Entity)
    MATCH (doc:Document {doc_id: chunk.doc_id})
    WHERE claim.kb_id = seed.kb_id AND chunk.kb_id = seed.kb_id
      AND doc.kb_id = chunk.kb_id
      AND chunk.graph_status = 'ACTIVE'
      AND doc.active_graph_version = chunk.doc_version
      AND any(version IN $document_versions
              WHERE version.doc_id = chunk.doc_id AND version.version = chunk.doc_version)
      AND all(qualifier IN candidate.qualifiers WHERE EXISTS {
          MATCH (seed)-[:SUBJECT_OF|OBJECT]-(identity_claim:Claim)-[:SUBJECT_OF|OBJECT]-(scope:Entity)
          MATCH (identity_claim)-[:SUPPORTED_BY]->(identity_chunk:Chunk)
          MATCH (identity_doc:Document {doc_id: identity_chunk.doc_id})
          WHERE identity_claim.polarity = 'POSITIVE'
            AND identity_claim.predicate IN ['PART_OF', 'OWNED_BY', 'LOCATED_IN']
            AND identity_claim.kb_id = seed.kb_id AND scope.kb_id = seed.kb_id
            AND identity_chunk.kb_id = seed.kb_id AND identity_doc.kb_id = seed.kb_id
            AND (scope.canonical_name_normalized = qualifier
                 OR any(alias IN coalesce(scope.aliases, []) WHERE toLower(trim(alias)) = qualifier))
            AND identity_chunk.graph_status = 'ACTIVE'
            AND identity_doc.active_graph_version = identity_chunk.doc_version
            AND any(version IN $document_versions
                    WHERE version.doc_id = identity_chunk.doc_id AND version.version = identity_chunk.doc_version)
      })
    RETURN DISTINCT seed.uid AS uid
    LIMIT 2
}
RETURN candidate.index AS candidate_index, candidate.name AS name, uid
"""


async def check_graph_coverage(
    session: Any, *, entity_names: list[str], kb_ids: list[int], doc_ids: list[int],
    entity_candidates: list[GraphEntityCandidate] | None = None,
    graph_session_factory: Callable[[], Any] = get_neo4j_session,
) -> GraphCoverage:
    if not entity_names or not kb_ids or not doc_ids:
        return GraphCoverage("unmatched")
    names = list(dict.fromkeys(normalize_identity_text(name) for name in entity_names))
    candidates = entity_candidates if entity_candidates is not None else [
        GraphEntityCandidate(name=name) for name in names
    ]
    if not candidates or not any(candidate.required for candidate in candidates):
        return GraphCoverage("unmatched")
    parameters = [
        {"index": index, "name": normalize_identity_text(candidate.name),
         "entity_type": candidate.entity_type,
         "qualifiers": [normalize_identity_text(value) for value in candidate.qualifiers]}
        for index, candidate in enumerate(candidates)
    ]
    try:
        rows = (await session.execute(select(Document.id, Document.version).where(
            Document.id.in_(doc_ids), Document.kb_id.in_(kb_ids),
            Document.is_deleted.is_(False), Document.status == "DONE",
        ))).all()
    except asyncio.CancelledError:
        # SQL 请求被整体检查超时取消后，恢复只读事务，才能用同一 Session 执行基础召回。
        await session.rollback()
        raise
    versions = [{"doc_id": int(row.id), "version": int(row.version)} for row in rows]
    if not versions:
        return GraphCoverage("unmatched")
    async with graph_session_factory() as graph_session:
        result = await graph_session.run(
            _ENTITY_COVERAGE, entity_names=names,
            entity_candidates=parameters,
            kb_ids=sorted(set(kb_ids)), document_versions=versions,
        )
        records = await result.data()
    matches: dict[int, set[str]] = defaultdict(set)
    for record in records:
        name, uid = record.get("name"), record.get("uid")
        index = record.get("candidate_index")
        # 兼容旧的按名称调用；带类型/角色的请求必须按候选索引解析，不能合并同名不同类型。
        if index is None and entity_candidates is None and name in names:
            index = names.index(name)
        if isinstance(index, int) and 0 <= index < len(candidates) and isinstance(uid, str) and uid.strip():
            matches[index].add(uid)
    required = [index for index, candidate in enumerate(candidates) if candidate.required]
    if any(len(matches[index]) > 1 for index in required):
        return GraphCoverage("ambiguous")
    if any(not matches[index] for index in required):
        return GraphCoverage("unmatched")
    unresolved = tuple(candidate.name for index, candidate in enumerate(candidates)
                       if not candidate.required and len(matches[index]) != 1)
    # 同名辅助对象不猜测身份，也不能将它们全部作为种子加入召回。
    seeds = tuple(sorted({next(iter(uids)) for uids in matches.values() if len(uids) == 1}))
    return GraphCoverage("partial" if unresolved else "covered", seeds, unresolved)
