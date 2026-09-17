"""在当前授权文档版本中匹配图谱实体，检查是否有可用的活动关系证据。"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from sqlalchemy import select

from ..core.neo4j import get_neo4j_session
from ..graph.identity import normalize_identity_text
from ..models.document import Document
from ..query_routing.models import GraphCoverage


_ENTITY_COVERAGE = """
UNWIND $entity_names AS name
CALL (name) {
    MATCH (seed:Entity)
    WHERE seed.kb_id IN $kb_ids
      AND (seed.canonical_name_normalized = name
           OR any(alias IN coalesce(seed.aliases, []) WHERE toLower(trim(alias)) = name))
    MATCH (seed)-[:SUBJECT_OF|OBJECT]-(claim:Claim)-[:SUPPORTED_BY]->(chunk:Chunk)
    MATCH (:Entity)-[:SUBJECT_OF]->(claim)-[:OBJECT]->(:Entity)
    MATCH (doc:Document {doc_id: chunk.doc_id})
    WHERE claim.kb_id = seed.kb_id AND chunk.kb_id = seed.kb_id
      AND doc.kb_id = chunk.kb_id
      AND chunk.graph_status = 'ACTIVE'
      AND doc.active_graph_version = chunk.doc_version
      AND any(version IN $document_versions
              WHERE version.doc_id = chunk.doc_id AND version.version = chunk.doc_version)
    RETURN DISTINCT seed.uid AS uid
    LIMIT 2
}
RETURN name, uid
"""


async def check_graph_coverage(
    session: Any, *, entity_names: list[str], kb_ids: list[int], doc_ids: list[int],
    graph_session_factory: Callable[[], Any] = get_neo4j_session,
) -> GraphCoverage:
    if not entity_names or not kb_ids or not doc_ids:
        return GraphCoverage("unmatched")
    names = list(dict.fromkeys(normalize_identity_text(name) for name in entity_names))
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
            kb_ids=sorted(set(kb_ids)), document_versions=versions,
        )
        records = await result.data()
    matches: dict[str, set[str]] = defaultdict(set)
    for record in records:
        name, uid = record.get("name"), record.get("uid")
        if name in names and isinstance(uid, str) and uid.strip():
            matches[name].add(uid)
    if any(len(matches[name]) > 1 for name in names):
        return GraphCoverage("ambiguous")
    if any(not matches[name] for name in names):
        return GraphCoverage("unmatched")
    return GraphCoverage("covered", tuple(sorted({uid for uids in matches.values() for uid in uids})))
