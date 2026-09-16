"""经过约束保护的 Neo4j 图谱幂等写入。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.neo4j import get_neo4j_session
from .identity import (
    IDENTITY_VERSION,
    build_chunk_uid,
    build_claim_uid,
    build_evidence_uid,
    build_entity_identity_key,
    build_entity_uid,
    build_mention_uid,
    normalize_identity_text,
)
from .models import GraphChunkContext, GraphExtraction
from .schema import load_schema_statements


_UPSERT_CONTEXT = """
MERGE (kb:KnowledgeBase {kb_id: $kb_id})
ON CREATE SET kb.created_at = datetime()
SET kb.updated_at = datetime()
MERGE (doc:Document {doc_id: $doc_id})
ON CREATE SET doc.created_at = datetime()
SET doc.kb_id = $kb_id,
    doc.file_name = $file_name,
    doc.updated_at = datetime()
MERGE (kb)-[:CONTAINS]->(doc)
MERGE (version:DocumentVersion {doc_id: $doc_id, version: $doc_version})
ON CREATE SET version.created_at = datetime()
SET version.graph_status = CASE
        WHEN version.graph_status = 'ACTIVE' THEN 'ACTIVE'
        ELSE 'CANDIDATE'
    END,
    version.updated_at = datetime()
MERGE (doc)-[:HAS_VERSION]->(version)
MERGE (chunk:Chunk {uid: $chunk_uid})
ON CREATE SET chunk.created_at = datetime()
SET chunk.kb_id = $kb_id,
    chunk.doc_id = $doc_id,
    chunk.doc_version = $doc_version,
    chunk.chunk_index = $chunk_index,
    chunk.content = $content,
    chunk.page_num = $page_num,
    chunk.section_title = $section_title,
    chunk.graph_status = CASE
        WHEN chunk.graph_status = 'ACTIVE' THEN 'ACTIVE'
        ELSE 'CANDIDATE'
    END,
    chunk.updated_at = datetime()
MERGE (version)-[:HAS_CHUNK]->(chunk)
"""

_UPSERT_ENTITIES = """
MATCH (chunk:Chunk {uid: $chunk_uid})
UNWIND $entities AS row
MERGE (entity:Entity {uid: row.uid})
ON CREATE SET entity.created_at = datetime(),
              entity.identity_version = row.identity_version,
              entity.kb_id = row.kb_id,
              entity.entity_type = row.entity_type,
              entity.identity_key = row.identity_key,
              entity.canonical_name = row.canonical_name,
              entity.identity_status = row.identity_status,
              entity.external_id_source = row.external_id_source,
              entity.external_id = row.external_id
SET entity.aliases = reduce(
        aliases = coalesce(entity.aliases, []),
        alias IN row.aliases |
        CASE WHEN alias IN aliases THEN aliases ELSE aliases + alias END
    ),
    entity.description = coalesce(row.description, entity.description),
    entity.canonical_name_normalized = row.canonical_name_normalized,
    entity.updated_at = datetime()
MERGE (chunk)-[mention:MENTIONS {uid: row.mention_uid}]->(entity)
SET mention.quote = row.evidence_quote,
    mention.extractor_version = $extractor_version,
    mention.updated_at = datetime()
"""

_UPSERT_CLAIMS = """
MATCH (chunk:Chunk {uid: $chunk_uid})
UNWIND $claims AS row
MATCH (subject:Entity {uid: row.subject_uid})
MATCH (object:Entity {uid: row.object_uid})
MERGE (claim:Claim {uid: row.uid})
ON CREATE SET claim.created_at = datetime(),
              claim.identity_version = row.identity_version,
              claim.kb_id = row.kb_id,
              claim.predicate = row.predicate,
              claim.polarity = row.polarity,
              claim.valid_from = row.valid_from,
              claim.valid_to = row.valid_to
SET claim.updated_at = datetime()
MERGE (subject)-[:SUBJECT_OF]->(claim)
MERGE (claim)-[:OBJECT]->(object)
MERGE (claim)-[evidence:SUPPORTED_BY {uid: row.evidence_uid}]->(chunk)
SET evidence.quote = row.evidence_quote,
    evidence.confidence = row.confidence,
    evidence.extractor_version = $extractor_version,
    evidence.updated_at = datetime()
"""

_DELETE_DOCUMENT_VERSION = """
MATCH (version:DocumentVersion {doc_id: $doc_id, version: $doc_version})
OPTIONAL MATCH (version)-[:HAS_CHUNK]->(chunk:Chunk)
DETACH DELETE chunk, version
"""

_DELETE_ORPHAN_CLAIMS = """
MATCH (claim:Claim)
WHERE NOT (claim)-[:SUPPORTED_BY]->(:Chunk)
DETACH DELETE claim
"""

_DELETE_ORPHAN_ENTITIES = """
MATCH (entity:Entity)
WHERE NOT ()-[:MENTIONS]->(entity)
  AND NOT (entity)-[:SUBJECT_OF]->(:Claim)
  AND NOT (:Claim)-[:OBJECT]->(entity)
DETACH DELETE entity
"""

_READ_CHUNK_SUMMARY = """
MATCH (chunk:Chunk {uid: $chunk_uid})
OPTIONAL MATCH (chunk)-[mention:MENTIONS]->(entity:Entity)
WITH chunk,
     count(DISTINCT entity) AS entity_count,
     count(DISTINCT mention) AS mention_count
OPTIONAL MATCH (claim:Claim)-[evidence:SUPPORTED_BY]->(chunk)
RETURN chunk.uid AS chunk_uid,
       entity_count,
       mention_count,
       count(DISTINCT claim) AS claim_count,
       count(DISTINCT evidence) AS evidence_count
"""

_BEGIN_DOCUMENT_VERSION = """
MERGE (kb:KnowledgeBase {kb_id: $kb_id})
ON CREATE SET kb.created_at = datetime()
SET kb.updated_at = datetime()
MERGE (doc:Document {doc_id: $doc_id})
ON CREATE SET doc.created_at = datetime()
SET doc.kb_id = $kb_id,
    doc.file_name = $file_name,
    doc.updated_at = datetime()
MERGE (kb)-[:CONTAINS]->(doc)
MERGE (version:DocumentVersion {doc_id: $doc_id, version: $doc_version})
ON CREATE SET version.created_at = datetime()
SET version.graph_status = CASE
        WHEN version.graph_status = 'ACTIVE' THEN 'ACTIVE'
        ELSE 'CANDIDATE'
    END,
    version.updated_at = datetime()
MERGE (doc)-[:HAS_VERSION]->(version)
"""

_ACTIVATE_DOCUMENT_VERSION = """
MATCH (doc:Document {doc_id: $doc_id})-[:HAS_VERSION]->
      (version:DocumentVersion {doc_id: $doc_id, version: $doc_version})
OPTIONAL MATCH (version)-[:HAS_CHUNK]->(chunk:Chunk)
WITH doc, version, collect(DISTINCT chunk) AS chunks
WHERE size(chunks) = $expected_chunk_count
  AND coalesce(doc.active_graph_version, 0) <= $doc_version
OPTIONAL MATCH (doc)-[:HAS_VERSION]->(other:DocumentVersion)
WITH doc, version, chunks, collect(DISTINCT other) AS versions
FOREACH (item IN versions |
    SET item.graph_status = CASE
        WHEN item.version = $doc_version THEN 'ACTIVE'
        ELSE 'RETIRED'
    END
)
FOREACH (item IN chunks | SET item.graph_status = 'ACTIVE')
SET version.graph_status = 'ACTIVE',
    doc.active_graph_version = $doc_version,
    doc.graph_updated_at = datetime(),
    doc.updated_at = datetime()
RETURN size(chunks) AS chunk_count
"""

_PRUNE_INACTIVE_DOCUMENT_VERSIONS = """
MATCH (doc:Document {doc_id: $doc_id})-[:HAS_VERSION]->
      (version:DocumentVersion)
WHERE version.version <> $keep_version
OPTIONAL MATCH (version)-[:HAS_CHUNK]->(chunk:Chunk)
DETACH DELETE chunk, version
"""


@dataclass(slots=True, frozen=True)
class GraphWriteResult:
    chunk_uid: str
    entity_count: int
    claim_count: int


@dataclass(slots=True, frozen=True)
class GraphChunkSummary:
    chunk_uid: str
    entity_count: int
    mention_count: int
    claim_count: int
    evidence_count: int


@dataclass(slots=True, frozen=True)
class _GraphWritePayload:
    context: dict[str, Any]
    entities: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    extractor_version: str


class GraphRepository:
    """把已验证的抽取结果转换为稳定身份并写入 Neo4j。"""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Any] = get_neo4j_session,
    ) -> None:
        self.session_factory = session_factory

    async def ensure_schema(self) -> None:
        """幂等创建图谱约束和查询索引。"""

        async with self.session_factory() as session:
            for statement in load_schema_statements():
                result = await session.run(statement)
                await result.consume()

    async def upsert_extraction(
        self,
        *,
        context: GraphChunkContext,
        extraction: GraphExtraction,
        extractor_version: str,
    ) -> GraphWriteResult:
        extraction.validate_evidence(context.content)
        payload = self.prepare_payload(
            context=context,
            extraction=extraction,
            extractor_version=extractor_version,
        )
        async with self.session_factory() as session:
            await session.execute_write(self._write_payload, payload)
        return GraphWriteResult(
            chunk_uid=payload.context["chunk_uid"],
            entity_count=len(payload.entities),
            claim_count=len(payload.claims),
        )

    async def begin_document_version(
        self,
        *,
        kb_id: int,
        doc_id: int,
        doc_version: int,
        file_name: str,
    ) -> None:
        """幂等创建不可见的候选文档图版本，支持零 Chunk 文档。"""

        if min(kb_id, doc_id, doc_version) <= 0 or not file_name.strip():
            raise ValueError("图版本上下文参数无效")
        async with self.session_factory() as session:
            result = await session.run(
                _BEGIN_DOCUMENT_VERSION,
                kb_id=kb_id,
                doc_id=doc_id,
                doc_version=doc_version,
                file_name=file_name.strip(),
            )
            await result.consume()

    async def activate_document_version(
        self,
        *,
        doc_id: int,
        doc_version: int,
        expected_chunk_count: int,
    ) -> None:
        """仅在候选 Chunk 完整时，原子切换文档的活动图版本。"""

        if doc_id <= 0 or doc_version <= 0 or expected_chunk_count < 0:
            raise ValueError("活动图版本参数无效")
        async with self.session_factory() as session:
            switched = await session.execute_write(
                self._activate_version,
                doc_id,
                doc_version,
                expected_chunk_count,
            )
        if not switched:
            raise RuntimeError(
                "候选图版本不存在或 Chunk 数量不完整，拒绝切换活动版本"
            )

    async def prune_inactive_document_versions(
        self,
        *,
        doc_id: int,
        keep_version: int,
    ) -> None:
        """活动版本切换后清理旧证据，再回收无证据事实与孤立实体。"""

        if doc_id <= 0 or keep_version <= 0:
            raise ValueError("待清理图版本参数无效")
        async with self.session_factory() as session:
            await session.execute_write(
                self._prune_versions,
                doc_id,
                keep_version,
            )

    @staticmethod
    def prepare_payload(
        *,
        context: GraphChunkContext,
        extraction: GraphExtraction,
        extractor_version: str,
    ) -> _GraphWritePayload:
        """纯函数式准备写入参数，便于离线审计和单元测试。"""

        if not extractor_version.strip():
            raise ValueError("extractor_version 不能为空")

        chunk_uid = build_chunk_uid(context)
        local_uid: dict[str, str] = {}
        entities: list[dict[str, Any]] = []
        for entity in extraction.entities:
            uid = build_entity_uid(context.kb_id, entity)
            local_uid[entity.local_id] = uid
            entities.append(
                {
                    "uid": uid,
                    "identity_version": IDENTITY_VERSION,
                    "kb_id": context.kb_id,
                    "entity_type": entity.entity_type,
                    "identity_key": build_entity_identity_key(entity),
                    "identity_status": (
                        "AUTHORITATIVE"
                        if entity.external_id is not None
                        else "PROVISIONAL"
                    ),
                    "canonical_name": entity.name,
                    "canonical_name_normalized": normalize_identity_text(
                        entity.name
                    ),
                    "aliases": entity.aliases,
                    "description": entity.description,
                    "external_id_source": entity.external_id_source,
                    "external_id": entity.external_id,
                    "evidence_quote": entity.evidence_quote,
                    "mention_uid": build_mention_uid(chunk_uid, uid),
                }
            )

        claims: list[dict[str, Any]] = []
        for relation in extraction.relations:
            subject_uid = local_uid[relation.source_local_id]
            object_uid = local_uid[relation.target_local_id]
            claim_uid = build_claim_uid(
                kb_id=context.kb_id,
                subject_uid=subject_uid,
                object_uid=object_uid,
                relation=relation,
            )
            claims.append(
                {
                    "uid": claim_uid,
                    "identity_version": IDENTITY_VERSION,
                    "kb_id": context.kb_id,
                    "subject_uid": subject_uid,
                    "object_uid": object_uid,
                    "predicate": relation.predicate,
                    "polarity": relation.polarity,
                    "confidence": relation.confidence,
                    "evidence_quote": relation.evidence_quote,
                    "valid_from": relation.valid_from,
                    "valid_to": relation.valid_to,
                    "evidence_uid": build_evidence_uid(claim_uid, chunk_uid),
                }
            )

        return _GraphWritePayload(
            context={
                **context.model_dump(),
                "chunk_uid": chunk_uid,
            },
            entities=entities,
            claims=claims,
            extractor_version=extractor_version.strip(),
        )

    async def delete_document_version(
        self,
        *,
        doc_id: int,
        doc_version: int,
        collect_orphans: bool = True,
    ) -> None:
        """删除一个图版本的证据，并按需回收无证据事实和孤立实体。"""

        if doc_id <= 0 or doc_version <= 0:
            raise ValueError("doc_id 和 doc_version 必须大于 0")
        async with self.session_factory() as session:
            await session.execute_write(
                self._delete_version,
                doc_id,
                doc_version,
                collect_orphans,
            )

    async def get_chunk_summary(self, chunk_uid: str) -> GraphChunkSummary | None:
        """读取一个 Chunk 的去重节点和关系计数，用于写后审计。"""

        if not chunk_uid.strip():
            raise ValueError("chunk_uid 不能为空")
        async with self.session_factory() as session:
            return await session.execute_read(self._read_chunk_summary, chunk_uid)

    @staticmethod
    async def _write_payload(tx: Any, payload: _GraphWritePayload) -> None:
        context_parameters = {
            **payload.context,
            "extractor_version": payload.extractor_version,
        }
        result = await tx.run(_UPSERT_CONTEXT, **context_parameters)
        await result.consume()
        if payload.entities:
            result = await tx.run(
                _UPSERT_ENTITIES,
                chunk_uid=payload.context["chunk_uid"],
                entities=payload.entities,
                extractor_version=payload.extractor_version,
            )
            await result.consume()
        if payload.claims:
            result = await tx.run(
                _UPSERT_CLAIMS,
                chunk_uid=payload.context["chunk_uid"],
                claims=payload.claims,
                extractor_version=payload.extractor_version,
            )
            await result.consume()

    @staticmethod
    async def _delete_version(
        tx: Any,
        doc_id: int,
        doc_version: int,
        collect_orphans: bool,
    ) -> None:
        result = await tx.run(
            _DELETE_DOCUMENT_VERSION,
            doc_id=doc_id,
            doc_version=doc_version,
        )
        await result.consume()
        if collect_orphans:
            for query in (_DELETE_ORPHAN_CLAIMS, _DELETE_ORPHAN_ENTITIES):
                result = await tx.run(query)
                await result.consume()

    @staticmethod
    async def _activate_version(
        tx: Any,
        doc_id: int,
        doc_version: int,
        expected_chunk_count: int,
    ) -> bool:
        result = await tx.run(
            _ACTIVATE_DOCUMENT_VERSION,
            doc_id=doc_id,
            doc_version=doc_version,
            expected_chunk_count=expected_chunk_count,
        )
        record = await result.single()
        return record is not None

    @staticmethod
    async def _prune_versions(
        tx: Any,
        doc_id: int,
        keep_version: int,
    ) -> None:
        result = await tx.run(
            _PRUNE_INACTIVE_DOCUMENT_VERSIONS,
            doc_id=doc_id,
            keep_version=keep_version,
        )
        await result.consume()
        for query in (_DELETE_ORPHAN_CLAIMS, _DELETE_ORPHAN_ENTITIES):
            result = await tx.run(query)
            await result.consume()

    @staticmethod
    async def _read_chunk_summary(
        tx: Any,
        chunk_uid: str,
    ) -> GraphChunkSummary | None:
        result = await tx.run(_READ_CHUNK_SUMMARY, chunk_uid=chunk_uid)
        record = await result.single()
        if record is None:
            return None
        return GraphChunkSummary(
            chunk_uid=record["chunk_uid"],
            entity_count=record["entity_count"],
            mention_count=record["mention_count"],
            claim_count=record["claim_count"],
            evidence_count=record["evidence_count"],
        )
