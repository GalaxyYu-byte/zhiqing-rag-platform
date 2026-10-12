"""直接查询授权目录，文档一行；计数和分页取同一数据库查询快照。"""

import hashlib
import json
from datetime import UTC, datetime

import psycopg
from psycopg import sql
from starlette.concurrency import run_in_threadpool

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.modules.knowledge.permissions import (
    document_access_sql,
    knowledge_base_access_sql,
)
from zhiqing_rag.query_metadata.compiler import compile_filters
from zhiqing_rag.query_retrieval.repository import QueryRetrievalRepository
from zhiqing_rag.query_routing.models import StructuredPlan
from zhiqing_rag.query_routing.scope import AccessScope

from .models import (
    MAX_STRUCTURED_DOCUMENTS,
    StructuredDocument,
    StructuredProvenance,
    StructuredResult,
    StructuredSource,
)

_DOCUMENT_COLUMNS = """doc.kb_id AS knowledge_base_id, doc.id AS document_id,
    r.id AS revision_id, g.id AS generation_id, doc.title,
    r.original_filename AS document_name, r.mime_type AS file_type,
    doc.business_status, doc.department_id, doc.event_date,
    r.created_at AS uploaded_at, doc.updated_at, doc.document_code,
    doc.confidentiality, doc.metadata"""
_SOURCE_COLUMNS = ("knowledge_base_id", "document_id", "revision_id", "generation_id")


def _query(connection, schema, statement, parameters=()):
    return connection.execute(
        sql.SQL(statement).format(schema=sql.Identifier(schema)), parameters
    ).fetchall()


def _document(row: dict) -> StructuredDocument:
    data = {name: row[name] for name in StructuredDocument.model_fields}
    for name in (*_SOURCE_COLUMNS, "department_id"):
        data[name] = str(data[name]) if data[name] is not None else None
    return StructuredDocument(**data)


def _source(row: dict) -> StructuredSource:
    document = _document(row)
    payload = document.model_dump(mode="json")
    payload.update({
        name: row.get(name) for name in ("document_code", "confidentiality", "metadata")
    })
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return StructuredSource(
        **{name: getattr(document, name) for name in _SOURCE_COLUMNS},
        updated_at=row["updated_at"],
        metadata_sha256=fingerprint,
    )


def _from_documents(identity, *, filters=()):
    predicate, arguments = document_access_sql(identity, published_only=True)
    filter_sql, filter_args = compile_filters(filters, document_alias="doc", revision_alias="r")
    return (
        """ FROM {schema}.document doc
        JOIN {schema}.knowledge_base kb ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
        JOIN {schema}.index_generation g ON g.tenant_id = doc.tenant_id
          AND g.document_id = doc.id AND g.id = doc.active_generation_id
        JOIN {schema}.document_revision r ON r.tenant_id = doc.tenant_id
          AND r.document_id = doc.id AND r.id = g.revision_id
        WHERE """ + predicate + " AND g.status = 'READY' AND (" + filter_sql + ")",
        (*arguments, *filter_args),
    )


def revalidate_provenance(connection, schema, identity, provenance: StructuredProvenance) -> bool:
    """复核全部参与来源和知识库锚点；零条结果也必须经过 ACL。"""
    if (int(provenance.tenant_id), int(provenance.user_id)) != (
        identity.tenant_id, identity.user_id,
    ):
        return False
    predicate, arguments = knowledge_base_access_sql(identity)
    kb_ids = [int(value) for value in provenance.knowledge_base_ids]
    anchors = _query(
        connection, schema,
        "SELECT kb.id FROM {schema}.knowledge_base kb WHERE " + predicate
        + " AND kb.id = ANY(%s::bigint[]) FOR SHARE OF kb",
        (*arguments, kb_ids),
    )
    if {row["id"] for row in anchors} != set(kb_ids):
        return False
    selected_ids = [int(value) for value in provenance.selected_document_ids]
    if selected_ids:
        predicate, arguments = document_access_sql(identity, published_only=False)
        selected = _query(
            connection, schema,
            """SELECT doc.id FROM {schema}.document doc
            JOIN {schema}.knowledge_base kb
              ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id WHERE """ + predicate
            + " AND doc.id = ANY(%s::bigint[]) AND doc.kb_id = ANY(%s::bigint[])"
            + " FOR SHARE OF doc, kb",
            (*arguments, selected_ids, kb_ids),
        )
        if {row["id"] for row in selected} != set(selected_ids):
            return False
    if not provenance.sources:
        return True
    statement, parameters = _from_documents(identity, filters=provenance.applied_filters)
    sources = provenance.sources
    arrays = [[int(getattr(source, name)) for source in sources] for name in _SOURCE_COLUMNS]
    rows = _query(
        connection, schema, "SELECT " + _DOCUMENT_COLUMNS + statement + """
        AND (doc.kb_id, doc.id, r.id, g.id) IN (
          SELECT * FROM unnest(%s::bigint[], %s::bigint[], %s::bigint[], %s::bigint[]))
        FOR SHARE OF doc, kb, g, r""",
        (*parameters, *arrays),
    )
    expected = {source.document_id: source for source in sources}
    actual = {source.document_id: source for source in map(_source, rows)}
    return expected == actual


class StructuredQueryRepository:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def execute(
        self, *, identity: AuthIdentity, scope: AccessScope, plan: StructuredPlan
    ) -> StructuredResult:
        if (identity.tenant_id, identity.user_id) != (scope.tenant_id, scope.user_id):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "统计范围与当前身份不一致")
        try:
            return await run_in_threadpool(self._execute, identity, scope, plan)
        except psycopg.Error as error:
            raise AuthError(503, "STRUCTURED_DATABASE_UNAVAILABLE", "统计服务暂不可用") from error

    async def revalidate(self, *, identity, scope, result: StructuredResult) -> bool:
        if (identity.tenant_id, identity.user_id) != (scope.tenant_id, scope.user_id):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "统计范围与当前身份不一致")
        provenance = result.provenance
        if provenance is None or provenance.knowledge_base_ids != tuple(
            map(str, scope.knowledge_base_ids)
        ):
            return False
        if provenance.selected_document_ids and provenance.selected_document_ids != tuple(
            map(str, scope.document_ids)
        ):
            return False
        allowed = {
            (str(doc.document_id), str(doc.knowledge_base_id), str(doc.generation_id))
            for doc in scope.published_documents
        }
        if any(
            (source.document_id, source.knowledge_base_id, source.generation_id) not in allowed
            for source in provenance.sources
        ):
            return False
        try:
            return await run_in_threadpool(self._revalidate, identity, provenance)
        except psycopg.Error as error:
            raise AuthError(
                503, "STRUCTURED_DATABASE_UNAVAILABLE", "统计复核服务暂不可用",
            ) from error

    def _revalidate(self, identity, provenance):
        with database_connection(self.settings) as connection:
            fresh = QueryRetrievalRepository(self.settings)._refresh_identity(connection, identity)
            return revalidate_provenance(connection, self.settings.db_schema, fresh, provenance)

    def _execute(self, identity, scope, plan):
        with database_connection(self.settings) as connection:
            fresh = QueryRetrievalRepository(self.settings)._refresh_identity(connection, identity)
            predicate, arguments = knowledge_base_access_sql(fresh)
            kb_ids = list(scope.knowledge_base_ids)
            anchors = _query(
                connection, self.settings.db_schema,
                "SELECT kb.id FROM {schema}.knowledge_base kb WHERE " + predicate
                + " AND kb.id = ANY(%s::bigint[]) FOR SHARE OF kb",
                (*arguments, kb_ids),
            )
            if {row["id"] for row in anchors} != set(kb_ids):
                raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "选定知识库授权已变化")
            filters = getattr(scope, "resolved_filters", ())
            statement, parameters = _from_documents(fresh, filters=filters)
            initial = scope.published_documents
            rows = _query(
                connection, self.settings.db_schema,
                "SELECT " + _DOCUMENT_COLUMNS + statement + """
                AND (doc.id, doc.kb_id, g.id) IN (
                  SELECT * FROM unnest(%s::bigint[], %s::bigint[], %s::bigint[]))
                ORDER BY r.created_at DESC, doc.id DESC LIMIT %s
                FOR SHARE OF doc, kb, g, r""",
                (*parameters,
                 [document.document_id for document in initial],
                 [document.knowledge_base_id for document in initial],
                 [document.generation_id for document in initial],
                 MAX_STRUCTURED_DOCUMENTS + 1),
            )
            if len(rows) > MAX_STRUCTURED_DOCUMENTS:
                raise AuthError(
                    422, "STRUCTURED_SCOPE_LIMIT",
                    "精确统计最多支持 10000 份文档，请缩小知识库或筛选条件后重试。",
                )
            executed_at = datetime.now(UTC)
            provenance = StructuredProvenance(
                tenant_id=str(scope.tenant_id), user_id=str(scope.user_id),
                knowledge_base_ids=tuple(map(str, scope.knowledge_base_ids)),
                selected_document_ids=tuple(map(str, scope.document_ids)),
                operation=plan.params.operation, applied_filters=filters,
                executed_at=executed_at, sources=tuple(map(_source, rows)),
            )
            start, limit = plan.params.offset, plan.params.limit
            return StructuredResult(
                operation=plan.params.operation, total=len(rows), offset=start, limit=limit,
                documents=tuple(map(_document, rows[start:start + limit]))
                if plan.params.operation == "list_documents" else (),
                applied_filters=filters, executed_at=executed_at, provenance=provenance,
            )
