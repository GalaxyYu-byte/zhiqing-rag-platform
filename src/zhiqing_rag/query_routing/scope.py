"""服务端读取授权范围；计划本身不授予权限，执行前仍需重新校验。"""

from dataclasses import dataclass

from psycopg import Connection, sql

from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.knowledge.permissions import (
    Identity,
    document_access_sql,
    knowledge_base_access_sql,
)
from zhiqing_rag.query_metadata.models import ResolvedFilter

from .models import PlanQueryBody, ScopeSummary


@dataclass(frozen=True, slots=True)
class PublishedDocument:
    document_id: int
    knowledge_base_id: int
    generation_id: int


@dataclass(frozen=True, slots=True)
class AccessScope:
    tenant_id: int
    user_id: int
    knowledge_base_ids: tuple[int, ...]
    document_ids: tuple[int, ...]
    published_documents: tuple[PublishedDocument, ...]
    resolved_filters: tuple[ResolvedFilter, ...] = ()

    def __post_init__(self):
        if any(
            type(value) is not tuple
            for value in (
                self.knowledge_base_ids,
                self.document_ids,
                self.published_documents,
                self.resolved_filters,
            )
        ):
            raise ValueError("授权范围必须使用不可变元组")
        if any(not isinstance(item, ResolvedFilter) for item in self.resolved_filters):
            raise ValueError("过滤范围必须使用服务端解析条件")
        values = (self.tenant_id, self.user_id, *self.knowledge_base_ids, *self.document_ids)
        if any(type(value) is not int or not 0 < value <= 2**63 - 1 for value in values):
            raise ValueError("授权范围包含无效 ID")
        if len(set(self.knowledge_base_ids)) != len(self.knowledge_base_ids):
            raise ValueError("知识库范围不能重复")
        if len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("文档范围不能重复")
        seen = set()
        for document in self.published_documents:
            if (
                not isinstance(document, PublishedDocument)
                or type(document.document_id) is not int
                or type(document.knowledge_base_id) is not int
                or document.document_id not in self.document_ids
                or document.knowledge_base_id not in self.knowledge_base_ids
                or type(document.generation_id) is not int
                or not 0 < document.generation_id <= 2**63 - 1
                or document.document_id in seen
            ):
                raise ValueError("已发布文档必须属于当前授权范围及唯一活动批次")
            seen.add(document.document_id)

    def summary(self) -> ScopeSummary:
        return ScopeSummary(
            knowledge_base_ids=tuple(str(value) for value in self.knowledge_base_ids),
            authorized_document_count=len(self.document_ids),
            published_document_count=len(self.published_documents),
        )


class QueryScopeRepository:
    def __init__(self, connection: Connection, schema: str):
        self.connection = connection
        self.schema = schema

    def query(self, statement: str, parameters=()):
        return self.connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(self.schema)),
            parameters,
        )

    def resolve(self, identity: Identity, body: PlanQueryBody) -> AccessScope:
        predicate, args = knowledge_base_access_sql(identity, permission="READ")
        rows = self.query(
            "SELECT kb.id FROM {schema}.knowledge_base kb WHERE " + predicate + " ORDER BY kb.id",
            args,
        ).fetchall()
        allowed = {row["id"] for row in rows}
        selected = (
            {int(value) for value in body.knowledge_base_ids}
            if body.knowledge_base_ids
            else allowed
        )
        if not selected.issubset(allowed):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "指定资源不可用或没有读取权限")
        kb_ids = tuple(sorted(selected))
        selected_docs = (
            tuple(int(value) for value in body.document_ids) if body.document_ids else None
        )
        if not kb_ids:
            if selected_docs:
                raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "指定资源不可用或没有读取权限")
            return AccessScope(identity.tenant_id, identity.user_id, (), (), ())
        predicate, args = document_access_sql(identity, permission="READ")
        doc_filter = " AND doc.id = ANY(%s)" if selected_docs else ""
        parameters = (*args, list(kb_ids), *([list(selected_docs)] if selected_docs else []))
        rows = self.query(
            """SELECT doc.id, doc.kb_id, doc.status, doc.active_generation_id,
                      g.status AS generation_status
               FROM {schema}.document doc JOIN {schema}.knowledge_base kb
                 ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
               LEFT JOIN {schema}.index_generation g
                 ON g.tenant_id = doc.tenant_id AND g.document_id = doc.id
                 AND g.id = doc.active_generation_id
               WHERE """
            + predicate
            + " AND doc.kb_id = ANY(%s)"
            + doc_filter
            + " ORDER BY doc.id",
            parameters,
        ).fetchall()
        document_ids = tuple(row["id"] for row in rows)
        if selected_docs and not set(selected_docs).issubset(document_ids):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "指定资源不可用或没有读取权限")
        published = tuple(
            PublishedDocument(row["id"], row["kb_id"], row["active_generation_id"])
            for row in rows
            if row["status"] == "PUBLISHED"
            and row["generation_status"] == "READY"
            and row["active_generation_id"] is not None
        )
        return AccessScope(identity.tenant_id, identity.user_id, kb_ids, document_ids, published)
