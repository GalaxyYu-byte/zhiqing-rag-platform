"""独立数据库连接上的授权召回与原文复核；不复用跨线程连接。"""

from dataclasses import fields, replace

from psycopg import sql
from starlette.concurrency import run_in_threadpool

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing.embedding import validate_embedding
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.modules.identity.auth_repository import AuthRepository
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.modules.identity.auth_service import authorization_from_session
from zhiqing_rag.modules.knowledge.permissions import document_access_sql
from zhiqing_rag.query_metadata.compiler import compile_filters
from zhiqing_rag.query_routing.scope import AccessScope

from .models import ChunkRecord, RecallHit

_CHUNK_COLUMNS = """
    c.tenant_id, c.kb_id AS knowledge_base_id, c.document_id, c.revision_id,
    c.generation_id, c.id AS chunk_id, c.chunk_index, doc.title AS document_title,
    r.original_filename AS document_name, c.content, c.content_sha256,
    c.token_count, c.page_number, c.section_title
"""
_COLUMN_NAMES = tuple(field.name for field in fields(ChunkRecord))


def _check_scope(identity: AuthIdentity, scope: AccessScope) -> None:
    if (identity.tenant_id, identity.user_id) != (scope.tenant_id, scope.user_id):
        raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "检索范围不属于当前用户或租户")


def _check_limit(limit: int) -> None:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("召回数量必须是 1 到 100 的整数")


def _record(row: dict) -> ChunkRecord:
    return ChunkRecord(**{name: row[name] for name in _COLUMN_NAMES})


def _chunk_key(chunk: ChunkRecord) -> tuple[int, ...]:
    return (
        chunk.tenant_id,
        chunk.knowledge_base_id,
        chunk.document_id,
        chunk.revision_id,
        chunk.generation_id,
        chunk.chunk_id,
    )


class QueryRetrievalRepository:
    """每次召回和复核均刷新登录身份，并取初始范围与当前授权的交集。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def dense(
        self,
        identity: AuthIdentity,
        scope: AccessScope,
        vector: tuple[float, ...],
        limit: int,
    ) -> tuple[RecallHit, ...]:
        _check_scope(identity, scope)
        _check_limit(limit)
        validate_embedding(vector)
        if self.settings.embedding_dimensions != 1024:
            raise ValueError("当前索引仅支持 1024 维查询向量")
        vector_text = "[" + ",".join(str(float(value)) for value in vector) + "]"
        return await run_in_threadpool(self._dense, identity, scope, vector_text, limit)

    async def bm25(
        self,
        identity: AuthIdentity,
        scope: AccessScope,
        query: str,
        limit: int,
    ) -> tuple[RecallHit, ...]:
        _check_scope(identity, scope)
        _check_limit(limit)
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("BM25 问题必须是 1 到 2000 字符的非空文本")
        return await run_in_threadpool(self._bm25, identity, scope, query.strip(), limit)

    async def revalidate(
        self,
        identity: AuthIdentity,
        scope: AccessScope,
        chunks: tuple[ChunkRecord, ...],
    ) -> tuple[ChunkRecord, ...]:
        _check_scope(identity, scope)
        return await run_in_threadpool(self._revalidate, identity, scope, chunks)

    def _refresh_identity(self, connection, identity: AuthIdentity) -> AuthIdentity:
        repository = AuthRepository(connection, self.settings.db_schema)
        expected = (identity.user_id, identity.tenant_id, identity.member_id)
        row = authorization_from_session(
            repository,
            identity.session_key,
            expected,
            identity.auth_version,
            identity.tenant_auth_version,
        )
        return replace(
            identity,
            department_id=row["department_id"],
            clearance=row["clearance"],
            role=row["role"],
        )

    def _authorized(self, identity: AuthIdentity, scope: AccessScope) -> tuple[str, tuple]:
        predicate, arguments = document_access_sql(identity, published_only=True)
        metadata_predicate, metadata_arguments = compile_filters(scope.resolved_filters)
        # 三列同时比对，禁止分别用 ANY 产生原范围中不存在的文档/知识库/批次组合。
        initial = scope.published_documents
        statement = (
            """
            FROM {schema}.document_chunk c
            JOIN {schema}.document doc
              ON doc.tenant_id = c.tenant_id AND doc.kb_id = c.kb_id
             AND doc.id = c.document_id AND doc.active_generation_id = c.generation_id
            JOIN {schema}.knowledge_base kb
              ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
            JOIN {schema}.index_generation g
              ON g.tenant_id = c.tenant_id AND g.document_id = c.document_id
             AND g.id = c.generation_id AND g.revision_id = c.revision_id
            JOIN {schema}.document_revision r
              ON r.tenant_id = c.tenant_id AND r.document_id = c.document_id
             AND r.id = c.revision_id
            WHERE """
            + predicate
            + " AND " + metadata_predicate
            + """
              AND g.status = 'READY' AND c.tenant_id = %s
              AND (c.document_id, c.kb_id, c.generation_id) IN (
                SELECT * FROM unnest(%s::bigint[], %s::bigint[], %s::bigint[])
              )
        """
        )
        return statement, (
            *arguments,
            *metadata_arguments,
            scope.tenant_id,
            [document.document_id for document in initial],
            [document.knowledge_base_id for document in initial],
            [document.generation_id for document in initial],
        )

    def _query(self, connection, statement: str, parameters):
        return connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(self.settings.db_schema)),
            parameters,
        ).fetchall()

    @staticmethod
    def _configure_planner(connection):
        # pg_search 0.25.6 的 JOIN Custom Scan 会使复杂 ACL 窄投影规划超时。
        # 只关闭本事务的 JOIN hook，基础 ||| / pdb.score Custom Scan 仍启用。
        connection.execute("SET LOCAL paradedb.enable_join_custom_scan = off")

    def _dense(self, identity, scope, vector_text, limit):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            if not scope.published_documents:
                return ()
            self._configure_planner(connection)
            authorized, parameters = self._authorized(fresh, scope)
            # 不可把其他模型空间的文档静默删掉后继续召回；整条 dense 失败后可降级 BM25。
            incompatible = (
                "SELECT 1 AS incompatible "
                + authorized
                + """
                AND c.embedding IS NOT NULL AND NOT EXISTS (
                  SELECT 1 FROM {schema}.embedding_profile profile
                  WHERE profile.id = g.embedding_profile_id
                    AND profile.provider = %s AND profile.model_name = %s
                    AND profile.dimensions = %s AND profile.distance_metric = %s
                ) LIMIT 1
            """
            )
            if self._query(
                connection,
                incompatible,
                (
                    *parameters,
                    "dashscope",
                    self.settings.embedding_model,
                    self.settings.embedding_dimensions,
                    "cosine",
                ),
            ):
                raise ValueError("授权范围存在不兼容的向量模型配置，不能跨模型空间比较")
            # MATERIALIZED 固定先授权后算距离；精确扫描避免全局 HNSW Top-K 再过滤漏召回。
            statement = (
                "WITH authorized AS MATERIALIZED (SELECT "
                + _CHUNK_COLUMNS
                + ", c.embedding "
                + authorized
                + " AND c.embedding IS NOT NULL) "
                + "SELECT "
                + ", ".join(_COLUMN_NAMES)
                + ", 1 - (embedding <=> %s::vector(1024)) AS score FROM authorized "
                + "ORDER BY embedding <=> %s::vector(1024), chunk_id ASC LIMIT %s"
            )
            rows = self._query(
                connection,
                statement,
                (*parameters, vector_text, vector_text, limit),
            )
            return tuple(RecallHit(_record(row), float(row["score"])) for row in rows)

    def _bm25(self, identity, scope, query, limit):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            if not scope.published_documents:
                return ()
            self._configure_planner(connection)
            authorized, parameters = self._authorized(fresh, scope)
            # ||| 的 text 重载为自然语言分词匹配，不使用 @@@ / parse / pdb.query 查询语法。
            # https://www.paradedb.com/docs/reference/full-text/match
            # 0.25.6 在同一句 SQL 中同时规划复杂 ACL JOIN 与 FTS 会超时。
            # 先完整授权得到 chunk ID，评分只针对这些 ID，再刷新身份并回读原文。
            authorized_ids = self._query(
                connection,
                "SELECT c.id AS chunk_id " + authorized,
                parameters,
            )
            if not authorized_ids:
                return ()
            statement = """SELECT c.id AS chunk_id, pdb.score(c.id) AS score
                     FROM {schema}.document_chunk c
                     WHERE c.content ||| %s::text AND c.tenant_id = %s
                       AND c.id = ANY(%s::bigint[])
                     ORDER BY score DESC, c.id ASC LIMIT %s"""
            rows = self._query(
                connection,
                statement,
                (
                    query,
                    scope.tenant_id,
                    [row["chunk_id"] for row in authorized_ids],
                    limit,
                ),
            )
            if not rows:
                return ()
            fresh = self._refresh_identity(connection, identity)
            authorized, parameters = self._authorized(fresh, scope)
            reread = self._query(
                connection,
                "SELECT " + _CHUNK_COLUMNS + authorized + " AND c.id = ANY(%s::bigint[])",
                (*parameters, [row["chunk_id"] for row in rows]),
            )
            records = {record.chunk_id: record for record in map(_record, reread)}
            return tuple(
                RecallHit(records[row["chunk_id"]], float(row["score"]))
                for row in rows
                if row["chunk_id"] in records
            )

    def _revalidate(self, identity, scope, chunks):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            allowed = {
                (document.document_id, document.knowledge_base_id, document.generation_id)
                for document in scope.published_documents
            }
            candidates = {
                _chunk_key(chunk): chunk
                for chunk in chunks
                if chunk.tenant_id == scope.tenant_id
                and (chunk.document_id, chunk.knowledge_base_id, chunk.generation_id) in allowed
            }
            if not candidates:
                return ()
            self._configure_planner(connection)
            authorized, parameters = self._authorized(fresh, scope)
            statement = (
                "SELECT "
                + _CHUNK_COLUMNS
                + authorized
                + """
                AND (c.id, c.document_id, c.kb_id, c.revision_id, c.generation_id) IN (
                  SELECT * FROM unnest(
                    %s::bigint[], %s::bigint[], %s::bigint[], %s::bigint[], %s::bigint[]
                  )
                )
            """
            )
            keys = tuple(candidates)
            # _chunk_key 的末项是 chunk_id；五列按 SQL 的顺序显式绑定。
            arrays = tuple([key[index] for key in keys] for index in (5, 2, 1, 3, 4))
            rows = self._query(connection, statement, (*parameters, *arrays))
            reread = {_chunk_key(record): record for record in map(_record, rows)}
            return tuple(reread[key] for key in keys if key in reread)
