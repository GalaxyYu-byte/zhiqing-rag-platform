"""短事务聊天存储；幂等、租约和历史授权均由服务端执行。"""

import hashlib
import json
import re
from dataclasses import replace
from functools import wraps

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.modules.identity.auth_repository import AuthRepository
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.modules.identity.auth_service import authorization_from_session
from zhiqing_rag.modules.knowledge.permissions import document_access_sql
from zhiqing_rag.query_answering.models import QueryAnswerResponse
from zhiqing_rag.query_structured.models import StructuredProvenance
from zhiqing_rag.query_structured.repository import revalidate_provenance

_PUBLIC_RUN_COLUMNS = (
    "id",
    "session_id",
    "client_request_id",
    "status",
    "question",
    "standalone_question",
    "created_at",
    "started_at",
    "finished_at",
    "error_code",
)
_SOURCE_COLUMNS = (
    "knowledge_base_id",
    "document_id",
    "revision_id",
    "generation_id",
    "chunk_id",
    "content_sha256",
)
_TERMINAL_FAILURES = {"FAILED", "CANCELLED", "ABANDONED"}


def _db_errors(method):
    @wraps(method)
    async def call(*args, **kwargs):
        try:
            return await method(*args, **kwargs)
        except psycopg.Error:
            raise AuthError(503, "CHAT_DATABASE_UNAVAILABLE", "会话存储暂时不可用") from None

    return call


def _public_run(row: dict) -> dict:
    result = {column: row.get(column) for column in _PUBLIC_RUN_COLUMNS}
    result["id"] = str(row["id"])
    result["session_id"] = str(row["session_id"])
    return result


def _public_session(row: dict) -> dict:
    return {
        **{
            column: row[column]
            for column in (
                "title",
                "status",
                "created_at",
                "last_active_at",
            )
        },
        "id": str(row["id"]),
    }


def _limit(value: int, maximum: int = 100) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"数量必须是 1 到 {maximum} 的整数")


def _id(value: int) -> None:
    if type(value) is not int or not 0 < value <= 2**63 - 1:
        raise ValueError("ID 必须为正 bigint")


def _not_found() -> AuthError:
    return AuthError(404, "CHAT_NOT_FOUND", "会话或轮次不存在")


def _lease_lost() -> AuthError:
    return AuthError(409, "CHAT_LEASE_LOST", "执行租约已失效，请刷新会话")


def _source_key(source: dict) -> tuple:
    return tuple(source[column] for column in _SOURCE_COLUMNS)


def _source(evidence: dict) -> dict:
    result = {column: int(evidence[column]) for column in _SOURCE_COLUMNS[:-1]}
    for value in result.values():
        _id(value)
    result["content_sha256"] = hashlib.sha256(evidence["content"].encode("utf-8")).hexdigest()
    return result


def _sources(snapshot: dict) -> tuple[dict, ...] | None:
    """快照版本或来源结构异常时 fail closed，绝不依赖可被删除的引用映射。"""
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        return None
    try:
        response = QueryAnswerResponse.model_validate_json(json.dumps(snapshot["response"]))
        sources = {
            _source_key(value): value
            for value in (_source(item.model_dump()) for item in response.retrieval.evidence)
        }
        history = snapshot.get("history_sources", [])
        if not isinstance(history, list):
            return None
        for value in history:
            if not isinstance(value, dict) or set(value) != set(_SOURCE_COLUMNS):
                return None
            normalized = {column: int(value[column]) for column in _SOURCE_COLUMNS[:-1]}
            for identifier in normalized.values():
                _id(identifier)
            digest = value["content_sha256"]
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                return None
            normalized["content_sha256"] = digest
            sources[_source_key(normalized)] = normalized
        return tuple(sources.values())
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


def _structured_sources(snapshot: dict) -> tuple[StructuredProvenance, ...] | None:
    """统计来源独立于原文 chunk；零条统计也必须保留授权范围锚点。"""
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        return None
    try:
        response = QueryAnswerResponse.model_validate_json(json.dumps(snapshot["response"]))
        structured = getattr(response, "structured", None)
        provenance_data = snapshot.get("structured_provenance")
        sources = []
        if structured is not None:
            if not isinstance(provenance_data, dict):
                return None
            provenance = StructuredProvenance.model_validate_json(json.dumps(provenance_data))
            if (
                structured.total != len(provenance.sources)
                or structured.operation != provenance.operation
                or structured.applied_filters != provenance.applied_filters
                or structured.executed_at != provenance.executed_at
                or provenance.knowledge_base_ids != response.scope.knowledge_base_ids
            ):
                return None
            if structured.operation == "list_documents":
                names = ("knowledge_base_id", "document_id", "revision_id", "generation_id")
                page = provenance.sources[structured.offset:structured.offset + structured.limit]
                actual = tuple(
                    tuple(getattr(doc, name) for name in names) for doc in structured.documents
                )
                expected = tuple(tuple(getattr(source, name) for name in names) for source in page)
                if actual != expected:
                    return None
            sources.append(provenance)
        elif provenance_data is not None:
            return None
        history = snapshot.get("history_structured_provenance", [])
        if not isinstance(history, list):
            return None
        sources.extend(
            StructuredProvenance.model_validate_json(json.dumps(value)) for value in history
        )
        return tuple(sources)
    except (AttributeError, KeyError, TypeError, ValueError, ValidationError):
        return None


class ChatRepository:
    """每个方法借用独立连接并提交短事务，不持有跨模型调用的连接。"""

    def __init__(self, settings: Settings):
        self.settings = settings

    @_db_errors
    async def create_session(self, identity: AuthIdentity, title: str = "新会话") -> dict:
        if not isinstance(title, str) or not title.strip() or len(title.strip()) > 300:
            raise ValueError("会话标题必须为 1 到 300 个字符")
        return await run_in_threadpool(self._create_session, identity, title.strip())

    @_db_errors
    async def list_sessions(
        self,
        identity: AuthIdentity,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        _limit(limit)
        if type(offset) is not int or offset < 0:
            raise ValueError("分页偏移必须为非负整数")
        return await run_in_threadpool(self._list_sessions, identity, limit, offset)

    @_db_errors
    async def list_runs(
        self,
        identity: AuthIdentity,
        session_id: int,
        before_run_id: int | None = None,
        limit: int = 20,
    ) -> dict:
        _id(session_id)
        _limit(limit)
        if before_run_id is not None:
            _id(before_run_id)
        return await run_in_threadpool(
            self._list_runs,
            identity,
            session_id,
            before_run_id,
            limit,
        )

    @_db_errors
    async def structured_history_in_scope(self, identity, session_id, run_id, scope) -> bool:
        _id(session_id)
        _id(run_id)
        if (identity.tenant_id, identity.user_id) != (scope.tenant_id, scope.user_id):
            return False
        return await run_in_threadpool(
            self._structured_history_in_scope, identity, session_id, run_id, scope,
        )

    @_db_errors
    async def get_run(self, identity: AuthIdentity, session_id: int, run_id: int) -> dict:
        _id(session_id)
        _id(run_id)
        return await run_in_threadpool(self._get_run, identity, session_id, run_id)

    @_db_errors
    async def start_run(
        self,
        identity: AuthIdentity,
        session_id: int,
        request: dict,
        request_sha256: str,
        lease_owner: str,
        lease_seconds: int,
    ) -> dict:
        _id(session_id)
        _limit(lease_seconds, 3600)
        if not isinstance(lease_owner, str) or not 1 <= len(lease_owner) <= 128:
            raise ValueError("执行进程标识无效")
        if re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
            raise ValueError("请求摘要无效")
        if not isinstance(request.get("question"), str) or not request["question"].strip():
            raise ValueError("问题不能为空")
        request_id = request.get("client_request_id")
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 128:
            raise ValueError("客户端请求标识无效")
        return await run_in_threadpool(
            self._start_run,
            identity,
            session_id,
            request,
            request_sha256,
            lease_owner,
            lease_seconds,
        )

    @_db_errors
    async def heartbeat(
        self,
        identity: AuthIdentity,
        session_id: int,
        run_id: int,
        *,
        lease_owner: str,
        lease_epoch: int,
        lease_seconds: int,
    ) -> None:
        _limit(lease_seconds, 3600)
        await run_in_threadpool(
            self._heartbeat,
            identity,
            session_id,
            run_id,
            lease_owner,
            lease_epoch,
            lease_seconds,
        )

    @_db_errors
    async def complete(
        self,
        identity: AuthIdentity,
        session_id: int,
        run_id: int,
        *,
        lease_owner: str,
        lease_epoch: int,
        response: dict,
        standalone_question: str | None,
        history_run_ids: list[int] | None = None,
        structured_provenance: dict | None = None,
    ) -> dict:
        response = QueryAnswerResponse.model_validate_json(json.dumps(response)).model_dump(
            mode="json",
        )
        history_run_ids = list(dict.fromkeys(history_run_ids or []))
        for identifier in history_run_ids:
            _id(identifier)
        if run_id in history_run_ids or len(history_run_ids) > 100:
            raise ValueError("历史上下文轮次无效")
        return await run_in_threadpool(
            self._complete,
            identity,
            session_id,
            run_id,
            lease_owner,
            lease_epoch,
            response,
            standalone_question,
            history_run_ids,
            structured_provenance,
        )

    @_db_errors
    async def terminate(
        self,
        identity: AuthIdentity,
        session_id: int,
        run_id: int,
        *,
        lease_owner: str,
        lease_epoch: int,
        status: str,
        error_code: str,
    ) -> None:
        if status not in _TERMINAL_FAILURES:
            raise ValueError("无效的失败终态")
        if not isinstance(error_code, str) or not 1 <= len(error_code) <= 100:
            raise ValueError("无效的失败错误码")
        await run_in_threadpool(
            self._terminate,
            identity,
            session_id,
            run_id,
            lease_owner,
            lease_epoch,
            status,
            error_code,
        )

    def _refresh_identity(self, connection, identity: AuthIdentity) -> AuthIdentity:
        row = authorization_from_session(
            AuthRepository(connection, self.settings.db_schema),
            identity.session_key,
            (identity.user_id, identity.tenant_id, identity.member_id),
            identity.auth_version,
            identity.tenant_auth_version,
        )
        return replace(
            identity,
            department_id=row["department_id"],
            clearance=row["clearance"],
            role=row["role"],
        )

    def _execute(self, connection, statement: str, parameters: tuple = ()):
        return connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(self.settings.db_schema)),
            parameters,
        )

    def _session(self, connection, identity, session_id, *, lock: bool = False) -> dict:
        row = self._execute(
            connection,
            """SELECT id, title, status, created_at, last_active_at
               FROM {schema}.chat_session
               WHERE tenant_id = %s AND owner_member_id = %s AND id = %s
                 AND status = 'ACTIVE' AND deleted_at IS NULL"""
            + (" FOR UPDATE" if lock else ""),
            (identity.tenant_id, identity.member_id, session_id),
        ).fetchone()
        if row is None:
            raise _not_found()
        return row

    def _create_session(self, identity, title):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            row = self._execute(
                connection,
                """INSERT INTO {schema}.chat_session (tenant_id, owner_member_id, title)
                   VALUES (%s, %s, %s)
                   RETURNING id, title, status, created_at, last_active_at""",
                (fresh.tenant_id, fresh.member_id, title),
            ).fetchone()
            return _public_session(row)

    def _list_sessions(self, identity, limit, offset):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            rows = self._execute(
                connection,
                """SELECT id, title, status, created_at, last_active_at
                   FROM {schema}.chat_session
                   WHERE tenant_id = %s AND owner_member_id = %s
                     AND status = 'ACTIVE' AND deleted_at IS NULL
                   ORDER BY last_active_at DESC, id DESC LIMIT %s OFFSET %s""",
                (fresh.tenant_id, fresh.member_id, limit, offset),
            ).fetchall()
            return [_public_session(row) for row in rows]

    def _authorized_sources(self, connection, identity, sources: tuple[dict, ...]) -> set[tuple]:
        if not sources:
            return set()
        predicate, parameters = document_access_sql(identity, published_only=True)
        # 多列一同匹配，避免 ANY 各列自由组合形成不属于原证据的来源。
        rows = self._execute(
            connection,
            """SELECT c.kb_id AS knowledge_base_id, c.document_id, c.revision_id,
                      c.generation_id, c.id AS chunk_id, c.content_sha256, c.content
               FROM {schema}.document_chunk c
               JOIN {schema}.document doc ON doc.tenant_id = c.tenant_id
                 AND doc.kb_id = c.kb_id AND doc.id = c.document_id
                 AND doc.active_generation_id = c.generation_id
               JOIN {schema}.knowledge_base kb ON kb.tenant_id = doc.tenant_id
                 AND kb.id = doc.kb_id
               JOIN {schema}.index_generation g ON g.tenant_id = c.tenant_id
                 AND g.document_id = c.document_id AND g.id = c.generation_id
                 AND g.revision_id = c.revision_id
               JOIN {schema}.document_revision r ON r.tenant_id = c.tenant_id
                 AND r.document_id = c.document_id AND r.id = c.revision_id
               WHERE """
            + predicate
            + """
                 AND g.status = 'READY' AND c.tenant_id = %s
                 AND (c.kb_id, c.document_id, c.revision_id, c.generation_id,
                      c.id, c.content_sha256) IN (
                   SELECT * FROM unnest(%s::bigint[], %s::bigint[], %s::bigint[],
                     %s::bigint[], %s::bigint[], %s::text[]))
               FOR SHARE OF doc, kb, g, c, r""",
            (
                *parameters,
                identity.tenant_id,
                *([source[column] for source in sources] for column in _SOURCE_COLUMNS),
            ),
        ).fetchall()
        return {
            _source_key(row)
            for row in rows
            if hashlib.sha256(row["content"].encode("utf-8")).hexdigest() == row["content_sha256"]
        }

    def _visible_runs(self, connection, identity, rows: list[dict]) -> list[dict]:
        run_sources = {}
        run_structured_sources = {}
        all_sources = {}
        for row in rows:
            sources = _sources(row["execution_snapshot"]) if row["status"] == "SUCCEEDED" else ()
            run_sources[row["id"]] = sources
            run_structured_sources[row["id"]] = (
                _structured_sources(row["execution_snapshot"])
                if row["status"] == "SUCCEEDED" else ()
            )
            for source in sources or ():
                all_sources[_source_key(source)] = source
        permitted = self._authorized_sources(connection, identity, tuple(all_sources.values()))
        results = []
        for row in rows:
            sources = run_sources[row["id"]]
            structured_sources = run_structured_sources[row["id"]]
            available = sources is not None and all(
                _source_key(source) in permitted for source in sources
            )
            available = available and structured_sources is not None and all(
                revalidate_provenance(connection, self.settings.db_schema, identity, provenance)
                for provenance in structured_sources
            )
            result = _public_run(row)
            result["visibility"] = (
                "pending"
                if row["status"] in {"PENDING", "RUNNING"}
                else "visible"
                if available
                else "unavailable"
            )
            result["response"] = (
                row["execution_snapshot"]["response"]
                if available and row["status"] == "SUCCEEDED"
                else None
            )
            if not available or row["status"] != "SUCCEEDED":
                result["standalone_question"] = None
            results.append(result)
        return results

    def _reap(self, connection, identity, session_id, lease_seconds):
        self._execute(
            connection,
            """UPDATE {schema}.chat_run
               SET status = 'ABANDONED', error_code = 'CHAT_RUN_EXPIRED',
                   finished_at = clock_timestamp(), updated_at = clock_timestamp(),
                   lease_owner = NULL, lease_expires_at = NULL
               WHERE tenant_id = %s AND session_id = %s
                 AND ((status = 'RUNNING' AND lease_expires_at <= clock_timestamp())
                   OR (status = 'PENDING' AND created_at <= clock_timestamp()
                       - make_interval(secs => %s)))""",
            (identity.tenant_id, session_id, lease_seconds),
        )

    def _list_runs(self, identity, session_id, before_run_id, limit):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            self._session(connection, fresh, session_id, lock=True)
            self._reap(connection, fresh, session_id, self.settings.chat_run_lease_seconds)
            rows = self._execute(
                connection,
                """SELECT * FROM {schema}.chat_run
                   WHERE tenant_id = %s AND session_id = %s"""
                + (" AND id < %s" if before_run_id is not None else "")
                + " ORDER BY id DESC LIMIT %s",
                (
                    fresh.tenant_id,
                    session_id,
                    *((before_run_id,) if before_run_id is not None else ()),
                    limit + 1,
                ),
            ).fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            return {
                "runs": self._visible_runs(connection, fresh, list(reversed(rows))),
                "has_more": has_more,
                "next_before_run_id": str(rows[-1]["id"]) if rows and has_more else None,
            }

    def _get_run(self, identity, session_id, run_id):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            self._session(connection, fresh, session_id, lock=True)
            self._reap(connection, fresh, session_id, self.settings.chat_run_lease_seconds)
            row = self._execute(
                connection,
                """SELECT * FROM {schema}.chat_run
                   WHERE tenant_id = %s AND session_id = %s AND id = %s""",
                (fresh.tenant_id, session_id, run_id),
            ).fetchone()
            if row is None:
                raise _not_found()
            return self._visible_runs(connection, fresh, [row])[0]

    def _structured_history_in_scope(self, identity, session_id, run_id, scope):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            self._session(connection, fresh, session_id, lock=True)
            row = self._execute(
                connection,
                """SELECT * FROM {schema}.chat_run
                   WHERE tenant_id = %s AND session_id = %s AND id = %s""",
                (fresh.tenant_id, session_id, run_id),
            ).fetchone()
            if row is None or row["status"] != "SUCCEEDED":
                return False
            provenance = _structured_sources(row["execution_snapshot"])
            if provenance is None:
                return False
            selected_bases = set(map(str, scope.knowledge_base_ids))
            selected_ids = set(map(str, scope.document_ids))
            selected_documents = {
                (str(doc.document_id), str(doc.knowledge_base_id), str(doc.generation_id))
                for doc in scope.published_documents
            }
            for item in provenance:
                if (
                    not set(item.knowledge_base_ids).issubset(selected_bases)
                    or not set(item.selected_document_ids).issubset(selected_ids)
                ) or any(
                    (source.document_id, source.knowledge_base_id, source.generation_id)
                    not in selected_documents for source in item.sources
                ):
                    return False
                if not revalidate_provenance(connection, self.settings.db_schema, fresh, item):
                    return False
            return True

    def _message(self, connection, identity, session_id, run_id, role, content, token_count=0):
        self._execute(
            connection,
            """INSERT INTO {schema}.chat_message
                 (tenant_id, session_id, run_id, role, content, sequence_no, token_count)
               SELECT %s, %s, %s, %s, %s, COALESCE(MAX(sequence_no), 0) + 1, %s
               FROM {schema}.chat_message WHERE tenant_id = %s AND session_id = %s""",
            (
                identity.tenant_id,
                session_id,
                run_id,
                role,
                content,
                token_count,
                identity.tenant_id,
                session_id,
            ),
        )

    def _start_run(self, identity, session_id, request, digest, owner, lease_seconds):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            self._session(connection, fresh, session_id, lock=True)
            self._reap(connection, fresh, session_id, lease_seconds)
            row = self._execute(
                connection,
                """SELECT * FROM {schema}.chat_run
                   WHERE tenant_id = %s AND session_id = %s AND client_request_id = %s""",
                (fresh.tenant_id, session_id, request["client_request_id"]),
            ).fetchone()
            if row is not None:
                if row["request_sha256"] != digest:
                    raise AuthError(409, "CHAT_IDEMPOTENCY_CONFLICT", "请求标识已用于不同请求")
                result = self._visible_runs(connection, fresh, [row])[0]
                snapshot = dict(row["execution_snapshot"])
                if result["visibility"] == "unavailable":
                    snapshot = {"schema_version": 1, "request": snapshot.get("request", {})}
                return {
                    **result,
                    "execution_snapshot": snapshot,
                    "request_sha256": digest,
                    "lease_epoch": row["lease_epoch"],
                    "lease_owner": row["lease_owner"],
                    "is_new": False,
                }
            active = self._execute(
                connection,
                """SELECT id FROM {schema}.chat_run
                   WHERE tenant_id = %s AND session_id = %s
                     AND status IN ('PENDING', 'RUNNING') LIMIT 1""",
                (fresh.tenant_id, session_id),
            ).fetchone()
            if active is not None:
                raise AuthError(409, "CHAT_RUN_ACTIVE", "当前会话已有正在执行的提问")
            snapshot = {"schema_version": 1, "request": request}
            row = self._execute(
                connection,
                """INSERT INTO {schema}.chat_run
                     (tenant_id, session_id, client_request_id, request_sha256, question,
                      status, execution_snapshot, lease_owner, lease_epoch,
                      lease_expires_at, started_at)
                   VALUES (%s, %s, %s, %s, %s, 'RUNNING', %s, %s, 1,
                           clock_timestamp() + make_interval(secs => %s), clock_timestamp())
                   RETURNING *""",
                (
                    fresh.tenant_id,
                    session_id,
                    request["client_request_id"],
                    digest,
                    request["question"],
                    Jsonb(snapshot),
                    owner,
                    lease_seconds,
                ),
            ).fetchone()
            self._message(connection, fresh, session_id, row["id"], "USER", request["question"])
            self._touch(connection, fresh, session_id, title=request["question"][:60])
            result = self._visible_runs(connection, fresh, [row])[0]
            return {
                **result,
                "execution_snapshot": snapshot,
                "request_sha256": digest,
                "lease_epoch": row["lease_epoch"],
                "lease_owner": owner,
                "is_new": True,
            }

    def _owned_run(self, connection, identity, session_id, run_id, owner, epoch):
        row = self._execute(
            connection,
            """SELECT * FROM {schema}.chat_run
               WHERE tenant_id = %s AND session_id = %s AND id = %s
                 AND status = 'RUNNING' AND lease_owner = %s AND lease_epoch = %s
                 AND lease_expires_at > clock_timestamp() FOR UPDATE""",
            (identity.tenant_id, session_id, run_id, owner, epoch),
        ).fetchone()
        if row is None:
            raise _lease_lost()
        return row

    def _touch(self, connection, identity, session_id, *, title=None):
        self._execute(
            connection,
            """UPDATE {schema}.chat_session SET last_active_at = clock_timestamp(),
                   updated_at = clock_timestamp()"""
            + (", title = CASE WHEN title = '新会话' THEN %s ELSE title END" if title else "")
            + " WHERE tenant_id = %s AND id = %s",
            (*((title,) if title else ()), identity.tenant_id, session_id),
        )

    def _heartbeat(self, identity, session_id, run_id, owner, epoch, lease_seconds):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            self._session(connection, fresh, session_id)
            row = self._execute(
                connection,
                """UPDATE {schema}.chat_run SET lease_expires_at = clock_timestamp()
                       + make_interval(secs => %s), updated_at = clock_timestamp()
                   WHERE tenant_id = %s AND session_id = %s AND id = %s
                     AND status = 'RUNNING' AND lease_owner = %s AND lease_epoch = %s
                     AND lease_expires_at > clock_timestamp() RETURNING id""",
                (lease_seconds, fresh.tenant_id, session_id, run_id, owner, epoch),
            ).fetchone()
            if row is None:
                raise _lease_lost()

    def _complete(
        self,
        identity,
        session_id,
        run_id,
        owner,
        epoch,
        response,
        standalone_question,
        history_run_ids,
        structured_provenance=None,
    ):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            self._session(connection, fresh, session_id, lock=True)
            current = self._owned_run(connection, fresh, session_id, run_id, owner, epoch)
            history_sources = {}
            history_structured = {}
            if history_run_ids:
                history = self._execute(
                    connection,
                    """SELECT * FROM {schema}.chat_run
                       WHERE tenant_id = %s AND session_id = %s AND id = ANY(%s::bigint[])
                         AND status = 'SUCCEEDED'""",
                    (fresh.tenant_id, session_id, history_run_ids),
                ).fetchall()
                if len(history) != len(history_run_ids):
                    raise AuthError(403, "CHAT_CONTEXT_UNAVAILABLE", "历史上下文已不可用")
                for row in history:
                    sources = _sources(row["execution_snapshot"])
                    if sources is None:
                        raise AuthError(403, "CHAT_CONTEXT_UNAVAILABLE", "历史上下文已不可用")
                    history_sources.update((_source_key(source), source) for source in sources)
                    structured_sources = _structured_sources(row["execution_snapshot"])
                    if structured_sources is None:
                        raise AuthError(403, "CHAT_CONTEXT_UNAVAILABLE", "历史统计来源已不可用")
                    for provenance in structured_sources:
                        data = provenance.model_dump(mode="json")
                        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
                        history_structured[key] = data
            snapshot = {
                "schema_version": 1,
                "request": current["execution_snapshot"]["request"],
                "response": response,
                "history_run_ids": history_run_ids,
                "history_sources": list(history_sources.values()),
                "structured_provenance": structured_provenance,
                "history_structured_provenance": list(history_structured.values()),
            }
            sources = _sources(snapshot)
            if sources is None:
                raise ValueError("回答快照无效")
            permitted = self._authorized_sources(connection, fresh, sources)
            if any(_source_key(source) not in permitted for source in sources):
                raise AuthError(403, "CHAT_SOURCE_UNAVAILABLE", "回答来源授权或发布版本已变化")
            structured_sources = _structured_sources(snapshot)
            if structured_sources is None:
                raise ValueError("统计回答缺少有效的授权来源快照")
            if any(
                not revalidate_provenance(
                    connection, self.settings.db_schema, fresh, provenance,
                )
                for provenance in structured_sources
            ):
                raise AuthError(403, "CHAT_SOURCE_UNAVAILABLE", "统计来源授权或元数据已变化")
            citations = response["retrieval"]["evidence"]
            seen = set()
            for evidence in citations:
                number = int(evidence["citation_id"][1:])
                if number in seen:
                    raise ValueError("证据引用序号不能重复")
                seen.add(number)
                source = _source(evidence)
                self._execute(
                    connection,
                    """INSERT INTO {schema}.answer_citation
                         (tenant_id, session_id, run_id, citation_no, document_id, revision_id,
                          source_generation_id, source_chunk_id, content_sha256,
                          page_number, section_title)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        fresh.tenant_id,
                        session_id,
                        run_id,
                        number,
                        source["document_id"],
                        source["revision_id"],
                        source["generation_id"],
                        source["chunk_id"],
                        source["content_sha256"],
                        evidence["page_number"],
                        evidence["section_title"],
                    ),
                )
            answer = response["answer"]
            trace = answer["trace"]
            self._message(
                connection,
                fresh,
                session_id,
                run_id,
                "ASSISTANT",
                json.dumps(response, ensure_ascii=False, separators=(",", ":")),
                trace["output_tokens"],
            )
            row = self._execute(
                connection,
                """UPDATE {schema}.chat_run SET status = 'SUCCEEDED', execution_snapshot = %s,
                       standalone_question = %s, model_name = %s, input_tokens = %s,
                       output_tokens = %s, finished_at = clock_timestamp(),
                       updated_at = clock_timestamp(), lease_owner = NULL, lease_expires_at = NULL
                   WHERE tenant_id = %s AND session_id = %s AND id = %s
                     AND status = 'RUNNING' AND lease_owner = %s AND lease_epoch = %s
                     AND lease_expires_at > clock_timestamp() RETURNING *""",
                (
                    Jsonb(snapshot),
                    standalone_question,
                    answer["model"],
                    trace["input_tokens"],
                    trace["output_tokens"],
                    fresh.tenant_id,
                    session_id,
                    run_id,
                    owner,
                    epoch,
                ),
            ).fetchone()
            if row is None:
                raise _lease_lost()
            self._touch(connection, fresh, session_id)
            # 本事务已复核并锁定证据来源；无需再次查询。
            return {**_public_run(row), "response": response, "visibility": "visible"}

    def _terminate(self, identity, session_id, run_id, owner, epoch, status, error_code):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            self._session(connection, fresh, session_id, lock=True)
            row = self._execute(
                connection,
                """UPDATE {schema}.chat_run SET status = %s, error_code = %s,
                       finished_at = clock_timestamp(), updated_at = clock_timestamp(),
                       lease_owner = NULL, lease_expires_at = NULL
                   WHERE tenant_id = %s AND session_id = %s AND id = %s
                     AND status = 'RUNNING' AND lease_owner = %s AND lease_epoch = %s
                     AND lease_expires_at > clock_timestamp() RETURNING id""",
                (status, error_code, fresh.tenant_id, session_id, run_id, owner, epoch),
            ).fetchone()
            if row is None:
                raise _lease_lost()
            self._touch(connection, fresh, session_id)
