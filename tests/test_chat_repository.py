"""聊天短事务、重试和授权恢复；假连接不触碰真实数据库。"""

import hashlib
import threading
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import UUID

import psycopg
import pytest

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.chat import repository as repository_module
from zhiqing_rag.modules.chat.repository import ChatRepository, _source, _source_key
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity, AuthTenant, AuthUser

NOW = datetime.now(UTC)
REQUEST = {
    "question": "工单如何关闭？",
    "client_request_id": "request-1",
    "knowledge_base_ids": None,
    "document_ids": None,
}
DIGEST = "a" * 64


def identity(**changes):
    return AuthIdentity(
        **{
            "user_id": 2,
            "tenant_id": 1,
            "member_id": 10,
            "department_id": 7,
            "clearance": 2,
            "role": "MEMBER",
            "auth_version": 1,
            "tenant_auth_version": 2,
            "session_key": UUID(int=1),
            "user": AuthUser(id="2", email="test@example.test", display_name="用户"),
            "tenant": AuthTenant(id="1", code="test", name="测试", role="MEMBER"),
            **changes,
        }
    )


def response(*, with_evidence=True, chunk_id="12"):
    evidence = {
        "citation_id": "E1",
        "rank": 1,
        "chunk_id": chunk_id,
        "knowledge_base_id": "3",
        "document_id": "4",
        "revision_id": "6",
        "generation_id": "5",
        "document_title": "工单管理",
        "document_name": "工单.pdf",
        "chunk_index": 0,
        "content": "  客户确认后关闭。\n",
        "token_count": 8,
        "page_number": 1,
        "section_title": "关闭",
        "scores": {"fusion_score": 0.1},
    }
    return {
        "analysis": {
            "status": "degraded",
            "original_query": REQUEST["question"],
            "analysis": None,
            "fallback_hint": "basic_retrieval",
            "message": "分析已关闭",
            "degradation_reason": "disabled",
            "model": "disabled",
            "latency_ms": 0,
        },
        "plan": {
            "route": "hybrid",
            "original_query": REQUEST["question"],
            "query": REQUEST["question"],
            "requested_route": "hybrid",
            "reason_code": "TEST",
        },
        "scope": {
            "knowledge_base_ids": ["3"],
            "authorized_document_count": 1,
            "published_document_count": 1,
        },
        "retrieval": {
            "status": "ok" if with_evidence else "empty",
            "evidence": [evidence] if with_evidence else [],
        },
        "answer": {
            "status": "answered" if with_evidence else "insufficient",
            "statements": [
                {
                    "text": "客户确认后可关闭。",
                    "citations": [{"citation_id": "E1", "quote": "客户确认后关闭"}],
                }
            ]
            if with_evidence
            else [],
            "review_status": "passed" if with_evidence else "skipped",
            "model": "unit",
            "trace": {"input_tokens": 32, "output_tokens": 18},
        },
    }


def run_row(identifier=1, *, status="RUNNING", request=None, result=None, **changes):
    request = request or REQUEST
    snapshot = {"schema_version": 1, "request": deepcopy(request)}
    if result is not None:
        snapshot["response"] = result
    return {
        "id": identifier,
        "tenant_id": 1,
        "session_id": 20,
        "client_request_id": request["client_request_id"],
        "request_sha256": DIGEST,
        "status": status,
        "question": request["question"],
        "standalone_question": None,
        "execution_snapshot": snapshot,
        "lease_owner": "worker" if status == "RUNNING" else None,
        "lease_epoch": 1,
        "lease_expires_at": NOW + timedelta(minutes=3),
        "created_at": NOW,
        "started_at": NOW,
        "finished_at": None,
        "error_code": None,
        **changes,
    }


class Cursor:
    def __init__(self, rows=()):
        self.rows = deepcopy(list(rows))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self):
        self.runs = {}
        self.messages = []
        self.citations = []
        self.calls = []
        self.events = []
        self.thread_ids = []
        self.allowed_sources = [_source(response()["retrieval"]["evidence"][0])]
        self.expire_during_complete = False
        self.error = None
        self.session = {
            "id": 20,
            "title": "会话",
            "status": "ACTIVE",
            "tenant_id": 1,
            "owner_member_id": 10,
            "created_at": NOW,
            "last_active_at": NOW,
        }
        self.auth = {
            "user_id": 2,
            "tenant_id": 1,
            "member_id": 10,
            "department_id": 99,
            "clearance": 0,
            "role": "MEMBER",
            "auth_version": 1,
            "tenant_auth_version": 2,
        }

    def execute(self, statement, parameters=()):
        rendered = statement if isinstance(statement, str) else statement.as_string(None)
        text = " ".join(rendered.split())
        self.calls.append((text, parameters))
        self.thread_ids.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        if "AS session_user_id" in text:
            return Cursor(
                [
                    {
                        **self.auth,
                        "session_user_id": 2,
                        "session_tenant_id": 1,
                        "tenant_member_id": 10,
                        "revoked_at": None,
                        "expires_at": NOW + timedelta(hours=1),
                        "authorization_active": True,
                    }
                ]
            )
        if 'INSERT INTO "zhiqing_rag".chat_session' in text:
            self.session = {
                **self.session,
                "tenant_id": parameters[0],
                "owner_member_id": parameters[1],
                "title": parameters[2],
            }
            return Cursor([self.session])
        if 'FROM "zhiqing_rag".chat_session' in text:
            allowed = parameters[:2] == (self.session["tenant_id"], self.session["owner_member_id"])
            allowed = allowed and self.session["status"] == "ACTIVE"
            if "AND id = %s" in text:
                allowed = allowed and parameters[2] == self.session["id"]
            return Cursor([self.session] if allowed else [])
        if "SELECT c.kb_id AS knowledge_base_id" in text:
            return Cursor(
                {"content": "  客户确认后关闭。\n", **source} for source in self.allowed_sources
            )
        if 'UPDATE "zhiqing_rag".chat_session' in text:
            return Cursor()
        if 'UPDATE "zhiqing_rag".chat_run' in text and "CHAT_RUN_EXPIRED" in text:
            seconds = parameters[-1]
            for row in self.runs.values():
                expired = row["status"] == "RUNNING" and row["lease_expires_at"] <= NOW
                pending = row["status"] == "PENDING" and row["created_at"] <= (
                    NOW - timedelta(seconds=seconds)
                )
                if expired or pending:
                    row.update(
                        status="ABANDONED",
                        finished_at=NOW,
                        lease_owner=None,
                        lease_expires_at=None,
                        error_code="CHAT_RUN_EXPIRED",
                    )
            return Cursor()
        if 'INSERT INTO "zhiqing_rag".chat_run' in text:
            identifier = max(self.runs, default=0) + 1
            request = parameters[5].obj["request"]
            row = run_row(
                identifier, request=request, lease_owner=parameters[6], request_sha256=parameters[3]
            )
            self.runs[identifier] = row
            return Cursor([row])
        if 'SELECT * FROM "zhiqing_rag".chat_run' in text:
            rows = [
                row
                for row in self.runs.values()
                if (row["tenant_id"], row["session_id"]) == parameters[:2]
            ]
            if "client_request_id = %s" in text:
                rows = [row for row in rows if row["client_request_id"] == parameters[2]]
            elif "id = ANY" in text:
                rows = [
                    row
                    for row in rows
                    if row["id"] in parameters[2] and row["status"] == "SUCCEEDED"
                ]
            elif "AND id = %s" in text:
                rows = [row for row in rows if row["id"] == parameters[2]]
                if "lease_owner = %s" in text:
                    rows = [row for row in rows if self._owns(row, parameters[3], parameters[4])]
            if "ORDER BY id DESC" in text:
                if "id < %s" in text:
                    rows = [row for row in rows if row["id"] < parameters[2]]
                rows = sorted(rows, key=lambda row: row["id"], reverse=True)[: parameters[-1]]
            return Cursor(rows)
        if 'SELECT id FROM "zhiqing_rag".chat_run' in text:
            return Cursor(
                row for row in self.runs.values() if row["status"] in {"RUNNING", "PENDING"}
            )
        if 'INSERT INTO "zhiqing_rag".chat_message' in text:
            self.messages.append(parameters)
            return Cursor()
        if 'INSERT INTO "zhiqing_rag".answer_citation' in text:
            self.citations.append(parameters)
            return Cursor()
        if 'UPDATE "zhiqing_rag".chat_run' in text:
            if "status = 'SUCCEEDED'" in text:
                snapshot, standalone, model, input_tokens, output_tokens = parameters[:5]
                _, _, identifier, owner, epoch = parameters[5:]
                row = self.runs.get(identifier)
                if self.expire_during_complete or not self._owns(row, owner, epoch):
                    return Cursor()
                row.update(
                    status="SUCCEEDED",
                    execution_snapshot=deepcopy(snapshot.obj),
                    standalone_question=standalone,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    lease_owner=None,
                    lease_expires_at=None,
                    finished_at=NOW,
                )
                return Cursor([row])
            if "make_interval" in text:
                seconds, _, _, identifier, owner, epoch = parameters
                row = self.runs.get(identifier)
                if not self._owns(row, owner, epoch):
                    return Cursor()
                row["lease_expires_at"] = NOW + timedelta(seconds=seconds)
                return Cursor([{"id": identifier}])
            status, code, _, _, identifier, owner, epoch = parameters
            row = self.runs.get(identifier)
            if not self._owns(row, owner, epoch):
                return Cursor()
            row.update(
                status=status,
                error_code=code,
                finished_at=NOW,
                lease_owner=None,
                lease_expires_at=None,
            )
            return Cursor([{"id": identifier}])
        raise AssertionError(f"未处理的 SQL: {text}")

    @staticmethod
    def _owns(row, owner, epoch):
        return (
            row is not None
            and row["status"] == "RUNNING"
            and row["lease_owner"] == owner
            and row["lease_epoch"] == epoch
            and row["lease_expires_at"] > NOW
        )


@pytest.fixture
def database(monkeypatch):
    connection = FakeConnection()

    @contextmanager
    def borrow(settings):
        original = deepcopy((connection.runs, connection.messages, connection.citations))
        connection.events.append("begin")
        try:
            yield connection
        except BaseException:
            connection.runs, connection.messages, connection.citations = original
            connection.events.append("rollback")
            raise
        else:
            connection.events.append("commit")

    monkeypatch.setattr(repository_module, "database_connection", borrow)
    return connection


def repository():
    return ChatRepository(Settings(_env_file=None))


async def start(repo=None, *, request=None, digest=DIGEST, owner="worker"):
    return await (repo or repository()).start_run(
        identity(),
        20,
        request or REQUEST,
        digest,
        owner,
        120,
    )


async def complete(repo=None, *, result=None, history=None):
    return await (repo or repository()).complete(
        identity(),
        20,
        1,
        lease_owner="worker",
        lease_epoch=1,
        response=result or response(),
        standalone_question="工单如何关闭？",
        history_run_ids=history,
    )


async def test_start_writes_once_under_session_lock_and_refreshes_identity(database):
    main_thread = threading.get_ident()
    first = await start()
    retry = await start(owner="another-worker")
    assert first["is_new"] and not retry["is_new"]
    assert first["id"] == retry["id"] == "1"
    assert retry["lease_owner"] == "worker"
    assert len(database.runs) == len(database.messages) == 1
    statements = [text for text, _ in database.calls]
    lock = next(
        index
        for index, text in enumerate(statements)
        if ".chat_session" in text and "FOR UPDATE" in text
    )
    assert lock < next(index for index, text in enumerate(statements) if "INSERT INTO" in text)
    assert all(thread != main_thread for thread in database.thread_ids)
    assert database.events == ["begin", "commit", "begin", "commit"]


async def test_same_request_id_different_full_hash_conflicts(database):
    await start()
    with pytest.raises(AuthError) as error:
        await start(digest="b" * 64)
    assert (error.value.status_code, error.value.code) == (409, "CHAT_IDEMPOTENCY_CONFLICT")
    assert len(database.runs) == len(database.messages) == 1


async def test_different_request_id_cannot_overlap(database):
    await start()
    with pytest.raises(AuthError) as error:
        await start(request={**REQUEST, "client_request_id": "request-2"})
    assert error.value.code == "CHAT_RUN_ACTIVE"
    assert len(database.messages) == 1


@pytest.mark.parametrize("status", ["RUNNING", "PENDING"])
async def test_expired_active_run_is_abandoned_before_new_request(database, status):
    database.runs[1] = run_row(
        status=status,
        created_at=NOW - timedelta(minutes=10),
        lease_expires_at=NOW - timedelta(seconds=1),
    )
    result = await start(request={**REQUEST, "client_request_id": "new"})
    assert result["is_new"] and result["id"] == "2"
    assert database.runs[1]["status"] == "ABANDONED"
    assert database.runs[1]["lease_owner"] is None
    assert database.runs[1]["finished_at"] is not None


async def test_expired_same_request_retry_returns_terminal_instead_of_regenerating(database):
    database.runs[1] = run_row(lease_expires_at=NOW - timedelta(seconds=1))
    result = await start()
    assert not result["is_new"] and result["status"] == "ABANDONED"
    assert not database.messages


@pytest.mark.parametrize(
    "changes", [{"lease_epoch": 2}, {"lease_owner": "other"}, {"lease_expires_at": NOW}]
)
async def test_heartbeat_cannot_resurrect_expired_or_other_owner_run(database, changes):
    database.runs[1] = run_row(**changes)
    with pytest.raises(AuthError) as error:
        await repository().heartbeat(
            identity(), 20, 1, lease_owner="worker", lease_epoch=1, lease_seconds=120
        )
    assert error.value.code == "CHAT_LEASE_LOST"


async def test_complete_commits_response_citations_and_terminal_atomically(database):
    await start()
    result = await complete()
    assert result["status"] == "SUCCEEDED" and result["response"]["answer"]["status"] == "answered"
    assert len(database.messages) == 2 and len(database.citations) == 1
    stored = database.runs[1]
    assert (stored["input_tokens"], stored["output_tokens"]) == (32, 18)
    assert stored["lease_owner"] is None and stored["lease_expires_at"] is None
    assert stored["execution_snapshot"]["request"] == REQUEST
    assert stored["execution_snapshot"]["response"]["retrieval"]["evidence"][0]["content"] == (
        "  客户确认后关闭。\n"
    )
    query, args = next(call for call in database.calls if "SELECT c.kb_id" in call[0])
    assert "FOR SHARE OF doc, kb, g, c, r" in query
    assert "doc.active_generation_id = c.generation_id" in query
    assert "g.revision_id = c.revision_id" in query and "g.status = 'READY'" in query
    assert "doc.deleted_at IS NULL" in query and "kb.status = 'ACTIVE'" in query
    assert "doc.confidentiality <= %s" in query and "unnest" in query
    assert args[:9] == (1, 1, 2, 99, 0, 99, 10, 2, 99)
    assert args[-1] == [hashlib.sha256("  客户确认后关闭。\n".encode()).hexdigest()]


async def test_final_lease_expiry_rolls_back_all_response_writes(database):
    await start()
    database.expire_during_complete = True
    with pytest.raises(AuthError) as error:
        await complete()
    assert error.value.code == "CHAT_LEASE_LOST"
    assert database.runs[1]["status"] == "RUNNING"
    assert len(database.messages) == 1 and not database.citations
    assert database.events[-1] == "rollback"


async def test_permission_change_before_commit_does_not_store_answer(database):
    await start()
    database.allowed_sources = []
    with pytest.raises(AuthError) as error:
        await complete()
    assert error.value.code == "CHAT_SOURCE_UNAVAILABLE"
    assert len(database.messages) == 1 and not database.citations
    assert "response" not in database.runs[1]["execution_snapshot"]


async def test_history_restores_full_answer_but_does_not_depend_on_citation_rows(database):
    await start()
    await complete()
    database.citations.clear()
    result = await repository().get_run(identity(), 20, 1)
    assert result["visibility"] == "visible"
    citation = result["response"]["answer"]["statements"][0]["citations"][0]
    assert citation["quote"] == "客户确认后关闭"
    assert "execution_snapshot" not in result and "lease_owner" not in result


@pytest.mark.parametrize("source_change", ["removed", "hash", "generation", "revision", "content"])
async def test_history_hides_response_after_source_becomes_unavailable(database, source_change):
    await start()
    await complete()
    database.citations.clear()
    if source_change == "removed":
        database.allowed_sources = []
    elif source_change == "hash":
        database.allowed_sources[0]["content_sha256"] = "f" * 64
    elif source_change == "generation":
        database.allowed_sources[0]["generation_id"] = 99
    elif source_change == "revision":
        database.allowed_sources[0]["revision_id"] = 99
    else:
        database.allowed_sources[0]["content"] = "已被篡改但旧hash未变"
    result = await repository().get_run(identity(), 20, 1)
    assert result["response"] is None and result["visibility"] == "unavailable"
    assert result["standalone_question"] is None and result["question"] == REQUEST["question"]
    retry = await start()
    assert retry["response"] is None
    assert "response" not in retry["execution_snapshot"]


async def test_missing_or_unknown_snapshot_fails_closed(database):
    database.runs[1] = run_row(status="SUCCEEDED", execution_snapshot={"schema_version": 99})
    result = await repository().get_run(identity(), 20, 1)
    assert result["visibility"] == "unavailable" and result["response"] is None


async def test_keyset_pagination_returns_each_page_in_ascending_order(database):
    for identifier in range(1, 6):
        database.runs[identifier] = run_row(identifier, status="FAILED")
    latest = await repository().list_runs(identity(), 20, limit=2)
    older = await repository().list_runs(
        identity(),
        20,
        before_run_id=int(latest["next_before_run_id"]),
        limit=2,
    )
    assert [row["id"] for row in latest["runs"]] == ["4", "5"]
    assert latest["has_more"] and latest["next_before_run_id"] == "4"
    assert [row["id"] for row in older["runs"]] == ["2", "3"]


@pytest.mark.parametrize("change", [{"tenant_id": 9}, {"member_id": 11}])
async def test_other_tenant_or_member_gets_same_not_found(database, monkeypatch, change):
    # 绕过登录 token 一致性检查，仅独立核验会话归属的隐藏策略。
    monkeypatch.setattr(ChatRepository, "_refresh_identity", lambda self, conn, user: user)
    with pytest.raises(AuthError) as error:
        await repository().get_run(identity(**change), 20, 1)
    assert (error.value.status_code, error.value.code) == (404, "CHAT_NOT_FOUND")
    assert not any(".chat_run" in text for text, _ in database.calls)


async def test_context_provenance_is_flattened_and_rechecked_at_commit_and_read(database):
    prior = response(chunk_id="15")
    source = _source(prior["retrieval"]["evidence"][0])
    database.allowed_sources.append(source)
    database.runs[2] = run_row(2, status="SUCCEEDED", result=prior)
    database.runs[1] = run_row()
    await complete(history=[2])
    snapshot = database.runs[1]["execution_snapshot"]
    assert snapshot["history_run_ids"] == [2] and snapshot["history_sources"] == [source]
    database.allowed_sources = [
        item for item in database.allowed_sources if _source_key(item) != _source_key(source)
    ]
    result = await repository().get_run(identity(), 20, 1)
    assert result["visibility"] == "unavailable" and result["response"] is None


async def test_changed_context_source_rejects_new_response_with_other_valid_evidence(database):
    database.runs[2] = run_row(2, status="SUCCEEDED", result=response(chunk_id="15"))
    database.runs[1] = run_row()
    with pytest.raises(AuthError) as error:
        await complete(history=[2])
    assert error.value.code == "CHAT_SOURCE_UNAVAILABLE"
    assert not database.messages and not database.citations


async def test_transitive_context_sources_are_flattened_without_recursive_history_reads(database):
    inherited = _source(response(chunk_id="17")["retrieval"]["evidence"][0])
    direct = _source(response(chunk_id="15")["retrieval"]["evidence"][0])
    database.allowed_sources.extend([inherited, direct])
    database.runs[2] = run_row(2, status="SUCCEEDED", result=response(chunk_id="15"))
    database.runs[2]["execution_snapshot"]["history_sources"] = [inherited, direct]
    database.runs[1] = run_row()
    await complete(history=[2])
    sources = database.runs[1]["execution_snapshot"]["history_sources"]
    assert len(sources) == 2
    assert {_source_key(source) for source in sources} == {
        _source_key(inherited),
        _source_key(direct),
    }
    history_queries = [text for text, _ in database.calls if "id = ANY" in text]
    assert len(history_queries) == 1
    database.allowed_sources = [
        item for item in database.allowed_sources if _source_key(item) != _source_key(inherited)
    ]
    result = await repository().get_run(identity(), 20, 1)
    assert result["visibility"] == "unavailable"


async def test_terminate_marks_failure_without_storing_draft(database):
    await start()
    await repository().terminate(
        identity(),
        20,
        1,
        lease_owner="worker",
        lease_epoch=1,
        status="CANCELLED",
        error_code="CHAT_CLIENT_DISCONNECTED",
    )
    assert database.runs[1]["status"] == "CANCELLED"
    assert len(database.messages) == 1 and not database.citations


async def test_create_and_list_sessions_use_current_member(database):
    result = await repository().create_session(identity(), " 我的会话 ")
    assert result["title"] == "我的会话"
    listed = await repository().list_sessions(identity())
    assert len(listed) == 1


async def test_session_pagination_is_scoped_and_validated(database):
    await repository().list_sessions(identity(), limit=25, offset=50)
    statement, parameters = database.calls[-1]
    assert "ORDER BY last_active_at DESC, id DESC LIMIT %s OFFSET %s" in statement
    assert parameters == (1, 10, 25, 50)
    with pytest.raises(ValueError):
        await repository().list_sessions(identity(), offset=-1)


def test_migration_scope_contains_only_chat_tables():
    from scripts.migrate_chat_persistence import chat_tables

    assert [table.name for table in chat_tables()] == [
        "chat_session",
        "chat_run",
        "chat_message",
        "answer_citation",
    ]


@pytest.mark.parametrize("status", ["PENDING", "RUNNING"])
async def test_history_poll_reaps_crashed_run_without_new_question(database, status):
    database.runs[1] = run_row(
        status=status,
        created_at=NOW - timedelta(minutes=3),
        lease_expires_at=NOW - timedelta(seconds=1),
    )
    result = await repository().get_run(identity(), 20, 1)
    assert result["status"] == "ABANDONED" and result["visibility"] == "visible"
    assert result["response"] is None


async def test_database_error_is_redacted(database):
    database.error = psycopg.OperationalError("password=super-secret connection failure")
    with pytest.raises(AuthError) as error:
        await repository().list_sessions(identity())
    assert (error.value.status_code, error.value.code) == (503, "CHAT_DATABASE_UNAVAILABLE")
    assert "super-secret" not in str(error.value)
