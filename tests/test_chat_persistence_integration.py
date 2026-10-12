"""可选真实 PostgreSQL 验证；每项测试仅写入新生成的隔离 schema。"""

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row
from sqlalchemy import BigInteger, Column, MetaData, Table, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable
from test_chat_repository import REQUEST, identity, response

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.infrastructure.database.connection import close_database_pool
from zhiqing_rag.modules.chat import AnswerCitation, ChatMessage, ChatRun, ChatSession
from zhiqing_rag.modules.chat.repository import ChatRepository
from zhiqing_rag.modules.identity.auth_schemas import AuthError

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CHAT_DB_TESTS") != "1",
    reason="设置 RUN_CHAT_DB_TESTS=1 才执行 PostgreSQL 隔离 schema 聊天验证",
)


class IsolatedRepository(ChatRepository):
    """只替代正式登录会话，其他存储与来源授权 SQL 均原样访问真实数据库。"""

    def _refresh_identity(self, connection, current):
        return current


@dataclass
class Database:
    connection: psycopg.Connection
    settings: Settings
    schema: str
    repository: IsolatedRepository

    def execute(self, statement, parameters=None):
        return self.connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(self.schema)), parameters,
        )

    def scalar(self, statement, parameters=None):
        return next(iter(self.execute(statement, parameters).fetchone().values()))


def _chat_tables(schema):
    metadata = MetaData()
    # SQLAlchemy 在编译复合 FK 时需要元数据中的外部表；仅声明被聊天引用的列。
    Table("tenant", metadata, Column("id", BigInteger, primary_key=True), schema=schema)
    Table("tenant_member", metadata, Column("id", BigInteger, primary_key=True),
          Column("tenant_id", BigInteger), UniqueConstraint("tenant_id", "id"), schema=schema)
    Table("document_revision", metadata, Column("id", BigInteger, primary_key=True),
          Column("tenant_id", BigInteger), Column("document_id", BigInteger),
          UniqueConstraint("tenant_id", "document_id", "id"), schema=schema)

    def referred_schema(table, target_schema, constraint, original_schema):
        return schema

    return tuple(model.__table__.to_metadata(
        metadata, schema=schema, referred_schema_fn=referred_schema,
    ) for model in (ChatSession, ChatRun, ChatMessage, AnswerCitation))


def _prepare(database):
    statements = [
        "CREATE TABLE {schema}.tenant (id bigint PRIMARY KEY)",
        """CREATE TABLE {schema}.tenant_member (
             id bigint PRIMARY KEY, tenant_id bigint NOT NULL, UNIQUE (tenant_id, id))""",
        """CREATE TABLE {schema}.knowledge_base (
             id bigint PRIMARY KEY, tenant_id bigint NOT NULL,
             status text NOT NULL, deleted_at timestamptz)""",
        """CREATE TABLE {schema}.knowledge_base_grant (
             id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, tenant_id bigint NOT NULL,
             kb_id bigint NOT NULL, user_id bigint, department_id bigint,
             permission text NOT NULL)""",
        """CREATE TABLE {schema}.document (
             id bigint PRIMARY KEY, tenant_id bigint NOT NULL, kb_id bigint NOT NULL,
             active_generation_id bigint, status text NOT NULL, deleted_at timestamptz,
             confidentiality smallint NOT NULL, department_id bigint, created_by bigint NOT NULL,
             title text NOT NULL)""",
        """CREATE TABLE {schema}.document_access_grant (
             id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, tenant_id bigint NOT NULL,
             document_id bigint NOT NULL, user_id bigint, department_id bigint,
             permission text NOT NULL)""",
        """CREATE TABLE {schema}.document_revision (
             id bigint PRIMARY KEY, tenant_id bigint NOT NULL, document_id bigint NOT NULL,
             UNIQUE (tenant_id, document_id, id))""",
        """CREATE TABLE {schema}.index_generation (
             id bigint PRIMARY KEY, tenant_id bigint NOT NULL, document_id bigint NOT NULL,
             revision_id bigint NOT NULL, status text NOT NULL)""",
        """CREATE TABLE {schema}.document_chunk (
             id bigint PRIMARY KEY, tenant_id bigint NOT NULL, kb_id bigint NOT NULL,
             document_id bigint NOT NULL, revision_id bigint NOT NULL,
             generation_id bigint NOT NULL,
             content text NOT NULL, content_sha256 text NOT NULL)""",
    ]
    for statement in statements:
        database.execute(statement)
    dialect = postgresql.dialect()
    for table in _chat_tables(database.schema):
        database.connection.execute(str(CreateTable(table).compile(dialect=dialect)))
        for index in sorted(table.indexes, key=lambda value: value.name):
            database.connection.execute(str(CreateIndex(index).compile(dialect=dialect)))
    database.execute("INSERT INTO {schema}.tenant VALUES (1)")
    database.execute("INSERT INTO {schema}.tenant_member VALUES (10, 1)")
    database.execute("INSERT INTO {schema}.knowledge_base VALUES (3, 1, 'ACTIVE', NULL)")
    database.execute("""INSERT INTO {schema}.knowledge_base_grant
                          (tenant_id, kb_id, user_id, permission) VALUES (1, 3, 2, 'READ')""")
    database.execute("""INSERT INTO {schema}.document
                          VALUES (4, 1, 3, 5, 'PUBLISHED', NULL, 0, NULL, 10, '工单管理')""")
    database.execute("INSERT INTO {schema}.document_revision VALUES (6, 1, 4)")
    database.execute("INSERT INTO {schema}.index_generation VALUES (5, 1, 4, 6, 'READY')")
    evidence = response()["retrieval"]["evidence"][0]
    database.execute(
        "INSERT INTO {schema}.document_chunk VALUES (12, 1, 3, 4, 6, 5, %s, %s)",
        (evidence["content"], hashlib.sha256(evidence["content"].encode()).hexdigest()),
    )


@pytest.fixture
def database():
    configured = get_settings()
    # 本套测试仅针对本地 PostgreSQL，防止环境配置意外指向共享远端数据库。
    if configured.db_host not in {"localhost", "127.0.0.1", "::1"}:
        pytest.skip("聊天集成测试仅允许本地 PostgreSQL")
    try:
        connection = psycopg.connect(
            host=configured.db_host, port=configured.db_port, dbname=configured.db_name,
            user=configured.db_username, password=configured.db_password.get_secret_value(),
            connect_timeout=3, autocommit=True, row_factory=dict_row,
            options="-c statement_timeout=15000 -c lock_timeout=5000",
        )
    except psycopg.Error:
        pytest.skip("本地 PostgreSQL 不可连接，未创建测试 schema")
    generated_name = "chat_test_" + uuid4().hex
    assert re.fullmatch(r"chat_test_[0-9a-f]{32}", generated_name)
    assert generated_name != configured.db_schema
    settings = configured.model_copy(update={"db_schema": generated_name})
    created = False
    try:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(generated_name)))
        created = True
        environment = Database(connection, settings, generated_name, IsolatedRepository(settings))
        _prepare(environment)
        yield environment
    finally:
        close_database_pool(settings)
        if created:
            # 只允许删除此 fixture 当次创建的准确名字，拒绝业务 schema 或拼接路径。
            assert settings.db_schema == generated_name
            assert re.fullmatch(r"chat_test_[0-9a-f]{32}", generated_name)
            assert generated_name != configured.db_schema
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(
                sql.Identifier(generated_name),
            ))
        connection.close()


def request(identifier="integration-1"):
    return {**REQUEST, "client_request_id": identifier}


def digest(value):
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(serialized).hexdigest()


async def create_session(database):
    session = await database.repository.create_session(identity())
    return int(session["id"])


async def start(database, session_id, value=None, *, owner="worker"):
    value = value or request()
    return await database.repository.start_run(
        identity(), session_id, value, digest(value), owner, 120,
    )


async def complete(database, session_id, run, *, owner="worker", epoch=1):
    return await database.repository.complete(
        identity(), session_id, int(run["id"]), lease_owner=owner, lease_epoch=epoch,
        response=response(), standalone_question=REQUEST["question"],
    )


async def test_success_restores_json_and_paginates_real_records(database):
    session_id = await create_session(database)
    ids = []
    for index in range(3):
        run = await start(database, session_id, request(f"page-{index}"))
        ids.append(run["id"])
        saved = await complete(database, session_id, run)
        assert saved["status"] == "SUCCEEDED"
    recent = await database.repository.list_runs(identity(), session_id, limit=2)
    older = await database.repository.list_runs(
        identity(), session_id, before_run_id=int(recent["next_before_run_id"]), limit=2,
    )
    assert [run["id"] for run in recent["runs"]] == ids[1:]
    assert recent["has_more"] and [run["id"] for run in older["runs"]] == ids[:1]
    assert not older["has_more"]
    restored = await database.repository.get_run(identity(), session_id, int(ids[-1]))
    assert restored["response"]["retrieval"]["evidence"][0]["content"] == "  客户确认后关闭。\n"
    assert database.scalar("SELECT count(*) FROM {schema}.chat_message") == 6
    assert database.scalar("SELECT count(*) FROM {schema}.answer_citation") == 3
    stored = database.execute("""SELECT input_tokens, output_tokens, execution_snapshot,
                                     lease_owner, lease_expires_at FROM {schema}.chat_run
                               WHERE id = %s""", (int(ids[-1]),)).fetchone()
    assert (stored["input_tokens"], stored["output_tokens"]) == (32, 18)
    assert stored["execution_snapshot"]["request"]["client_request_id"] == "page-2"
    assert stored["lease_owner"] is None and stored["lease_expires_at"] is None
    replay = await start(database, session_id, request("page-2"))
    assert not replay["is_new"] and replay["id"] == ids[-1] and replay["status"] == "SUCCEEDED"
    sessions = await database.repository.list_sessions(identity())
    assert sessions[0]["title"] == REQUEST["question"]


async def test_concurrent_same_request_creates_one_run_and_one_user_message(database):
    session_id = await create_session(database)
    runs = await asyncio.gather(*(start(database, session_id, owner=f"worker-{index}")
                                  for index in range(6)))
    assert sum(run["is_new"] for run in runs) == 1
    assert len({run["id"] for run in runs}) == 1
    assert database.scalar("SELECT count(*) FROM {schema}.chat_run") == 1
    assert database.scalar("SELECT count(*) FROM {schema}.chat_message") == 1
    with pytest.raises(AuthError) as error:
        await database.repository.start_run(
            identity(), session_id, {**request(), "question": "不同问题"}, "f" * 64, "new", 120,
        )
    assert error.value.code == "CHAT_IDEMPOTENCY_CONFLICT"


async def test_concurrent_different_requests_allow_only_one_active_run(database):
    session_id = await create_session(database)
    results = await asyncio.gather(start(database, session_id, request("a")),
                                   start(database, session_id, request("b")),
                                   return_exceptions=True)
    successes = [item for item in results if isinstance(item, dict)]
    failures = [item for item in results if isinstance(item, AuthError)]
    assert len(successes) == len(failures) == 1
    assert failures[0].code == "CHAT_RUN_ACTIVE"
    assert database.scalar("SELECT count(*) FROM {schema}.chat_run") == 1
    # 绕过 repository 的直接写入仍被部分唯一索引拒绝。
    with pytest.raises(psycopg.errors.UniqueViolation):
        database.execute("""INSERT INTO {schema}.chat_run
                          (tenant_id, session_id, client_request_id, request_sha256, question)
                          VALUES (1, %s, 'raw-duplicate', %s, '直接写入')""",
                         (session_id, "a" * 64))


@pytest.mark.parametrize("changed", ["owner", "epoch", "expired"])
async def test_old_or_expired_lease_cannot_complete(database, changed):
    session_id = await create_session(database)
    run = await start(database, session_id)
    if changed == "owner":
        database.execute("UPDATE {schema}.chat_run SET lease_owner = 'new-worker'")
    elif changed == "epoch":
        database.execute("UPDATE {schema}.chat_run SET lease_epoch = 2")
    else:
        database.execute("""UPDATE {schema}.chat_run
                            SET lease_expires_at = clock_timestamp() - interval '1 second'""")
    with pytest.raises(AuthError) as error:
        await complete(database, session_id, run)
    assert error.value.code == "CHAT_LEASE_LOST"
    assert database.scalar("SELECT count(*) FROM {schema}.chat_message WHERE role='ASSISTANT'") == 0
    assert database.scalar("SELECT count(*) FROM {schema}.answer_citation") == 0


@pytest.mark.parametrize("status", ["RUNNING", "PENDING"])
async def test_history_poll_reaps_real_expired_running_and_pending(database, status):
    session_id = await create_session(database)
    run = await start(database, session_id)
    if status == "PENDING":
        database.execute("""UPDATE {schema}.chat_run SET status = 'PENDING', lease_owner = NULL,
                             lease_expires_at = NULL,
                             created_at = clock_timestamp() - interval '10 minutes'""")
    else:
        database.execute("""UPDATE {schema}.chat_run
                            SET lease_expires_at = clock_timestamp() - interval '1 second'""")
    restored = await database.repository.get_run(identity(), session_id, int(run["id"]))
    assert restored["status"] == "ABANDONED"
    retry = await start(database, session_id)
    assert not retry["is_new"] and retry["status"] == "ABANDONED"
    new = await start(database, session_id, request("next"))
    assert new["is_new"]


@pytest.mark.parametrize("change", ["acl", "chunk", "generation", "content"])
async def test_history_source_change_hides_response_even_after_citation_purge(database, change):
    session_id = await create_session(database)
    run = await start(database, session_id)
    await complete(database, session_id, run)
    database.execute("DELETE FROM {schema}.answer_citation")
    restored = await database.repository.get_run(identity(), session_id, int(run["id"]))
    assert restored["visibility"] == "visible" and restored["response"] is not None
    if change == "acl":
        database.execute("DELETE FROM {schema}.knowledge_base_grant")
    elif change == "chunk":
        database.execute("DELETE FROM {schema}.document_chunk")
    elif change == "generation":
        database.execute("UPDATE {schema}.document SET active_generation_id = 999")
    else:
        # 改正文却保留旧哈希，也不能播放快照。
        database.execute("UPDATE {schema}.document_chunk SET content = '正文已经变化'")
    hidden = await database.repository.get_run(identity(), session_id, int(run["id"]))
    assert hidden["visibility"] == "unavailable" and hidden["response"] is None
    assert hidden["standalone_question"] is None
    history = await database.repository.list_runs(identity(), session_id)
    assert history["runs"][0]["response"] is None


async def test_revocation_before_complete_keeps_only_user_message(database):
    session_id = await create_session(database)
    run = await start(database, session_id)
    database.execute("DELETE FROM {schema}.knowledge_base_grant")
    with pytest.raises(AuthError) as error:
        await complete(database, session_id, run)
    assert error.value.code == "CHAT_SOURCE_UNAVAILABLE"
    assert database.scalar("SELECT count(*) FROM {schema}.chat_message WHERE role='ASSISTANT'") == 0
    assert database.scalar("SELECT count(*) FROM {schema}.answer_citation") == 0
    assert database.scalar("SELECT status FROM {schema}.chat_run") == "RUNNING"


async def test_deleting_context_chunk_hides_followup_with_other_valid_current_chunk(database):
    session_id = await create_session(database)
    first = await start(database, session_id)
    await complete(database, session_id, first)
    evidence = response(chunk_id="15")["retrieval"]["evidence"][0]
    database.execute(
        "INSERT INTO {schema}.document_chunk VALUES (15, 1, 3, 4, 6, 5, %s, %s)",
        (evidence["content"], hashlib.sha256(evidence["content"].encode()).hexdigest()),
    )
    second = await start(database, session_id, request("followup"))
    await database.repository.complete(
        identity(), session_id, int(second["id"]), lease_owner="worker", lease_epoch=1,
        response=response(chunk_id="15"), standalone_question=REQUEST["question"],
        history_run_ids=[int(first["id"])],
    )
    database.execute("DELETE FROM {schema}.answer_citation")
    database.execute("DELETE FROM {schema}.document_chunk WHERE id = 12")
    assert database.scalar("SELECT count(*) FROM {schema}.document_chunk WHERE id = 15") == 1
    hidden = await database.repository.get_run(identity(), session_id, int(second["id"]))
    assert hidden["visibility"] == "unavailable" and hidden["response"] is None


@pytest.mark.parametrize("changes", [{"tenant_id": 9}, {"member_id": 11}])
async def test_session_ownership_uses_both_tenant_and_member(database, changes):
    session_id = await create_session(database)
    run = await start(database, session_id)
    with pytest.raises(AuthError) as error:
        await database.repository.get_run(identity(**changes), session_id, int(run["id"]))
    assert (error.value.status_code, error.value.code) == (404, "CHAT_NOT_FOUND")


async def test_failure_after_assistant_and_citation_insert_rolls_back_entire_completion(database):
    session_id = await create_session(database)
    run = await start(database, session_id)
    database.execute("""CREATE FUNCTION {schema}.reject_success() RETURNS trigger
      LANGUAGE plpgsql AS $body$
      BEGIN
        IF NEW.status = 'SUCCEEDED' THEN
          RAISE EXCEPTION USING ERRCODE = '23514', MESSAGE = 'integration rejection';
        END IF;
        RETURN NEW;
      END;
      $body$""")
    database.execute("""CREATE TRIGGER reject_success BEFORE UPDATE ON {schema}.chat_run
                         FOR EACH ROW EXECUTE FUNCTION {schema}.reject_success()""")
    with pytest.raises(AuthError) as error:
        await complete(database, session_id, run)
    assert error.value.code == "CHAT_DATABASE_UNAVAILABLE" and error.value.status_code == 503
    assert database.scalar("SELECT count(*) FROM {schema}.chat_message") == 1
    assert database.scalar("SELECT count(*) FROM {schema}.answer_citation") == 0
    assert database.scalar("SELECT status FROM {schema}.chat_run") == "RUNNING"
    snapshot = database.scalar("SELECT execution_snapshot FROM {schema}.chat_run")
    assert "response" not in snapshot
