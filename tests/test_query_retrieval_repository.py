"""召回 SQL、独立连接及实时授权复核；使用假连接，不访问外部服务。"""

import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import psycopg
import pytest

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing.embedding import EmbeddingValidationError
from zhiqing_rag.modules.identity.auth_schemas import (
    AuthError,
    AuthIdentity,
    AuthTenant,
    AuthUser,
)
from zhiqing_rag.query_retrieval import repository as repository_module
from zhiqing_rag.query_retrieval.models import ChunkRecord, RecallHit
from zhiqing_rag.query_retrieval.repository import QueryRetrievalRepository
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument

VECTOR = (1.0,) + (0.0,) * 1023


def identity():
    return AuthIdentity(
        user_id=2,
        tenant_id=1,
        member_id=10,
        department_id=7,
        role="MEMBER",
        clearance=2,
        auth_version=1,
        tenant_auth_version=2,
        session_key=uuid4(),
        user=AuthUser(id="2", email="user@example.test", display_name="用户"),
        tenant=AuthTenant(id="1", code="unit", name="测试租户", role="MEMBER"),
    )


def scope():
    return AccessScope(
        1,
        2,
        (3, 8),
        (4, 9),
        (
            PublishedDocument(4, 3, 5),
            PublishedDocument(9, 8, 11),
        ),
    )


def chunk(**changes):
    return ChunkRecord(
        **{
            "tenant_id": 1,
            "knowledge_base_id": 3,
            "document_id": 4,
            "revision_id": 6,
            "generation_id": 5,
            "chunk_id": 12,
            "chunk_index": 0,
            "document_title": "测试制度",
            "document_name": "测试制度.pdf",
            "content": "原文",
            "content_sha256": "a" * 64,
            "token_count": 2,
            "page_number": 1,
            "section_title": "第一节",
            **changes,
        }
    )


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows=()):
        self.session = {
            "user_id": 2,
            "tenant_id": 1,
            "tenant_member_id": 10,
            "revoked_at": None,
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        }
        # 当前部门及密级故意不同，证明不能复用请求开始时的旧授权参数。
        self.identity = {
            "user_id": 2,
            "tenant_id": 1,
            "member_id": 10,
            "department_id": 99,
            "clearance": 0,
            "role": "MEMBER",
            "auth_version": 1,
            "tenant_auth_version": 2,
        }
        self.rows = list(rows)
        self.calls = []
        self.thread_ids = []
        self.error = None
        self.incompatible = False
        self.revalidated_rows = None
        self.revoke_after_score = False

    def execute(self, statement, parameters=()):
        rendered = statement if isinstance(statement, str) else statement.as_string(None)
        self.calls.append((rendered, deepcopy(parameters)))
        self.thread_ids.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        if "AS session_user_id" in rendered:
            if self.session is None:
                return Cursor([])
            return Cursor([{
                **(self.identity or {}),
                "session_user_id": self.session["user_id"],
                "session_tenant_id": self.session["tenant_id"],
                "tenant_member_id": self.session["tenant_member_id"],
                "revoked_at": self.session["revoked_at"],
                "expires_at": self.session["expires_at"],
                "authorization_active": self.identity is not None,
            }])
        if ".auth_session a" in rendered:
            return Cursor([self.session] if self.session is not None else [])
        if ".tenant_member m" in rendered and "m.auth_version" in rendered:
            return Cursor([self.identity] if self.identity is not None else [])
        if "SELECT 1 AS incompatible" in rendered:
            return Cursor([{"incompatible": 1}] if self.incompatible else [])
        if "pdb.score(c.id)" in rendered and self.revoke_after_score:
            self.session["revoked_at"] = datetime.now(UTC)
        if "doc.status = 'PUBLISHED'" in rendered and "c.id = ANY" in rendered:
            if self.revalidated_rows is not None:
                return Cursor(deepcopy(self.revalidated_rows))
        return Cursor(deepcopy(self.rows))


@pytest.fixture
def fake_database(monkeypatch):
    connections = []
    template = FakeConnection()
    events = []

    @contextmanager
    def connection(settings):
        current = FakeConnection(template.rows)
        current.session = deepcopy(template.session)
        current.identity = deepcopy(template.identity)
        current.error = template.error
        current.incompatible = template.incompatible
        current.revalidated_rows = template.revalidated_rows
        current.revoke_after_score = template.revoke_after_score
        connections.append(current)
        events.append("open")
        try:
            yield current
        finally:
            events.append("close")

    monkeypatch.setattr(repository_module, "database_connection", connection)
    return template, connections, events


def repository():
    return QueryRetrievalRepository(Settings(_env_file=None))


def assert_authorized(statement, parameters):
    assert '"zhiqing_rag".document_chunk c' in statement
    assert "doc.active_generation_id = c.generation_id" in statement
    assert "g.revision_id = c.revision_id" in statement
    assert "r.id = c.revision_id" in statement
    assert "g.status = 'READY'" in statement
    assert "doc.status = 'PUBLISHED'" in statement
    assert "doc.deleted_at IS NULL" in statement
    assert "kb.status = 'ACTIVE'" in statement
    assert "doc.confidentiality <= %s" in statement
    assert "(c.document_id, c.kb_id, c.generation_id) IN" in statement
    assert "unnest(%s::bigint[], %s::bigint[], %s::bigint[])" in statement
    assert parameters[:9] == (1, 1, 2, 99, 0, 99, 10, 2, 99)
    assert parameters[9:13] == (1, [4, 9], [3, 8], [5, 11])


async def test_dense_filters_materialized_authorized_scope_before_exact_top_k(fake_database):
    template, connections, events = fake_database
    template.rows = [{**asdict(chunk()), "score": 0.75}]
    result = await repository().dense(identity(), scope(), VECTOR, 5)
    assert result == (RecallHit(chunk(), 0.75),)
    assert events == ["open", "close"]
    assert len(connections[0].calls) == 4
    statement, parameters = connections[0].calls[-1]
    assert_authorized(statement, parameters)
    assert "WITH authorized AS MATERIALIZED" in statement
    assert "FROM authorized" in statement
    assert "c.embedding IS NOT NULL" in statement
    assert statement.index("WITH authorized") < statement.index("ORDER BY")
    assert "embedding <=> %s::vector(1024)" in statement
    assert parameters[-1] == 5
    assert parameters[-2] == parameters[-3]
    assert parameters[-2].startswith("[1.0,0.0,")
    assert "1.0,0.0," not in statement


async def test_bm25_is_parameterized_literal_text_without_query_parser(fake_database):
    template, connections, _ = fake_database
    template.rows = [{**asdict(chunk()), "score": 3.5}]
    query = "  content:secret OR tenant_id:999 '*'; DROP TABLE documents; --  "
    result = await repository().bm25(identity(), scope(), query, 10)
    assert result == (RecallHit(chunk(), 3.5),)
    assert len(connections[0].calls) == 6
    authorized, authorized_parameters = connections[0].calls[2]
    assert_authorized(authorized, authorized_parameters)
    assert authorized.startswith("SELECT c.id AS chunk_id ")
    assert "LIMIT" not in authorized
    statement, parameters = connections[0].calls[3]
    assert "c.content ||| %s::text" in statement
    assert "pdb.score(c.id) AS score" in statement
    assert "ORDER BY score DESC, c.id ASC LIMIT %s" in statement
    assert all(fragment not in statement for fragment in ("parse(", "@@@", "pdb.query"))
    assert query.strip() not in statement
    assert parameters == (query.strip(), 1, [12], 10)
    assert "c.id = ANY(%s::bigint[])" in statement
    assert "JOIN" not in statement
    assert "SELECT c.id AS chunk_id, pdb.score(c.id) AS score" in statement
    assert statement.count("LIMIT") == 1
    reread, reread_parameters = connections[0].calls[5]
    assert_authorized(reread, reread_parameters)
    assert reread_parameters[-1] == [12]
    assert "c.id = ANY(%s::bigint[])" in reread


async def test_calls_use_separate_closed_connections_and_worker_threads(fake_database):
    _, connections, events = fake_database
    initial = threading.get_ident()
    current_identity = identity()
    source = repository()
    await source.dense(current_identity, scope(), VECTOR, 5)
    await source.bm25(current_identity, scope(), "工单", 5)
    await source.revalidate(current_identity, scope(), (chunk(),))
    assert len(connections) == 3
    assert events == ["open", "close"] * 3
    assert all(thread_id != initial for conn in connections for thread_id in conn.thread_ids)
    assert all(conn.calls[0][1] == (current_identity.session_key,) for conn in connections)
    assert all("AS authorization_active" in conn.calls[0][0] for conn in connections)


async def test_revalidate_rereads_fresh_content_preserves_order_and_removes_missing(fake_database):
    template, connections, _ = fake_database
    first, second, deleted = chunk(), chunk(chunk_id=13), chunk(chunk_id=14)
    fresh = replace(second, content="数据库复核正文", content_sha256="b" * 64, page_number=7)
    template.rows = [asdict(fresh), asdict(first)]
    result = await repository().revalidate(identity(), scope(), (first, second, deleted, first))
    assert result == (first, fresh)
    assert result[1].content != second.content
    statement, parameters = connections[0].calls[-1]
    assert_authorized(statement, parameters)
    assert "(c.id, c.document_id, c.kb_id, c.revision_id, c.generation_id) IN" in statement
    assert parameters[-5:] == ([12, 13, 14], [4, 4, 4], [3, 3, 3], [6, 6, 6], [5, 5, 5])


async def test_revalidate_cannot_expand_initial_exact_document_generation_scope(fake_database):
    template, connections, _ = fake_database
    outsider = [
        chunk(tenant_id=99),
        chunk(document_id=999),
        chunk(knowledge_base_id=8),
        chunk(generation_id=11),
    ]
    # 假连接即便返回越界数据，也不能使其进入输出。
    template.rows = [asdict(item) for item in [*outsider, chunk()]]
    result = await repository().revalidate(identity(), scope(), (*outsider, chunk()))
    assert result == (chunk(),)
    assert connections[0].calls[-1][1][-5:] == ([12], [4], [3], [6], [5])


@pytest.mark.parametrize("method", ["dense", "bm25", "revalidate"])
async def test_revoked_session_fails_before_any_chunk_read(fake_database, method):
    template, connections, events = fake_database
    template.session["revoked_at"] = datetime.now(UTC)
    with pytest.raises(AuthError) as error:
        await invoke(repository(), method)
    assert error.value.code == "SESSION_REVOKED"
    assert len(connections[0].calls) == 1
    assert events == ["open", "close"]


async def invoke(source, method, *, current_identity=None, access=None):
    current_identity = current_identity or identity()
    access = access or scope()
    if method == "dense":
        return await source.dense(current_identity, access, VECTOR, 5)
    if method == "bm25":
        return await source.bm25(current_identity, access, "工单", 5)
    return await source.revalidate(current_identity, access, (chunk(),))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "expired",
        "wrong_user",
        "wrong_tenant",
        "wrong_member",
    ],
)
async def test_session_must_be_current_and_bound_to_all_identity_ids(fake_database, mutation):
    template, connections, _ = fake_database
    if mutation == "missing":
        template.session = None
    elif mutation == "expired":
        template.session["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
    else:
        key = {
            "wrong_user": "user_id",
            "wrong_tenant": "tenant_id",
            "wrong_member": "tenant_member_id",
        }[mutation]
        template.session[key] = 999
    with pytest.raises(AuthError) as error:
        await invoke(repository(), "bm25")
    expected = "SESSION_REVOKED" if mutation in {"missing", "expired"} else "ACCESS_TOKEN_INVALID"
    assert error.value.code == expected
    assert len(connections[0].calls) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "disabled",
        "auth_version",
        "tenant_auth_version",
        "user_id",
        "tenant_id",
        "member_id",
    ],
)
async def test_live_member_and_tenant_versions_are_checked_before_chunk_sql(
    fake_database,
    mutation,
):
    template, connections, _ = fake_database
    if mutation == "disabled":
        template.identity = None
    else:
        template.identity[mutation] = 999
    with pytest.raises(AuthError) as error:
        await invoke(repository(), "dense")
    expected = (
        "AUTHORIZATION_CHANGED"
        if mutation in {"disabled", "auth_version", "tenant_auth_version"}
        else "ACCESS_TOKEN_INVALID"
    )
    assert error.value.code == expected
    assert len(connections[0].calls) == 1


@pytest.mark.parametrize("method", ["dense", "bm25", "revalidate"])
async def test_mismatched_scope_never_opens_database(fake_database, method):
    _, connections, _ = fake_database
    with pytest.raises(AuthError) as error:
        await invoke(repository(), method, access=replace(scope(), user_id=999))
    assert error.value.code == "QUERY_SCOPE_FORBIDDEN"
    assert connections == []


@pytest.mark.parametrize("method", ["dense", "bm25", "revalidate"])
async def test_empty_scope_still_refreshes_authorization_without_reading_chunks(
    fake_database,
    method,
):
    _, connections, _ = fake_database
    assert await invoke(repository(), method, access=AccessScope(1, 2, (), (), ())) == ()
    assert len(connections[0].calls) == 1


async def test_empty_candidates_still_check_live_session(fake_database):
    _, connections, _ = fake_database
    assert await repository().revalidate(identity(), scope(), ()) == ()
    assert len(connections[0].calls) == 1


@pytest.mark.parametrize("method", ["dense", "bm25", "revalidate"])
async def test_database_errors_are_not_silently_converted_to_empty_results(fake_database, method):
    template, _, events = fake_database
    template.error = psycopg.OperationalError("database unavailable")
    with pytest.raises(psycopg.OperationalError):
        await invoke(repository(), method)
    assert events == ["open", "close"]


@pytest.mark.parametrize(
    "vector",
    [
        (),
        (0.0,) * 1024,
        (float("nan"),) + (0.0,) * 1023,
        (float("inf"),) + (0.0,) * 1023,
        (True,) + (0.0,) * 1023,
    ],
)
async def test_invalid_embeddings_are_rejected_before_database_connection(fake_database, vector):
    _, connections, _ = fake_database
    with pytest.raises(EmbeddingValidationError):
        await repository().dense(identity(), scope(), vector, 5)
    assert connections == []


@pytest.mark.parametrize("limit", [0, 101, True, 1.0, "5"])
async def test_invalid_limits_are_rejected_without_database(fake_database, limit):
    _, connections, _ = fake_database
    with pytest.raises(ValueError):
        await repository().bm25(identity(), scope(), "工单", limit)
    assert connections == []


@pytest.mark.parametrize("query", ["", "  ", "x" * 2001, None, 123])
async def test_empty_or_invalid_keyword_query_is_rejected_before_database(fake_database, query):
    _, connections, _ = fake_database
    with pytest.raises(ValueError):
        await repository().bm25(identity(), scope(), query, 5)
    assert connections == []


async def test_dense_checks_all_authorized_vector_profiles_without_revision_assumptions(
    fake_database,
):
    _, connections, _ = fake_database
    await repository().dense(identity(), scope(), VECTOR, 5)
    statement, parameters = connections[0].calls[2]
    assert_authorized(statement, parameters)
    assert "SELECT 1 AS incompatible" in statement
    assert "c.embedding IS NOT NULL AND NOT EXISTS" in statement
    assert "profile.id = g.embedding_profile_id" in statement
    assert "profile.provider = %s AND profile.model_name = %s" in statement
    assert "profile.dimensions = %s AND profile.distance_metric = %s" in statement
    assert parameters[-4:] == ("dashscope", "text-embedding-v3", 1024, "cosine")
    assert "model_revision" not in statement


async def test_incompatible_vector_profile_fails_entire_dense_route_before_distance_sql(
    fake_database,
):
    template, connections, events = fake_database
    template.incompatible = True
    with pytest.raises(ValueError, match="跨模型空间"):
        await repository().dense(identity(), scope(), VECTOR, 5)
    assert len(connections[0].calls) == 3
    assert "<=>" not in connections[0].calls[-1][0]
    assert events == ["open", "close"]
    # BM25 无向量空间要求，仍可在同一初始范围检索全部文档。
    await repository().bm25(identity(), scope(), "工单", 5)
    assert connections[1].calls[-1][0].startswith("SELECT c.id AS chunk_id")


async def test_configured_dimension_must_match_fixed_index_before_connecting(fake_database):
    _, connections, _ = fake_database
    source = QueryRetrievalRepository(Settings(_env_file=None, embedding_dimensions=1536))
    with pytest.raises(ValueError, match="1024"):
        await source.dense(identity(), scope(), VECTOR, 5)
    assert connections == []


async def test_bm25_rechecks_permission_after_scoring_and_drops_no_longer_readable_text(
    fake_database,
):
    template, connections, _ = fake_database
    template.rows = [{**asdict(chunk()), "score": 1.0}]
    template.revalidated_rows = []
    assert await repository().bm25(identity(), scope(), "工单", 5) == ()
    assert len(connections[0].calls) == 6


async def test_bm25_revocation_between_scoring_and_reread_propagates_auth_error(fake_database):
    template, connections, _ = fake_database
    template.rows = [{**asdict(chunk()), "score": 1.0}]
    template.revoke_after_score = True
    with pytest.raises(AuthError) as error:
        await repository().bm25(identity(), scope(), "工单", 5)
    assert error.value.code == "SESSION_REVOKED"
    assert len(connections[0].calls) == 5


async def test_bm25_final_reread_preserves_score_order_not_database_order(fake_database):
    template, _, _ = fake_database
    first, second = chunk(), chunk(chunk_id=13)
    template.rows = [{**asdict(first), "score": 4.0}, {**asdict(second), "score": 2.0}]
    template.revalidated_rows = [asdict(second), asdict(first)]
    result = await repository().bm25(identity(), scope(), "工单", 5)
    assert result == (RecallHit(first, 4.0), RecallHit(second, 2.0))


async def test_empty_authorized_ids_skip_indexed_scoring_query(fake_database):
    _, connections, _ = fake_database
    assert await repository().bm25(identity(), scope(), "工单", 5) == ()
    assert len(connections[0].calls) == 3
    assert all("pdb.score" not in statement for statement, _ in connections[0].calls)


@pytest.mark.parametrize("method", ["dense", "bm25", "revalidate"])
async def test_expansion_of_join_custom_scan_is_disabled_only_for_local_transaction(
    fake_database,
    method,
):
    _, connections, _ = fake_database
    await invoke(repository(), method)
    assert connections[0].calls[1] == ("SET LOCAL paradedb.enable_join_custom_scan = off", ())
    assert all(
        "SET LOCAL paradedb.enable_custom_scan =" not in text for text, _ in connections[0].calls
    )
