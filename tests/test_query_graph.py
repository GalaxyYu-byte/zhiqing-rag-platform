"""在线图召回的授权、原文出处、歧义和预算；全部使用本地假依赖。"""

import asyncio
import hashlib
import threading
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity, AuthTenant, AuthUser
from zhiqing_rag.query_retrieval import graph as graph_module
from zhiqing_rag.query_retrieval.graph import (
    AuthorizedGraphProjection,
    Neo4jGraphReader,
    OnlineGraphRepository,
    OnlineGraphService,
)
from zhiqing_rag.query_retrieval.models import ChunkRecord
from zhiqing_rag.query_routing.models import GraphParams, RetrievalParams
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument

SEED = "a" * 64
OBJECT = "b" * 64
ASSERTION = "c" * 64
CONTENT = "前言。审批完成后，系统管理员负责权限开通。"
QUOTE = "审批完成后，系统管理员负责权限开通。"


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


def settings(**changes):
    return Settings(_env_file=None, neo4j_enabled=True, **changes)


def projection(**changes):
    return replace(AuthorizedGraphProjection(1, 3, 4, 6, 5, 20, "graph:rebuild:ready"), **changes)


def chunk(**changes):
    return replace(
        ChunkRecord(
            tenant_id=1,
            knowledge_base_id=3,
            document_id=4,
            revision_id=6,
            generation_id=5,
            chunk_id=12,
            chunk_index=0,
            document_title="权限制度",
            document_name="权限制度.md",
            content=CONTENT,
            content_sha256=hashlib.sha256(CONTENT.encode()).hexdigest(),
            token_count=30,
            page_number=None,
            section_title="开通",
        ),
        **changes,
    )


def relation(**changes):
    value = {
        "assertion_id": ASSERTION,
        "seed_entity_uid": SEED,
        "subject_entity_uid": SEED,
        "object_entity_uid": OBJECT,
        "subject_type": "Actor",
        "object_type": "Step",
        "predicate": "RESPONSIBLE_FOR",
        "condition_text": "审批完成后",
        "evidence": {
            **projection().parameters(),
            "chunk_id": 12,
            "content_sha256": chunk().content_sha256,
            "quote": QUOTE,
            "quote_start": CONTENT.index(QUOTE),
            "quote_end": len(CONTENT),
        },
    }
    value.update(changes)
    return value


def params(**changes):
    value = dict(seed_entity_uids=(SEED,), projections=(projection().ref(),), max_hops=1)
    value.update(changes)
    return GraphParams(**value)


class Repository:
    def __init__(self):
        self.current = (projection(),)
        self.records = (chunk(),)
        self.calls = []
        self.chunk_error = None

    async def projections(self, identity, scope):
        self.calls.append(("projections", identity, scope))
        return self.current

    async def chunks(self, identity, scope, projections, evidence):
        self.calls.append(("chunks", identity, scope, projections, evidence))
        if self.chunk_error is not None:
            raise self.chunk_error
        return self.records


class Reader:
    def __init__(self):
        self.seed_rows = (
            {
                "normalized_name": "系统管理员",
                "entity_uid": SEED,
                "entity_type": "Actor",
            },
        )
        self.relation_rows = (relation(),)
        self.calls = []
        self.error = None

    async def seeds(self, projections, *, names=(), uids=(), limit):
        self.calls.append(("seeds", projections, names, uids, limit))
        if self.error is not None:
            raise self.error
        return self.seed_rows

    async def relations(self, projections, uids, limit):
        self.calls.append(("relations", projections, uids, limit))
        return self.relation_rows


@pytest.fixture
def service():
    return OnlineGraphService(settings(), repository=Repository(), reader=Reader())


async def test_readiness_matches_all_canonical_entities_and_keeps_retrieval_settings(service):
    result = await service.readiness(
        identity(),
        scope(),
        "系统管理员负责什么？",
        ("系统管理员",),
        RetrievalParams(candidate_k=20, top_k=7, rerank=False),
    )
    assert result.query == "系统管理员负责什么？"
    assert result.matched_entities == ("系统管理员",)
    assert result.params.seed_entity_uids == (SEED,)
    assert result.params.projections == (projection().ref(),)
    assert result.params.candidate_k == 20
    assert result.params.top_k == 7
    assert result.params.rerank is False
    assert result.params.max_hops == 1
    assert [call[0] for call in service.repository.calls] == [
        "projections",
        "chunks",
    ]
    assert [call[0] for call in service.reader.calls] == ["seeds", "relations"]


@pytest.mark.parametrize(
    "entities,rows",
    [
        (
            ("系统管理员", "资源负责人"),
            (
                {
                    "normalized_name": "系统管理员",
                    "entity_uid": SEED,
                    "entity_type": "Actor",
                },
            ),
        ),
        (
            ("系统管理员",),
            (
                {"normalized_name": "系统管理员", "entity_uid": SEED, "entity_type": "Actor"},
                {"normalized_name": "系统管理员", "entity_uid": OBJECT, "entity_type": "Actor"},
            ),
        ),
        (
            ("管理员",),
            (
                {
                    "normalized_name": "系统管理员",
                    "entity_uid": SEED,
                    "entity_type": "Actor",
                },
            ),
        ),
        (
            ("系统管理员",),
            (
                {
                    "normalized_name": "系统管理员",
                    "entity_uid": SEED,
                    "entity_type": "NotAnEntity",
                },
            ),
        ),
    ],
)
async def test_missing_ambiguous_alias_or_unknown_entity_does_not_approve(service, entities, rows):
    service.reader.seed_rows = rows
    assert (
        await service.readiness(identity(), scope(), "谁负责开通？", entities, RetrievalParams())
        is None
    )
    assert not any(call[0] == "relations" for call in service.reader.calls)


async def test_isolated_seed_and_invalid_source_do_not_approve_graph(service):
    service.reader.relation_rows = ()
    assert (
        await service.readiness(
            identity(), scope(), "谁负责开通？", ("系统管理员",), RetrievalParams()
        )
        is None
    )
    service.reader.relation_rows = (relation(),)
    service.repository.records = ()
    assert (
        await service.readiness(
            identity(), scope(), "谁负责开通？", ("系统管理员",), RetrievalParams()
        )
        is None
    )


async def test_recall_keeps_direction_condition_and_exact_original_quote(service):
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.seed_count == result.assertion_count == 1
    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.chunk == chunk()
    evidence = hit.graph_relations[0]
    assert evidence.predicate == "RESPONSIBLE_FOR"
    assert evidence.condition_text == "审批完成后"
    assert evidence.source_quote == QUOTE
    assert evidence.subject_entity_uid == SEED
    assert evidence.object_entity_uid == OBJECT
    assert hit.chunk.content[evidence.quote_start : evidence.quote_end] == evidence.source_quote


async def test_plan_projection_is_intersected_with_fresh_current_projection(service):
    outsider = projection(document_id=9, knowledge_base_id=8, generation_id=11, projection_id=21)
    service.repository.current = (projection(), outsider)
    await service.recall(identity(), scope(), params(), 5)
    assert all(call[1] == (projection(),) for call in service.reader.calls)


@pytest.mark.parametrize(
    "change",
    [
        {"generation_id": 99},
        {"projection_id": 99},
        {"namespace": "different-namespace"},
    ],
)
async def test_retired_or_replaced_projection_does_not_query_neo4j(service, change):
    service.repository.current = (projection(**change),)
    assert not (await service.recall(identity(), scope(), params(), 5)).hits
    assert service.reader.calls == []


async def test_lost_seed_visibility_does_not_expand_relations(service):
    service.reader.seed_rows = ()
    assert not (await service.recall(identity(), scope(), params(), 5)).hits
    assert [call[0] for call in service.reader.calls] == ["seeds"]


@pytest.mark.parametrize(
    "change",
    [
        {"tenant_id": True},
        {"chunk_id": True},
        {"tenant_id": 99},
        {"kb_id": 8},
        {"document_id": 99},
        {"generation_id": 99},
        {"revision_id": 99},
        {"namespace": "private-graph"},
        {"chunk_id": 99},
        {"content_sha256": "d" * 64},
        {"quote": "虚构原句"},
        {"quote": ""},
        {"quote_start": 0},
        {"quote_start": True},
        {"quote_end": 999},
    ],
)
async def test_cross_scope_old_generation_hash_or_quote_mismatch_cannot_return(service, change):
    item = relation()
    item["evidence"].update(change)
    service.reader.relation_rows = (item,)
    assert not (await service.recall(identity(), scope(), params(), 5)).hits


@pytest.mark.parametrize(
    "change",
    [
        {"predicate": "UNKNOWN"},
        {"subject_type": "Resource"},
        {"object_type": "Actor"},
        {"condition_text": "未经审批"},
        {"seed_entity_uid": OBJECT},
        {"assertion_id": "unsafe id"},
    ],
)
async def test_invalid_predicate_endpoint_condition_or_seed_is_rejected(service, change):
    service.reader.relation_rows = (relation(**change),)
    assert not (await service.recall(identity(), scope(), params(), 5)).hits


async def test_real_chunk_hash_is_verified_even_if_graph_and_stored_hash_agree(service):
    service.repository.records = (chunk(content="被修改的原文"),)
    assert not (await service.recall(identity(), scope(), params(), 5)).hits


async def test_acl_revocation_during_graph_read_propagates_before_return(service):
    service.repository.chunk_error = AuthError(401, "SESSION_REVOKED", "登录已撤销")
    with pytest.raises(AuthError, match="登录已撤销"):
        await service.recall(identity(), scope(), params(), 5)


async def test_readiness_revocation_after_seed_lookup_still_refreshes_original_text_access(service):
    async def relations(projections, uids, limit):
        service.repository.chunk_error = AuthError(401, "SESSION_REVOKED", "登录已撤销")
        return (relation(),)

    service.reader.relations = relations
    with pytest.raises(AuthError, match="登录已撤销"):
        await service.readiness(
            identity(), scope(), "系统管理员负责什么？", ("系统管理员",), RetrievalParams()
        )
    assert [call[0] for call in service.repository.calls] == ["projections", "chunks"]


async def test_formal_recall_refreshes_authorization_after_successful_readiness(service):
    ready = await service.readiness(
        identity(), scope(), "系统管理员负责什么？", ("系统管理员",), RetrievalParams()
    )
    service.repository.current = ()
    result = await service.recall(identity(), scope(), ready.params, 5)
    assert result.hits == ()
    assert [call[0] for call in service.repository.calls] == [
        "projections",
        "chunks",
        "projections",
    ]
    assert [call[0] for call in service.reader.calls] == ["seeds", "relations"]


async def test_document_revocation_between_graph_expansion_and_chunk_read_returns_empty(service):
    async def relations(projections, uids, limit):
        service.repository.records = ()
        return (relation(),)

    service.reader.relations = relations
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.hits == ()
    assert result.assertion_count == 0


async def test_readiness_uses_only_deterministic_unicode_case_normalization(service):
    service.reader.seed_rows = (
        {
            "normalized_name": "admin",
            "entity_uid": SEED,
            "entity_type": "Actor",
        },
    )
    result = await service.readiness(
        identity(), scope(), "ＡＤＭＩＮ 负责什么？", ("ＡＤＭＩＮ",), RetrievalParams()
    )
    assert result.matched_entities == ("ＡＤＭＩＮ",)
    assert service.reader.calls[0][2] == ("admin",)


async def test_readiness_seed_query_budget_is_fail_closed(service):
    service.reader.seed_rows = service.reader.seed_rows * 41
    assert (
        await service.readiness(
            identity(), scope(), "管理员负责什么？", ("系统管理员",), RetrievalParams()
        )
        is None
    )
    assert not any(call[0] == "relations" for call in service.reader.calls)


async def test_forged_plan_projection_cannot_expand_trusted_projection_set(service):
    forged = projection(tenant_id=99, document_id=99, projection_id=99, namespace="private")
    requested = params(projections=(projection().ref(), forged.ref()))
    result = await service.recall(identity(), scope(), requested, 5)
    assert result.hits
    assert all(call[1] == (projection(),) for call in service.reader.calls)


async def test_final_relation_revalidation_keeps_current_projection_and_rechecks_identity(service):
    outsiders = (
        chunk(tenant_id=99),
        chunk(knowledge_base_id=8),
        chunk(document_id=9),
        chunk(revision_id=99),
        chunk(generation_id=99),
    )
    assert await service.revalidate_relations(
        identity(), scope(), params(), (chunk(), *outsiders)
    ) == frozenset({12})
    assert [call[0] for call in service.repository.calls] == ["projections"]
    assert service.reader.calls == []
    with pytest.raises(AuthError):
        await service.revalidate_relations(
            replace(identity(), tenant_id=99), scope(), params(), (chunk(),)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"projection_id": 99},
        {"namespace": "replacement"},
        {"generation_id": 99},
    ],
)
async def test_final_relation_revalidation_excludes_replaced_projection_even_same_chunk(
    service, changes
):
    service.repository.current = (projection(**changes),)
    assert (
        await service.revalidate_relations(identity(), scope(), params(), (chunk(),)) == frozenset()
    )
    assert service.reader.calls == []


async def test_final_relation_revalidation_excludes_stale_or_no_longer_authorized_projection(
    service,
):
    service.repository.current = ()
    assert (
        await service.revalidate_relations(identity(), scope(), params(), (chunk(),)) == frozenset()
    )
    service.settings.query_graph_enabled = False
    service.repository.current = (projection(),)
    assert (
        await service.revalidate_relations(identity(), scope(), params(), (chunk(),)) == frozenset()
    )


async def test_final_relation_revalidation_propagates_revoked_identity(service):
    async def projections(identity, scope):
        raise AuthError(401, "SESSION_REVOKED", "登录已撤销")

    service.repository.projections = projections
    with pytest.raises(AuthError, match="登录已撤销"):
        await service.revalidate_relations(identity(), scope(), params(), (chunk(),))


async def test_deduplicates_assertions_groups_chunks_and_obeys_result_limit(service):
    other = chunk(chunk_id=13)
    service.repository.records = (chunk(), other)
    second = relation(assertion_id="d" * 64)
    second["evidence"]["chunk_id"] = 13
    service.reader.relation_rows = (relation(), relation(), second)
    result = await service.recall(identity(), scope(), params(), 1)
    assert len(result.hits) == 1
    assert len(result.hits[0].graph_relations) == 1
    assert result.assertion_count == 2


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"max_hops": 2}, "GRAPH_HOPS_UNSUPPORTED"),
        ({"seed_entity_uids": ("untrusted",)}, "GRAPH_SEED_INVALID"),
        ({"seed_entity_uids": (SEED, SEED)}, "GRAPH_SEED_BUDGET_INVALID"),
    ],
)
async def test_unsupported_hops_or_invalid_seed_fails_before_read(service, changes, code):
    service.settings.query_graph_max_hops = 1
    with pytest.raises(ValueError, match=code):
        await service.recall(identity(), scope(), params(**changes), 5)
    assert service.repository.calls == []


async def test_graph_switches_and_budgets(service):
    service.settings.query_graph_enabled = False
    assert (
        await service.readiness(identity(), scope(), "测试", ("系统管理员",), RetrievalParams())
        is None
    )
    with pytest.raises(ValueError, match="GRAPH_DISABLED"):
        await service.recall(identity(), scope(), params(), 5)
    service.settings.query_graph_enabled = True
    service.settings.query_graph_seed_limit = 1
    assert (
        await service.readiness(
            identity(), scope(), "测试", ("系统管理员", "负责人"), RetrievalParams()
        )
        is None
    )
    service.settings.query_graph_relation_limit = 1
    service.reader.relation_rows = (relation(), relation())
    with pytest.raises(ValueError, match="GRAPH_RELATION_BUDGET_EXCEEDED"):
        await service.recall(identity(), scope(), params(), 5)


async def test_scope_identity_mismatch_prevents_all_reads(service):
    with pytest.raises(AuthError):
        await service.readiness(
            replace(identity(), tenant_id=99), scope(), "测试", ("管理员",), RetrievalParams()
        )
    assert service.repository.calls == []


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class Connection:
    def __init__(self):
        self.calls = []
        self.threads = []
        self.revoked = False
        self.projections = [asdict(projection())]
        self.chunks = [asdict(chunk())]

    def execute(self, statement, parameters=()):
        rendered = statement if isinstance(statement, str) else statement.as_string(None)
        self.calls.append((rendered, deepcopy(parameters)))
        self.threads.append(threading.get_ident())
        if "AS session_user_id" in rendered:
            return Cursor(
                [
                    {
                        "user_id": 2,
                        "tenant_id": 1,
                        "member_id": 10,
                        "department_id": 99,
                        "clearance": 0,
                        "role": "MEMBER",
                        "auth_version": 1,
                        "tenant_auth_version": 2,
                        "session_user_id": 2,
                        "session_tenant_id": 1,
                        "tenant_member_id": 10,
                        "revoked_at": datetime.now(UTC) if self.revoked else None,
                        "expires_at": datetime.now(UTC) + timedelta(hours=1),
                        "authorization_active": True,
                    }
                ]
            )
        return Cursor(self.projections if "SELECT gp.tenant_id" in rendered else self.chunks)


@pytest.fixture
def database(monkeypatch):
    connections, events = [], []
    template = Connection()

    @contextmanager
    def connect(settings):
        connection = deepcopy(template)
        connections.append(connection)
        events.append("open")
        try:
            yield connection
        finally:
            events.append("close")

    monkeypatch.setattr(graph_module, "database_connection", connect)
    return template, connections, events


async def test_projection_sql_uses_fresh_acl_and_exact_document_generation_namespace(database):
    _, connections, events = database
    result = await OnlineGraphRepository(settings()).projections(identity(), scope())
    assert result == (projection(),)
    statement, parameters = connections[0].calls[-1]
    assert "gp.status = 'READY'" in statement and "g.status = 'READY'" in statement
    assert "doc.active_generation_id = gp.generation_id" in statement
    assert "g.revision_id = gp.revision_id" in statement
    assert "doc.deleted_at IS NULL" in statement and "doc.status = 'PUBLISHED'" in statement
    assert "(doc.id, doc.kb_id, gp.generation_id) IN" in statement
    assert "unnest(%s::bigint[], %s::bigint[], %s::bigint[])" in statement
    assert parameters[:9] == (1, 1, 2, 99, 0, 99, 10, 2, 99)
    assert parameters[-4:] == (1, [4, 9], [3, 8], [5, 11])
    assert "gp.graph_namespace = %s" not in statement
    assert result[0].namespace != settings().graph_namespace
    assert events == ["open", "close"]
    assert all(thread != threading.get_ident() for thread in connections[0].threads)


async def test_chunk_sql_revalidates_projection_and_full_source_tuple_on_new_connection(database):
    _, connections, events = database
    repository = OnlineGraphRepository(settings())
    current = identity()
    await repository.projections(current, scope())
    result = await repository.chunks(current, scope(), (projection(),), (relation()["evidence"],))
    assert result == (chunk(),)
    statement, parameters = connections[-1].calls[-1]
    assert "gp.status = 'READY'" in statement
    assert "gp.revision_id = c.revision_id" in statement
    assert "(gp.id, gp.graph_namespace, gp.document_id, gp.generation_id) IN" in statement
    assert "(c.id, c.document_id, c.kb_id, c.revision_id, c.generation_id) IN" in statement
    assert "doc.active_generation_id = c.generation_id" in statement
    assert parameters[-9:] == ([20], ["graph:rebuild:ready"], [4], [5], [12], [4], [3], [6], [5])
    assert events == ["open", "close", "open", "close"]
    assert len(connections) == 2
    assert all(conn.calls[0][1] == (current.session_key,) for conn in connections)


async def test_repository_drops_outside_initial_tuple_even_if_driver_returns_it(database):
    template, _, _ = database
    template.projections += [asdict(projection(knowledge_base_id=8))]
    assert await OnlineGraphRepository(settings()).projections(identity(), scope()) == (
        projection(),
    )
    template.chunks += [asdict(chunk(document_id=99)), asdict(chunk(generation_id=99))]
    result = await OnlineGraphRepository(settings()).chunks(
        identity(), scope(), (projection(),), (relation()["evidence"],)
    )
    assert result == (chunk(),)


async def test_repository_revoked_session_fails_before_projection_read(database):
    template, connections, events = database
    template.revoked = True
    with pytest.raises(AuthError) as error:
        await OnlineGraphRepository(settings()).projections(identity(), scope())
    assert error.value.code == "SESSION_REVOKED"
    assert len(connections[0].calls) == 1
    assert events == ["open", "close"]


class Result:
    def __aiter__(self):
        async def rows():
            yield {"row": "read"}

        return rows()


class Session:
    def __init__(self, driver):
        self.driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.driver.events.append("session_closed")

    async def run(self, query, parameters):
        self.driver.calls.append((query, parameters))
        if self.driver.slow:
            await asyncio.sleep(1)
        return Result()


class Driver:
    def __init__(self):
        self.calls, self.events, self.options = [], [], None
        self.slow = False

    def session(self, **options):
        self.options = options
        return Session(self)

    async def close(self):
        self.events.append("driver_closed")


@pytest.fixture
def driver(monkeypatch):
    fake = Driver()

    def create(uri, **options):
        fake.create_options = options
        return fake

    monkeypatch.setattr(graph_module.AsyncGraphDatabase, "driver", create)
    return fake


async def test_neo4j_queries_are_fixed_read_only_parameterized_and_projection_scoped(driver):
    reader = Neo4jGraphReader(settings())
    await reader.seeds((projection(),), names=("'); DELETE n; //",), limit=41)
    await reader.relations((projection(),), (SEED,), 100)
    assert driver.options == {"database": "neo4j", "default_access_mode": "READ"}
    assert driver.create_options["max_transaction_retry_time"] == 0
    assert driver.create_options["connection_acquisition_timeout"] == 3
    assert driver.events == ["session_closed", "driver_closed"] * 2
    seed_query, seed_parameters = driver.calls[0]
    assert seed_query.timeout == 3
    assert "'); DELETE n; //" not in seed_query.text
    assert seed_parameters["names"] == ["'); DELETE n; //"]
    relation_query, relation_parameters = driver.calls[1]
    for alias in ("a", "e", "se", "oe"):
        for field in ("namespace", "tenant_id", "kb_id", "document_id", "generation_id"):
            assert f"{alias}.{field}=p.{field}" in relation_query.text
    assert "SUPPORTED_BY" in relation_query.text
    assert "condition_text" in relation_query.text
    assert "LIMIT $row_limit" in relation_query.text
    assert "*1.." not in relation_query.text
    assert relation_parameters["projections"] == [projection().parameters()]
    assert relation_parameters["uids"] == [SEED]
    assert relation_parameters["row_limit"] == 100


async def test_neo4j_timeout_closes_session_and_driver(driver):
    driver.slow = True
    reader = Neo4jGraphReader(settings(query_graph_timeout_seconds=0.01))
    with pytest.raises(TimeoutError):
        await reader.seeds((projection(),), names=("系统管理员",), limit=41)
    assert driver.events == ["session_closed", "driver_closed"]


async def test_neo4j_driver_is_reused_only_within_operation_and_always_closed(driver):
    reader = Neo4jGraphReader(settings())
    async with reader.operation():
        await reader.seeds((projection(),), names=("系统管理员",), limit=41)
        await reader.relations((projection(),), (SEED,), 100)
        assert driver.events == ["session_closed", "session_closed"]
    assert driver.events == ["session_closed", "session_closed", "driver_closed"]
    await reader.seeds((projection(),), names=("系统管理员",), limit=41)
    assert driver.events[-2:] == ["session_closed", "driver_closed"]
