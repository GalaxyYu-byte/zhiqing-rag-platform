"""候选事务与图校验；可选 PostgreSQL 临时表实测，不改业务表或调用模型。"""

import copy
import hashlib
import os
from datetime import datetime

import pytest
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Identity,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import DateTime

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.documents import (
    BackgroundTask,
    Document,
    DocumentChunk,
    IndexGeneration,
    TaskAttempt,
)
from zhiqing_rag.modules.retrieval import GraphProjection, active_graph_projection
from zhiqing_rag.modules.schema import SCHEMA
from zhiqing_rag.workers import graph_queue, rebuild_graph
from zhiqing_rag.workers.graph_extraction import EXTRACTOR_VERSION, Extraction
from zhiqing_rag.workers.graph_storage import GraphStore


class TestClock(GenericFunction):
    __test__ = False
    name = "clock_timestamp"
    type = DateTime()
    inherit_cache = True


@compiles(TestClock, "sqlite")
def sqlite_clock(element, compiler, **kwargs):
    return "CURRENT_TIMESTAMP"


def extraction():
    quote = "财务审核报销申请"
    return Extraction.model_validate(
        {
            "entities": [
                {
                    "id": "a",
                    "type": "Actor",
                    "name": "财务",
                    "canonical_name": "财务",
                    "evidence_quote": quote,
                },
                {
                    "id": "s",
                    "type": "Step",
                    "name": "审核报销申请",
                    "canonical_name": "审核报销申请",
                    "evidence_quote": quote,
                },
            ],
            "relations": [
                {
                    "type": "REVIEWS",
                    "source_id": "a",
                    "target_id": "s",
                    "evidence_quote": quote,
                    "condition_text": None,
                }
            ],
        }
    )


@pytest.fixture(
    params=["sqlite"] + (["postgres"] if os.getenv("RUN_GRAPH_REBUILD_PG") == "1" else [])
)
def database(monkeypatch, request):
    # PostgreSQL 仅使用本连接的 pg_temp，物理测试表不带业务 FK；核验真实 JSONB、
    # READY 唯一索引和切换事务，不代替业务授权或完整业务 Schema 的集成测试。
    postgres = request.param == "postgres"
    if postgres:
        from zhiqing_rag.core.config import get_settings

        owner_engine = create_engine(get_settings().database_url)
        engine = owner_engine.connect().execution_options(schema_translate_map={SCHEMA: "pg_temp"})
    else:
        owner_engine = create_engine("sqlite://")
        engine = owner_engine.execution_options(schema_translate_map={SCHEMA: None})
    metadata = MetaData(schema="pg_temp" if postgres else None)
    for model in (
        Document,
        IndexGeneration,
        DocumentChunk,
        BackgroundTask,
        GraphProjection,
        TaskAttempt,
    ):
        columns = []
        for col in model.__table__.columns:
            datatype = (
                col.type
                if postgres
                else (
                    JSON()
                    if isinstance(col.type, JSONB)
                    else Integer()
                    if isinstance(col.type, BigInteger)
                    else Text()
                    if col.name == "embedding"
                    else col.type
                )
            )
            default = getattr(col.server_default, "arg", None)
            if default is not None:
                default = text(str(default) if postgres else str(default).replace("::jsonb", ""))
            if postgres and col.primary_key:
                default = Identity(start=1000)
            columns.append(
                Column(
                    col.name,
                    datatype,
                    primary_key=col.primary_key,
                    nullable=col.nullable,
                    server_default=default,
                )
            )
        table = Table(
            model.__tablename__, metadata, *columns, prefixes=["TEMPORARY"] if postgres else []
        )
        if model is GraphProjection:
            Index(
                "only_one_ready",
                table.c.tenant_id,
                table.c.generation_id,
                unique=True,
                sqlite_where=table.c.status == "READY",
                postgresql_where=table.c.status == "READY",
            )
    metadata.create_all(engine)
    if postgres:
        engine.commit()
    monkeypatch.setattr(
        rebuild_graph, "verify_access", lambda session, document, revision_id=None: None,
    )
    monkeypatch.setattr(
        graph_queue, "verify_access", lambda session, document, revision_id=None: None,
    )
    monkeypatch.setattr(rebuild_graph, "lock_work_knowledge_base", lambda *args: 1)
    monkeypatch.setattr(graph_queue, "lock_work_knowledge_base", lambda *args: 1)
    content = "财务审核报销申请"
    with Session(engine) as session, session.begin():
        for doc_id in (1, 2):
            session.add(
                Document(
                    id=doc_id,
                    tenant_id=1,
                    kb_id=1,
                    title=f"测试文档{doc_id}",
                    created_by=1,
                    status="PUBLISHED",
                    desired_revision_id=doc_id,
                    active_generation_id=doc_id,
                    metadata_={},
                )
            )
            session.add(
                IndexGeneration(
                    id=doc_id,
                    tenant_id=1,
                    document_id=doc_id,
                    revision_id=doc_id,
                    embedding_profile_id=1,
                    generation_no=1,
                    parser_version="test",
                    cleaner_version="test",
                    chunker_version="test",
                    chunk_config={},
                    status="READY",
                    chunk_count=1,
                    ready_at=datetime.now(),
                )
            )
            session.add(
                DocumentChunk(
                    id=doc_id,
                    tenant_id=1,
                    kb_id=1,
                    document_id=doc_id,
                    revision_id=doc_id,
                    generation_id=doc_id,
                    chunk_index=0,
                    content=content,
                    embedding_content=content,
                    token_count=10,
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    metadata_={
                        "_graph_extraction_cache": {
                            "model": "old-model",
                            "extractor_version": "old-version",
                            "extraction": extraction().model_dump(mode="json"),
                        }
                    },
                )
            )
            session.add(
                GraphProjection(
                    id=doc_id,
                    tenant_id=1,
                    document_id=doc_id,
                    revision_id=doc_id,
                    generation_id=doc_id,
                    projection_version=1,
                    graph_namespace="old-graph",
                    extractor_version="old-version",
                    model_name="old-model",
                    status="READY",
                    ready_at=datetime.now(),
                    entity_count=2,
                    relation_count=1,
                )
            )
    yield engine
    if postgres:
        engine.close()
    owner_engine.dispose()


def build_local(engine):
    settings = Settings(_env_file=None, deepseek_api_key="test")
    item = rebuild_graph.plan(engine, settings, 1, [1])[0]
    projection_id, task_id = rebuild_graph.enqueue(engine, settings, item)
    context = rebuild_graph.projection_context(engine, 1, projection_id)
    queue = graph_queue.GraphQueue(
        engine, settings.model_copy(update={"graph_namespace": context["namespace"]})
    )
    assert queue.claim("ordinary-worker") is None
    lease = queue.claim("candidate-worker", task_id)
    assert lease.projection_version == 2
    scope, count = queue.scope(lease)
    chunk = queue.chunk(lease, 0)
    assert queue.cached_extraction(lease, chunk) is None
    queue.save_extraction(lease, chunk, extraction())
    assert queue.cached_extraction(lease, chunk) == extraction()
    store = object.__new__(GraphStore)
    store.settings = settings.model_copy(update={"graph_namespace": context["namespace"]})
    nodes, facts = store.chunk_records(scope, chunk, extraction())
    queue.checkpoint(lease, 1, count, {n["key"] for n in nodes}, {f["key"] for f in facts})
    queue.finish(lease, 2, 1)
    return rebuild_graph.projection_context(engine, 1, projection_id)


def snapshot(context):
    store = object.__new__(GraphStore)
    store.settings = Settings(_env_file=None, graph_namespace=context["namespace"])
    nodes, facts = store.chunk_records(context["scope"], context["chunks"][0], extraction())
    return {
        "entities": [
            {
                "key": row["key"],
                "type": row["type"],
                "name": row["name"],
                "normalized_name": row["normalized_name"],
                "namespace": context["namespace"],
                "tenant_id": 1,
                "kb_id": 1,
            }
            for row in nodes
        ],
        "evidence": list(
            {row["evidence"]["key"]: row["evidence"] for row in [*nodes, *facts]}.values()
        ),
        "assertions": [
            {
                "key": row["key"],
                "predicate": row["type"],
                "condition_text": row["condition_text"],
                "namespace": context["namespace"],
                "tenant_id": 1,
                "kb_id": 1,
                "document_id": 1,
                "generation_id": 1,
                "subjects": [row["source_key"]],
                "objects": [row["target_key"]],
                "evidence": [row["evidence"]["key"]],
            }
            for row in facts
        ],
    }


def approved(report):
    saved = copy.deepcopy(report)
    saved["reviewed_by"] = "test-reviewer"
    for row in saved["samples"]:
        row.update(verdict="SUPPORT", reason="原文明确给出主体、审核动作及对象。")
    return saved


def test_candidate_cache_does_not_update_frozen_chunk(database):
    def reject_update(mapper, connection, target):
        raise AssertionError("Published Chunk must remain immutable")

    event.listen(DocumentChunk, "before_update", reject_update)
    try:
        context = build_local(database)
        assert rebuild_graph.check_snapshot(context, snapshot(context))["structural_passed"]
    finally:
        event.remove(DocumentChunk, "before_update", reject_update)


def test_candidate_is_hidden_and_keeps_old_cache_then_switch_and_rollback(database):
    context = build_local(database)
    report = rebuild_graph.check_snapshot(context, snapshot(context))
    assert report["structural_passed"]
    with Session(database) as session:
        assert active_graph_projection(session, 1, 1).id == 1
        assert active_graph_projection(session, 1, 2).id == 2
        assert session.get(GraphProjection, context["projection_id"]).status == "BUILDING"
        chunk = session.get(DocumentChunk, 1)
        assert chunk.metadata_["_graph_extraction_cache"]["extractor_version"] == "old-version"
        assert "_graph_rebuild_cache" not in chunk.metadata_
        task = session.get(BackgroundTask, context["task_id"])
        assert task.payload["extraction_cache"]["1"]["extractor_version"] == EXTRACTOR_VERSION
    rebuild_graph.approve_review(approved(report), report)
    rebuild_graph.switch(database, 1, context["projection_id"], report)
    with Session(database) as session:
        assert active_graph_projection(session, 1, 1).id == context["projection_id"]
        assert active_graph_projection(session, 1, 2).id == 2
        assert session.get(GraphProjection, 1).status == "STALE"
    old_context = rebuild_graph.projection_context(database, 1, 1)
    old_report = rebuild_graph.check_snapshot(old_context, snapshot(old_context))
    assert old_report["structural_passed"]
    rebuild_graph.switch(
        database, 1, 1, old_report, rollback=True, expected_active_id=context["projection_id"]
    )
    with Session(database) as session:
        assert active_graph_projection(session, 1, 1).id == 1
        assert session.get(GraphProjection, context["projection_id"]).status == "STALE"


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing_evidence", "FACT_WITHOUT_VALID_EVIDENCE"),
        ("wrong_endpoint", "INVALID_RELATION_ENDPOINT"),
        ("bad_quote", "EVIDENCE_LOCATION_INVALID"),
        ("wrong_scope", "NODE_SCOPE_INVALID"),
        ("missing_fact", "RAW_PROJECTION_ASSERTIONS_MISMATCH"),
    ],
)
def test_projection_corruption_cannot_pass(database, mutation, error):
    context = build_local(database)
    data = snapshot(context)
    if mutation == "missing_evidence":
        data["assertions"][0]["evidence"] = []
    elif mutation == "wrong_endpoint":
        data["assertions"][0]["objects"] = ["other-document-node"]
    elif mutation == "bad_quote":
        data["evidence"][0]["quote"] = "不存在的证据"
    elif mutation == "wrong_scope":
        data["entities"][0]["tenant_id"] = 2
    else:
        data["assertions"] = []
    report = rebuild_graph.check_snapshot(context, data)
    assert not report["structural_passed"]
    assert error in report["issues"]
    with pytest.raises(ValueError, match="GRAPH_STRUCTURE_INVALID"):
        rebuild_graph.approve_review(approved(report), report)


def test_pending_insufficient_or_stale_evidence_report_is_blocked(database):
    context = build_local(database)
    report = rebuild_graph.check_snapshot(context, snapshot(context))
    with pytest.raises(ValueError, match="EVIDENCE_REVIEW_REQUIRED"):
        rebuild_graph.approve_review(report, report)
    saved = approved(report)
    saved["samples"][0]["verdict"] = "INSUFFICIENT"
    with pytest.raises(ValueError, match="EVIDENCE_REVIEW_NOT_PASSED"):
        rebuild_graph.approve_review(saved, report)
    saved = approved(report)
    saved["snapshot_fingerprint"] = "outdated"
    with pytest.raises(ValueError, match="REVIEW_STALE"):
        rebuild_graph.approve_review(saved, report)


def test_failed_candidate_and_stale_revision_keep_old_visible(database):
    settings = Settings(_env_file=None)
    projection_id, task_id = rebuild_graph.enqueue(
        database, settings, rebuild_graph.plan(database, settings, 1, [1])[0]
    )
    context = rebuild_graph.projection_context(database, 1, projection_id)
    queue = graph_queue.GraphQueue(
        database, settings.model_copy(update={"graph_namespace": context["namespace"]})
    )
    lease = queue.claim("test", task_id)
    queue.fail(lease, "GRAPH_EVIDENCE_INVALID", False)
    with Session(database) as session:
        assert active_graph_projection(session, 1, 1).id == 1
        assert session.get(GraphProjection, projection_id).status == "FAILED"
    with Session(database) as session, session.begin():
        session.get(Document, 1).desired_revision_id = 99
    with pytest.raises(ValueError, match="REVISION_UNAVAILABLE"):
        rebuild_graph.projection_context(database, 1, projection_id)


def test_activation_failure_rolls_back_old_ready_state(database):
    context = build_local(database)
    report = rebuild_graph.check_snapshot(context, snapshot(context))

    def fail_on_new_ready(session, flush_context, instances):
        if any(
            isinstance(row, GraphProjection)
            and row.id == context["projection_id"]
            and row.status == "READY"
            for row in session.dirty
        ):
            raise RuntimeError("simulated commit failure")

    event.listen(Session, "before_flush", fail_on_new_ready)
    try:
        with pytest.raises(RuntimeError, match="simulated commit failure"):
            rebuild_graph.switch(database, 1, context["projection_id"], report)
    finally:
        event.remove(Session, "before_flush", fail_on_new_ready)
    with Session(database) as session:
        assert active_graph_projection(session, 1, 1).id == 1
        assert session.get(GraphProjection, context["projection_id"]).status == "BUILDING"


def test_switch_rejects_changed_base_and_plan_is_bounded(database):
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="1～3"):
        rebuild_graph.plan(database, settings, 1, [1, 2, 3, 4])
    context = build_local(database)
    report = rebuild_graph.check_snapshot(context, snapshot(context))
    with Session(database) as session, session.begin():
        task = session.get(BackgroundTask, context["task_id"])
        task.payload = {**task.payload, "base_projection_id": 999}
    with pytest.raises(ValueError, match="BASE_GRAPH_CHANGED"):
        rebuild_graph.switch(database, 1, context["projection_id"], report)
    with Session(database) as session:
        assert active_graph_projection(session, 1, 1).id == 1


def test_excluded_fact_still_requires_bound_review_and_known_key(database):
    context = build_local(database)
    data = snapshot(context)
    context["task_payload"]["fact_exclusions"] = {"unknown": "原文不支持"}
    context["task_payload"]["fact_review"] = {
        "reviewed_by": "codex",
        "raw_cache_fingerprint": rebuild_graph.stable_key(
            context["task_payload"]["extraction_cache"]
        ),
    }
    report = rebuild_graph.check_snapshot(context, data)
    assert "FACT_EXCLUSION_KEY_INVALID" in report["issues"]
    context["task_payload"]["fact_review"]["raw_cache_fingerprint"] = "stale"
    assert "FACT_EXCLUSION_REVIEW_INVALID" in rebuild_graph.check_snapshot(
        context, data
    )["issues"]


@pytest.mark.asyncio
async def test_filter_candidate_keeps_raw_cache_and_old_graph_visible(database, monkeypatch):
    context = build_local(database)
    original_cache = copy.deepcopy(context["task_payload"]["extraction_cache"])
    extra = copy.deepcopy(original_cache["1"]["extraction"]["relations"][0])
    extra["type"] = "RESPONSIBLE_FOR"
    original_cache["1"]["extraction"]["relations"].append(extra)
    with Session(database) as session, session.begin():
        task = session.get(BackgroundTask, context["task_id"])
        task.payload = {**task.payload, "extraction_cache": original_cache}
    real_store = GraphStore

    class LocalStore(real_store):
        written = None

        def __init__(self, settings):
            self.settings = settings

        async def close(self):
            pass

        async def write_chunk(self, scope, chunk, result):
            LocalStore.written = result
            nodes, facts = self.chunk_records(scope, chunk, result)
            return {n["key"] for n in nodes}, {f["key"] for f in facts}

        async def snapshot(self, scope):
            return snapshot(rebuild_graph.projection_context(
                database, 1, context["projection_id"]
            ))

    monkeypatch.setattr(rebuild_graph, "GraphStore", LocalStore)
    store = LocalStore(Settings(_env_file=None, graph_namespace=context["namespace"]))
    _, facts = store.chunk_records(context["scope"], context["chunks"][0],
                                  Extraction.model_validate(original_cache["1"]["extraction"]))
    with pytest.raises(ValueError, match="FACT_EXCLUSION_KEY_INVALID"):
        await rebuild_graph.filter_candidate(
            database, Settings(_env_file=None), 1, context["projection_id"],
            {"unknown": "未支持"}, "codex"
        )
    assert LocalStore.written is None
    report = await rebuild_graph.filter_candidate(
        database, Settings(_env_file=None), 1, context["projection_id"],
        {facts[1]["key"]: "测试裁决剔除多余关系"}, "codex"
    )
    assert report["structural_passed"]
    assert len(LocalStore.written.relations) == 1
    with Session(database) as session:
        assert active_graph_projection(session, 1, 1).id == 1
        assert session.get(BackgroundTask, context["task_id"]).payload[
            "extraction_cache"
        ] == original_cache
        assert session.get(GraphProjection, context["projection_id"]).relation_count == 1
