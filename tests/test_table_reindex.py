"""历史表格重建的内容诊断、幂等入队与发布事务安全性。"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    event,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing import chunking
from zhiqing_rag.document_processing.chunking import ChunkConfig, chunk_document
from zhiqing_rag.document_processing.format_validation import DocumentFormat
from zhiqing_rag.document_processing.parsers.models import (
    ElementKind,
    ParsedDocument,
    ParsedElement,
)
from zhiqing_rag.modules.documents import (
    BackgroundTask,
    Document,
    DocumentChunk,
    DocumentRevision,
    EmbeddingProfile,
    IndexGeneration,
    OutboxEvent,
    TaskAttempt,
)
from zhiqing_rag.modules.documents.task_queue import LeaseLostError, ProcessingError, TaskLease
from zhiqing_rag.modules.retrieval import GraphProjection
from zhiqing_rag.modules.schema import SCHEMA
from zhiqing_rag.workers import publication, reindex_tables
from zhiqing_rag.workers.index_snapshot import chunk_fingerprint


@pytest.fixture
def table_chunks(monkeypatch):
    parsed = ParsedDocument(
        format=DocumentFormat.XLSX,
        original_filename="handover.xlsx",
        file_sha256="0" * 64,
        elements=(
            ParsedElement(
                ElementKind.TABLE,
                "",
                0,
                source_ref="sheet:交接/table:1",
                sheet_name="交接",
                heading_path=("交接",),
                table_rows=(
                    ("设备交接台账", "", "", "", ""),
                    ("说明：逐行核对后归档", "", "", "", ""),
                    ("批次", "责任岗位", "检查动作", "交接材料", "完成依据"),
                    *(
                        (f"批次{i}", "验收岗位", f"核对设备序号{i}", "签收清单", "双方确认")
                        for i in range(1, 13)
                    ),
                ),
            ),
        ),
    )
    config = ChunkConfig(max_tokens=95, overlap_tokens=8)
    corrected = chunk_document(parsed, config=config)
    with monkeypatch.context() as context:
        context.setattr(chunking, "_table_header_index", lambda element: 0)
        legacy = chunk_document(parsed, config=config)
    return legacy, corrected


@pytest.fixture
def database(monkeypatch, table_chunks):
    owner = create_engine("sqlite://")
    engine = owner.execution_options(schema_translate_map={SCHEMA: None})
    metadata = MetaData()
    for model in (
        Document,
        DocumentRevision,
        EmbeddingProfile,
        IndexGeneration,
        DocumentChunk,
        BackgroundTask,
        TaskAttempt,
        GraphProjection,
        OutboxEvent,
    ):
        columns = []
        for col in model.__table__.columns:
            datatype = (
                JSON()
                if isinstance(col.type, JSONB)
                else Integer()
                if isinstance(col.type, BigInteger)
                else Text()
                if col.name == "embedding"
                else col.type
            )
            default = getattr(col.server_default, "arg", None)
            columns.append(
                Column(
                    col.name,
                    datatype,
                    primary_key=col.primary_key,
                    nullable=col.nullable,
                    server_default=(
                        text(str(default).replace("::jsonb", "")) if default is not None else None
                    ),
                )
            )
        Table(model.__tablename__, metadata, *columns)
    metadata.create_all(engine)
    for module in (reindex_tables, publication):
        monkeypatch.setattr(module, "lock_work_knowledge_base", lambda *args: 1)
        monkeypatch.setattr(module, "verify_access", lambda *args: None)
    monkeypatch.setattr(
        publication,
        "func",
        SimpleNamespace(clock_timestamp=lambda: func.current_timestamp()),
    )
    legacy, _ = table_chunks
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                Document(
                    id=1,
                    tenant_id=1,
                    kb_id=1,
                    title="handover.xlsx",
                    created_by=1,
                    status="PUBLISHED",
                    desired_revision_id=1,
                    active_generation_id=1,
                ),
                DocumentRevision(
                    id=1,
                    tenant_id=1,
                    document_id=1,
                    revision_no=1,
                    original_filename="handover.xlsx",
                    file_size=1,
                    mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    file_sha256="0" * 64,
                    bucket="test",
                    object_key="test/source",
                    uploaded_by=1,
                ),
                EmbeddingProfile(
                    id=1,
                    code="test",
                    provider="dashscope",
                    model_name=Settings(_env_file=None).embedding_model,
                    model_revision="v1",
                    dimensions=1024,
                    distance_metric="cosine",
                ),
                IndexGeneration(
                    id=1,
                    tenant_id=1,
                    document_id=1,
                    revision_id=1,
                    embedding_profile_id=1,
                    generation_no=1,
                    parser_version="document-processing-v1",
                    cleaner_version="cleaning-v1",
                    chunker_version="chunking-v2",
                    chunk_config={"max_tokens": 95, "overlap_tokens": 8},
                    status="READY",
                    chunk_count=len(legacy),
                    ready_at=datetime.now(),
                ),
                GraphProjection(
                    id=1,
                    tenant_id=1,
                    document_id=1,
                    revision_id=1,
                    generation_id=1,
                    projection_version=1,
                    graph_namespace="test",
                    extractor_version="old",
                    model_name="test",
                    status="READY",
                    ready_at=datetime.now(),
                ),
            ]
        )
        session.add_all(
            [
                DocumentChunk(
                    tenant_id=1,
                    kb_id=1,
                    document_id=1,
                    revision_id=1,
                    generation_id=1,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    embedding_content=chunk.embedding_content,
                    content_sha256=chunk.content_sha256,
                    token_count=chunk.token_count,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    metadata_={**chunk.metadata, "source_refs": list(chunk.source_refs)},
                )
                for chunk in legacy
            ]
        )
    yield engine
    owner.dispose()


def planned(engine, table_chunks):
    settings = Settings(_env_file=None)
    item, old_chunks, _ = reindex_tables.capture(engine, settings, 1, 1)
    return {**item, **reindex_tables.compare_table_chunks(old_chunks, table_chunks[1])}


def running_lease(engine, item):
    generation_id, task_id = reindex_tables.enqueue(engine, Settings(_env_file=None), item)
    with Session(engine) as session, session.begin():
        task = session.get(BackgroundTask, task_id)
        task.status = "RUNNING"
        task.lease_epoch = task.attempt_count = 1
        task.lease_owner = "test"
        task.lease_expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=10)
        session.add(
            TaskAttempt(
                tenant_id=1,
                task_id=task_id,
                attempt_no=1,
                lease_epoch=1,
                worker_id="test",
                stage="SAVING",
            )
        )
    return TaskLease(task_id, 1, 1, 1, generation_id, "test", 1)


def fake_complete(session, scope, embedded):
    # 本测试验证发布事务；向量内容/维度由现有 chunk_storage 测试覆盖。
    generation = session.get(IndexGeneration, scope.generation_id)
    generation.status = "READY"
    generation.ready_at = datetime.now()


def test_detection_skips_corrected_and_non_table_differences(table_chunks):
    legacy, corrected = table_chunks
    report = reindex_tables.compare_table_chunks(legacy, corrected)
    assert report["needs_reindex"]
    assert report["affected_tables"][0]["header_rows"] == [1, 3]
    assert not reindex_tables.compare_table_chunks(corrected, corrected)["needs_reindex"]
    non_table = tuple(replace(chunk, metadata={"kind": "text"}) for chunk in legacy)
    assert not reindex_tables.compare_table_chunks(legacy, non_table)["needs_reindex"]


def test_fingerprint_ignores_graph_cache_but_tracks_content_and_locator(table_chunks):
    legacy, _ = table_chunks
    cached = tuple(
        replace(chunk, metadata={**chunk.metadata, "_graph_extraction_cache": {"x": 1}})
        for chunk in legacy
    )
    assert chunk_fingerprint(cached) == chunk_fingerprint(legacy)
    changed = (replace(legacy[0], page_number=2), *legacy[1:])
    assert chunk_fingerprint(changed) != chunk_fingerprint(legacy)
    with pytest.raises(ValueError, match="CHUNK_INTEGRITY_INVALID"):
        chunk_fingerprint((replace(legacy[0], content="damaged"), *legacy[1:]))


def test_enqueue_idempotent_keeps_old_pointer_chunks_and_revision(database, table_chunks):
    item = planned(database, table_chunks)
    first = reindex_tables.enqueue(database, Settings(_env_file=None), item)
    assert first == reindex_tables.enqueue(database, Settings(_env_file=None), item)
    with Session(database) as session:
        document = session.get(Document, 1)
        assert document.active_generation_id == document.desired_revision_id == 1
        assert len(session.scalars(select(IndexGeneration)).all()) == 2
        assert len(session.scalars(select(DocumentRevision)).all()) == 1
        assert (
            session.get(IndexGeneration, first[0]).chunker_version == reindex_tables.REINDEX_VERSION
        )
        assert session.get(IndexGeneration, 1).status == "READY"
        assert session.get(GraphProjection, 1).status == "READY"
    unchanged = {**item, "needs_reindex": False}
    assert reindex_tables.enqueue(database, Settings(_env_file=None), unchanged) is None


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("revision", "REVISION_UNAVAILABLE"),
        ("base", "BASE_GENERATION_CHANGED"),
        ("work", "ACTIVE_WORK_EXISTS"),
    ],
)
def test_stale_preview_and_concurrent_work_rejected(database, table_chunks, mutation, code):
    item = planned(database, table_chunks)
    with Session(database) as session, session.begin():
        document = session.get(Document, 1)
        if mutation == "revision":
            document.desired_revision_id = 99
        elif mutation == "base":
            document.active_generation_id = 99
        else:
            session.add(
                BackgroundTask(
                    tenant_id=1,
                    document_id=1,
                    revision_id=1,
                    generation_id=1,
                    task_type="GRAPH",
                    idempotency_key="active-graph",
                )
            )
    with pytest.raises(ValueError, match=code):
        reindex_tables.enqueue(database, Settings(_env_file=None), item)


def test_staging_rejects_changed_parse_and_same_revision_base(database, table_chunks):
    item = planned(database, table_chunks)
    lease = running_lease(database, item)
    with pytest.raises(ProcessingError, match="REINDEX_PREVIEW_CHANGED"):
        publication.stage_chunks_for_graph(database, lease, table_chunks[0])
    with Session(database) as session, session.begin():
        session.get(Document, 1).active_generation_id = 99
    with pytest.raises(ProcessingError, match="BASE_GENERATION_CHANGED"):
        publication.stage_chunks_for_graph(database, lease, table_chunks[1])
    with Session(database) as session:
        assert session.get(IndexGeneration, lease.generation_id).chunk_count == 0


def test_revoked_write_permission_prevents_enqueue(database, table_chunks, monkeypatch):
    item = planned(database, table_chunks)

    def revoked(*args):
        raise ProcessingError("ACCESS_REVOKED")

    monkeypatch.setattr(reindex_tables, "verify_access", revoked)
    with pytest.raises(ProcessingError, match="ACCESS_REVOKED"):
        reindex_tables.enqueue(database, Settings(_env_file=None), item)
    with Session(database) as session:
        assert session.get(Document, 1).active_generation_id == 1
        assert len(session.scalars(select(IndexGeneration)).all()) == 1


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("lease", "租约已失效"),
        ("base", "BASE_GENERATION_CHANGED"),
        ("revision", "REVISION_UNAVAILABLE"),
        ("access", "ACCESS_REVOKED"),
    ],
)
def test_publish_rechecks_lease_pointer_revision_and_access(
    database,
    table_chunks,
    monkeypatch,
    mutation,
    code,
):
    lease = running_lease(database, planned(database, table_chunks))
    publication.stage_chunks_for_graph(database, lease, table_chunks[1])
    monkeypatch.setattr(publication, "complete_embedded_chunks", fake_complete)
    with Session(database) as session, session.begin():
        if mutation == "lease":
            session.get(BackgroundTask, lease.task_id).lease_expires_at = (
                datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=1)
            )
        elif mutation == "base":
            session.get(Document, 1).active_generation_id = 99
        elif mutation == "revision":
            session.get(Document, 1).desired_revision_id = 99

    def revoked(*args):
        raise ProcessingError("ACCESS_REVOKED")

    if mutation == "access":
        monkeypatch.setattr(publication, "verify_access", revoked)
    error = LeaseLostError if mutation == "lease" else ProcessingError
    with pytest.raises(error, match=code):
        publication.publish_chunks(
            database,
            lease,
            tuple(SimpleNamespace(chunk=chunk) for chunk in table_chunks[1]),
            {},
        )
    with Session(database) as session:
        assert session.get(IndexGeneration, lease.generation_id).status == "BUILDING"
        assert session.get(GraphProjection, 1).status == "READY"
        assert session.scalars(select(OutboxEvent)).all() == []


def test_publication_atomically_switches_and_cancels_old_graph(database, table_chunks, monkeypatch):
    item = planned(database, table_chunks)
    lease = running_lease(database, item)
    corrected = table_chunks[1]
    publication.stage_chunks_for_graph(database, lease, corrected)
    with Session(database) as session, session.begin():
        old_graph = BackgroundTask(
            tenant_id=1,
            document_id=1,
            revision_id=1,
            generation_id=1,
            task_type="GRAPH",
            idempotency_key="late-graph",
            status="RUNNING",
            lease_owner="graph",
            lease_epoch=1,
            attempt_count=1,
            lease_expires_at=datetime.now(UTC).replace(tzinfo=None) + timedelta(minutes=10),
        )
        session.add(old_graph)
        session.flush()
        session.add(
            TaskAttempt(
                tenant_id=1,
                task_id=old_graph.id,
                attempt_no=1,
                lease_epoch=1,
                worker_id="graph",
                stage="SAVING",
            )
        )
        old_graph_id = old_graph.id
    monkeypatch.setattr(publication, "complete_embedded_chunks", fake_complete)

    def reject_old_chunk_update(mapper, connection, row):
        if row.generation_id == 1:
            raise AssertionError("published chunks must be immutable")

    event.listen(DocumentChunk, "before_update", reject_old_chunk_update)
    try:
        publication.publish_chunks(
            database, lease, tuple(SimpleNamespace(chunk=chunk) for chunk in corrected), {}
        )
    finally:
        event.remove(DocumentChunk, "before_update", reject_old_chunk_update)
    with Session(database) as session:
        assert session.get(Document, 1).active_generation_id == lease.generation_id
        assert session.get(IndexGeneration, 1).status == "READY"
        assert session.get(GraphProjection, 1).status == "STALE"
        assert session.get(BackgroundTask, old_graph_id).status == "CANCELLED"
        assert session.get(BackgroundTask, old_graph_id).lease_epoch == 2
        assert (
            session.scalar(select(TaskAttempt).where(TaskAttempt.task_id == old_graph_id)).status
            == "CANCELLED"
        )
        assert session.scalar(select(OutboxEvent)).event_type == "GRAPH_PURGE_GENERATION"
        old = session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.generation_id == 1)
            .order_by(DocumentChunk.chunk_index)
        ).all()
        assert chunk_fingerprint(old) == item["base_fingerprint"]
    with pytest.raises(LeaseLostError):
        publication.publish_chunks(database, lease, (), {})


def test_commit_failure_keeps_old_pointer_and_graph(database, table_chunks, monkeypatch):
    lease = running_lease(database, planned(database, table_chunks))
    publication.stage_chunks_for_graph(database, lease, table_chunks[1])
    monkeypatch.setattr(publication, "complete_embedded_chunks", fake_complete)

    def reject_new_pointer(session, flush_context, instances):
        if any(
            isinstance(row, Document) and row.active_generation_id == lease.generation_id
            for row in session.dirty
        ):
            raise RuntimeError("simulated failure")

    event.listen(Session, "before_flush", reject_new_pointer)
    try:
        with pytest.raises(RuntimeError, match="simulated failure"):
            publication.publish_chunks(
                database,
                lease,
                tuple(SimpleNamespace(chunk=chunk) for chunk in table_chunks[1]),
                {},
            )
    finally:
        event.remove(Session, "before_flush", reject_new_pointer)
    with Session(database) as session:
        assert session.get(Document, 1).active_generation_id == 1
        assert session.get(IndexGeneration, lease.generation_id).status == "BUILDING"
        assert session.get(GraphProjection, 1).status == "READY"
        assert session.scalars(select(OutboxEvent)).all() == []
