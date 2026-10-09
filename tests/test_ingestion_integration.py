"""异步任务持久化、租约恢复、原子发布及真实模型闭环；自动清理测试数据。"""

import asyncio
import os
import sys
from dataclasses import replace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from minio.error import S3Error
from pydantic import SecretStr
from sqlalchemy.orm import Session

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.document_processing.embedding import EmbeddedChunk
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.infrastructure.object_storage import document_object_storage
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.documents.repository import UploadRepository
from zhiqing_rag.modules.documents.router import get_upload_identity
from zhiqing_rag.modules.documents.schemas import UploadError
from zhiqing_rag.modules.documents.task_queue import IngestionQueue, LeaseLostError
from zhiqing_rag.modules.retrieval import active_graph_projection
from zhiqing_rag.workers import backfill_graph, graph, ingestion, publication
from zhiqing_rag.workers.graph_extraction import Extraction
from zhiqing_rag.workers.graph_queue import GraphQueue
from zhiqing_rag.workers.publication import publish_chunks

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INGESTION_INTEGRATION") != "1", reason="需要显式启用后台任务数据库实测"
)


@pytest.fixture
def uploaded():
    settings = get_settings()
    filename = f"async-integration-{uuid4().hex}.md"
    with database_connection(settings) as connection:
        repository = UploadRepository(connection, settings.db_schema)
        identity = repository.identity(
            settings.dev_upload_user_email, settings.dev_upload_tenant_code
        )
        base = repository.writable_bases(identity)[0]
    # 此用例验证后台入库闭环，显式注入数据库已验证的测试身份。
    app = create_app()
    app.dependency_overrides[get_upload_identity] = lambda: identity
    with TestClient(app, client=("127.0.0.1", 54321)) as client:
        try:
            response = client.post(
                "/api/documents/upload",
                files={
                    "file": (
                        filename,
                        f"# {filename}\n\n异步文档处理测试。知识需要可靠的来源。\n".encode(),
                        "text/markdown",
                    )
                },
                data={"knowledge_base_id": base.id},
            )
            assert response.status_code == 202, response.text
            body = response.json()
            assert body["processing_status"] == "PENDING"
            yield settings, identity, body, client
        finally:
            with database_connection(settings) as connection:
                repository = UploadRepository(connection, settings.db_schema)
                repository.query(
                    """DELETE FROM {schema}.outbox_event WHERE tenant_id = %s
                       AND aggregate_type = 'graph_cleanup' AND aggregate_id = %s""",
                    (identity.tenant_id, str(body["document_id"])),
                )
                rows = repository.query(
                    """SELECT id, document_id, bucket, object_key FROM {schema}.document_revision
                       WHERE tenant_id = %s AND original_filename = %s""",
                    (identity.tenant_id, filename),
                ).fetchall()
                for row in rows:
                    repository.query(
                        """UPDATE {schema}.document SET status = 'DRAFT', deleted_at = NULL,
                           active_generation_id = NULL, desired_revision_id = NULL
                           WHERE tenant_id = %s AND id = %s AND title = %s""",
                        (identity.tenant_id, row["document_id"], filename),
                    )
                    repository.query(
                        """DELETE FROM {schema}.task_attempt WHERE tenant_id = %s
                           AND task_id IN (SELECT id FROM {schema}.background_task
                             WHERE tenant_id = %s AND document_id = %s)""",
                        (identity.tenant_id, identity.tenant_id, row["document_id"]),
                    )
                    for table in (
                        "graph_projection",
                        "background_task",
                        "document_chunk",
                        "index_generation",
                        "document_revision",
                    ):
                        repository.query(
                            f"DELETE FROM {{schema}}.{table} WHERE tenant_id = %s "
                            "AND document_id = %s",
                            (identity.tenant_id, row["document_id"]),
                        )
                    repository.query(
                        "DELETE FROM {schema}.document WHERE tenant_id = %s AND id = %s "
                        "AND title = %s",
                        (identity.tenant_id, row["document_id"], filename),
                    )
            with document_object_storage(settings) as storage:
                for row in rows:
                    assert row["bucket"] == settings.minio_bucket
                    assert row["object_key"].startswith(settings.minio_object_prefix)
                    storage.client.remove_object(row["bucket"], row["object_key"])


def task_response(uploaded):
    _, _, body, client = uploaded
    response = client.get(f"/api/document-tasks/{body['task_id']}")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("running", [False, True])
def test_delete_cancels_processing_and_survives_reload(uploaded, running):
    settings, identity, body, client = uploaded
    queue = IngestionQueue(settings)
    lease = queue.claim("delete-test", int(body["task_id"])) if running else None
    with database_connection(settings) as connection:
        original = (
            UploadRepository(connection, settings.db_schema)
            .query(
                "SELECT bucket, object_key FROM {schema}.document_revision "
                "WHERE tenant_id = %s AND id = %s",
                (identity.tenant_id, int(body["revision_id"])),
            )
            .fetchone()
        )
    response = client.delete(f"/api/documents/{body['document_id']}")
    assert response.status_code == 204, response.text
    assert not response.content
    assert client.delete(f"/api/documents/{body['document_id']}").status_code == 404
    assert client.get(f"/api/document-tasks/{body['task_id']}").status_code == 404
    assert not any(
        row["document_id"] == body["document_id"]
        for row in client.get("/api/document-tasks").json()
    )
    assert client.post(f"/api/document-tasks/{body['task_id']}/retry").status_code == 404
    assert queue.claim("delete-test", int(body["task_id"])) is None
    if lease:
        with pytest.raises(LeaseLostError):
            queue.heartbeat(lease)
        worker = ingestion.IngestionWorker(settings)
        try:
            with pytest.raises(LeaseLostError):
                publish_chunks(worker.engine, lease, (), {})
        finally:
            worker.engine.dispose()
    assert_document_purged(settings, identity, body)
    with document_object_storage(settings) as storage:
        with pytest.raises(S3Error) as error:
            storage.client.stat_object(original["bucket"], original["object_key"])
        assert error.value.code == "NoSuchKey"


def assert_document_purged(settings, identity, body):
    with database_connection(settings) as connection:
        repository = UploadRepository(connection, settings.db_schema)
        for table in (
            "document",
            "document_revision",
            "document_chunk",
            "index_generation",
            "background_task",
            "graph_projection",
            "answer_citation",
        ):
            column = "id" if table == "document" else "document_id"
            count = repository.query(
                f"SELECT count(*) AS count FROM {{schema}}.{table} "
                f"WHERE tenant_id = %s AND {column} = %s",
                (identity.tenant_id, int(body["document_id"])),
            ).fetchone()
            assert count["count"] == 0, table
        for table in ("task_attempt", "outbox_event"):
            count = repository.query(
                f"SELECT count(*) AS count FROM {{schema}}.{table} "
                "WHERE tenant_id = %s AND task_id = %s",
                (identity.tenant_id, int(body["task_id"])),
            ).fetchone()
            assert count["count"] == 0, table


def test_delete_rejects_other_tenant_and_insufficient_clearance(uploaded):
    settings, identity, body, client = uploaded
    with database_connection(settings) as connection:
        repository = UploadRepository(connection, settings.db_schema)
        other = repository.identity("isolation_admin@zhiqing.test", "QA_ISOLATION")
        with pytest.raises(UploadError) as error:
            repository.delete_document(other, int(body["document_id"]))
        assert error.value.status_code == 404
        with pytest.raises(UploadError) as error:
            repository.delete_document(replace(identity, clearance=-1), int(body["document_id"]))
        assert error.value.status_code == 404
    assert task_response(uploaded)["document_status"] == "DRAFT"


@pytest.mark.asyncio
async def test_delete_published_document_unpublishes_index(uploaded, monkeypatch):
    settings, identity, body, client = uploaded
    monkeypatch.setattr(ingestion, "embed_chunks", fake_embeddings)
    worker = ingestion.IngestionWorker(settings)
    try:
        lease = worker.queue.claim(worker.worker_id, int(body["task_id"]))
        assert lease is not None
        await worker.process(lease)
        verify_publication(uploaded)
        assert client.delete(f"/api/documents/{body['document_id']}").status_code == 204
        assert_document_purged(settings, identity, body)
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            cleanup = repository.query(
                """SELECT event_type, payload FROM {schema}.outbox_event
                   WHERE tenant_id = %s AND aggregate_type = 'graph_cleanup'
                     AND aggregate_id = %s""",
                (identity.tenant_id, str(body["document_id"])),
            ).fetchone()
            assert cleanup is not None
            assert cleanup["event_type"] == "GRAPH_PURGE_DOCUMENT"
            assert settings.graph_namespace in cleanup["payload"]["namespaces"]
        assert client.get(f"/api/document-tasks/{body['task_id']}").status_code == 404
    finally:
        worker.engine.dispose()


def test_delete_storage_failure_is_visible_and_retryable(uploaded, monkeypatch):
    from zhiqing_rag.infrastructure.object_storage import DocumentObjectStorage

    settings, identity, body, client = uploaded
    remove = DocumentObjectStorage.remove

    def fail(*args):
        raise RuntimeError("storage-test-failure")

    monkeypatch.setattr(DocumentObjectStorage, "remove", fail)
    response = client.delete(f"/api/documents/{body['document_id']}")
    assert response.status_code == 503
    pending = task_response(uploaded)
    assert pending["document_status"] == "DELETED"
    assert pending["status"] == "CANCELLED"
    assert any(
        task["document_id"] == body["document_id"]
        for task in client.get("/api/document-tasks").json()
    )
    monkeypatch.setattr(DocumentObjectStorage, "remove", remove)
    assert client.delete(f"/api/documents/{body['document_id']}").status_code == 204
    assert_document_purged(settings, identity, body)


def test_upload_returns_durable_task_before_any_processing(uploaded):
    settings, identity, body, client = uploaded
    status = task_response(uploaded)
    assert status["status"] == "PENDING" and status["progress"] == 0
    assert status["chunk_count"] == 0 and status["document_status"] == "DRAFT"
    assert any(
        task["task_id"] == body["task_id"] for task in client.get("/api/document-tasks").json()
    )
    with database_connection(settings) as connection:
        repository = UploadRepository(connection, settings.db_schema)
        other = repository.identity("isolation_admin@zhiqing.test", "QA_ISOLATION")
        with pytest.raises(UploadError) as error:
            repository.task_status(other, int(body["task_id"]))
        assert error.value.status_code == 404


async def fake_embeddings(chunks, *, settings, stats, **kwargs):
    stats.model_chunks = len(chunks)
    stats.model_batches = 1
    return tuple(EmbeddedChunk(chunk, (0.5,) * 1024, settings.embedding_model) for chunk in chunks)


def verify_publication(uploaded):
    settings, identity, body, _ = uploaded
    status = task_response(uploaded)
    assert status["status"] == "SUCCEEDED", status
    assert status["document_status"] == "PUBLISHED" and status["progress"] == 100
    assert status["chunk_count"] > 0
    with database_connection(settings) as connection:
        repository = UploadRepository(connection, settings.db_schema)
        row = repository.query(
            """SELECT doc.active_generation_id, gen.status,
                      (SELECT count(*) FROM {schema}.document_chunk chunk
                       WHERE chunk.tenant_id = doc.tenant_id AND chunk.generation_id = gen.id
                         AND vector_dims(chunk.embedding) = 1024) AS vector_count
               FROM {schema}.document doc JOIN {schema}.index_generation gen
                 ON gen.tenant_id = doc.tenant_id AND gen.id = doc.active_generation_id
               WHERE doc.tenant_id = %s AND doc.id = %s""",
            (identity.tenant_id, int(body["document_id"])),
        ).fetchone()
        assert row["active_generation_id"] == int(body["generation_id"])
        assert row["status"] == "READY" and row["vector_count"] == status["chunk_count"]
        graph = repository.query(
            """SELECT task.status AS task_status, projection.status AS projection_status
               FROM {schema}.background_task task
               JOIN {schema}.graph_projection projection ON projection.tenant_id = task.tenant_id
                 AND projection.generation_id = task.generation_id
               WHERE task.tenant_id = %s AND task.document_id = %s
                 AND task.task_type = 'GRAPH' AND projection.projection_version = 1""",
            (identity.tenant_id, int(body["document_id"])),
        ).fetchone()
        assert graph is not None
        assert graph["task_status"] in {"PENDING", "RUNNING", "RETRY_WAIT", "SUCCEEDED", "FAILED"}
        assert graph["projection_status"] in {"BUILDING", "READY", "FAILED"}


@pytest.mark.asyncio
async def test_worker_download_parse_embed_store_publish_and_no_duplicate(uploaded, monkeypatch):
    settings, _, body, _ = uploaded
    monkeypatch.setattr(ingestion, "embed_chunks", fake_embeddings)
    worker = ingestion.IngestionWorker(settings)
    try:
        lease = worker.queue.claim(worker.worker_id, int(body["task_id"]))
        assert lease is not None
        assert worker.queue.claim("other-worker", lease.task_id) is None
        await worker.process(lease)
        verify_publication(uploaded)
        assert worker.queue.claim(worker.worker_id, lease.task_id) is None
        with pytest.raises(LeaseLostError):
            publish_chunks(worker.engine, lease, (), {})
        verify_publication(uploaded)
    finally:
        worker.engine.dispose()


@pytest.mark.asyncio
async def test_backfill_existing_published_chunks_is_idempotent(uploaded, monkeypatch):
    settings, identity, body, _ = uploaded
    monkeypatch.setattr(ingestion, "embed_chunks", fake_embeddings)
    worker = ingestion.IngestionWorker(settings)
    try:
        lease = worker.queue.claim(worker.worker_id, int(body["task_id"]))
        if lease is None:
            pytest.skip("外部文档 Worker 已领取临时测试任务")
        await worker.process(lease)
        verify_publication(uploaded)
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            repository.query(
                "DELETE FROM {schema}.graph_projection WHERE tenant_id = %s AND generation_id = %s",
                (identity.tenant_id, int(body["generation_id"])),
            )
            repository.query(
                "DELETE FROM {schema}.background_task WHERE tenant_id = %s "
                "AND generation_id = %s AND task_type = 'GRAPH'",
                (identity.tenant_id, int(body["generation_id"])),
            )
        candidates = backfill_graph.list_missing(
            worker.engine, tenant_id=identity.tenant_id, document_id=int(body["document_id"])
        )
        assert len(candidates) == 1
        assert candidates[0].generation_id == int(body["generation_id"])
        first = backfill_graph.enqueue_candidate(worker.engine, settings, candidates[0])
        assert first is not None
        assert backfill_graph.enqueue_candidate(worker.engine, settings, candidates[0]) is None
        assert (
            backfill_graph.list_missing(
                worker.engine, tenant_id=identity.tenant_id, document_id=int(body["document_id"])
            )
            == []
        )
        assert first in backfill_graph.list_pending(
            worker.engine, tenant_id=identity.tenant_id, document_id=int(body["document_id"])
        )
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            repository.query(
                """UPDATE {schema}.background_task SET status = 'FAILED', stage = 'FAILED',
                   attempt_count = 1, max_attempts = 1, finished_at = clock_timestamp(),
                   error_code = 'GRAPH_PROCESSING_FAILED' WHERE tenant_id = %s AND id = %s""",
                (identity.tenant_id, first),
            )
            repository.query(
                """UPDATE {schema}.graph_projection SET status = 'FAILED',
                   error_code = 'GRAPH_PROCESSING_FAILED'
                   WHERE tenant_id = %s AND generation_id = %s""",
                (identity.tenant_id, int(body["generation_id"])),
            )
        assert first in backfill_graph.list_pending(
            worker.engine,
            tenant_id=identity.tenant_id,
            document_id=int(body["document_id"]),
            statuses=("FAILED",),
        )
        assert backfill_graph.requeue_failed(worker.engine, settings, first)
        assert first in backfill_graph.list_pending(
            worker.engine, tenant_id=identity.tenant_id, document_id=int(body["document_id"])
        )
    finally:
        worker.engine.dispose()


@pytest.mark.asyncio
async def test_updated_document_retires_old_graph_after_new_publication(uploaded, monkeypatch):
    settings, identity, body, client = uploaded
    monkeypatch.setattr(ingestion, "embed_chunks", fake_embeddings)
    worker = ingestion.IngestionWorker(settings)
    try:
        first = worker.queue.claim(worker.worker_id, int(body["task_id"]))
        if first is None:
            pytest.skip("外部文档 Worker 已领取临时测试任务")
        await worker.process(first)
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            original = repository.query(
                """SELECT original_filename FROM {schema}.document_revision
                   WHERE tenant_id = %s AND id = %s""",
                (identity.tenant_id, int(body["revision_id"])),
            ).fetchone()
        updated = client.post(
            f"/api/documents/{body['document_id']}/revisions",
            files={
                "file": (
                    original["original_filename"],
                    b"# Updated graph revision",
                    "text/markdown",
                )
            },
            data={"expected_revision_id": body["revision_id"]},
        )
        assert updated.status_code == 202, updated.text
        newer = updated.json()
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            active = repository.query(
                "SELECT active_generation_id FROM {schema}.document "
                "WHERE tenant_id = %s AND id = %s",
                (identity.tenant_id, int(body["document_id"])),
            ).fetchone()
            assert active["active_generation_id"] == int(body["generation_id"])
        second = worker.queue.claim(worker.worker_id, int(newer["task_id"]))
        if second is None:
            pytest.skip("外部文档 Worker 已领取新修订测试任务")
        await worker.process(second)
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            active = repository.query(
                "SELECT active_generation_id FROM {schema}.document "
                "WHERE tenant_id = %s AND id = %s",
                (identity.tenant_id, int(body["document_id"])),
            ).fetchone()
            assert active["active_generation_id"] == int(newer["generation_id"])
            old_projection = repository.query(
                """SELECT status FROM {schema}.graph_projection
                   WHERE tenant_id = %s AND generation_id = %s""",
                (identity.tenant_id, int(body["generation_id"])),
            ).fetchone()
            assert old_projection["status"] == "STALE"
            cleanup = repository.query(
                """SELECT payload FROM {schema}.outbox_event
                   WHERE tenant_id = %s AND event_type = 'GRAPH_PURGE_GENERATION'
                     AND deduplication_key = %s""",
                (identity.tenant_id, f"graph:purge:{body['generation_id']}:v1"),
            ).fetchone()
            assert cleanup is not None
            assert cleanup["payload"]["generation_id"] == int(body["generation_id"])
        with Session(worker.engine) as session:
            visible = active_graph_projection(session, identity.tenant_id, int(body["document_id"]))
            assert visible is None or visible.generation_id == int(newer["generation_id"])
    finally:
        worker.engine.dispose()


@pytest.mark.asyncio
async def test_graph_queued_while_embeddings_are_still_running(uploaded, monkeypatch):
    settings, identity, body, _ = uploaded
    entered = asyncio.Event()
    release = asyncio.Event()

    async def waiting_embeddings(chunks, *, settings, stats, **kwargs):
        entered.set()
        await release.wait()
        return await fake_embeddings(chunks, settings=settings, stats=stats)

    monkeypatch.setattr(ingestion, "embed_chunks", waiting_embeddings)
    worker = ingestion.IngestionWorker(settings)
    lease = worker.queue.claim(worker.worker_id, int(body["task_id"]))
    processing = asyncio.create_task(worker.process(lease))
    try:
        await asyncio.wait_for(entered.wait(), 30)
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            row = repository.query(
                """SELECT task.stage, doc.status AS document_status,
                          (SELECT count(*) FROM {schema}.document_chunk chunk
                           WHERE chunk.tenant_id = task.tenant_id
                             AND chunk.generation_id = task.generation_id) AS chunks,
                          (SELECT count(*) FROM {schema}.document_chunk chunk
                           WHERE chunk.tenant_id = task.tenant_id
                             AND chunk.generation_id = task.generation_id
                             AND chunk.embedding IS NOT NULL) AS embedded,
                          (SELECT count(*) FROM {schema}.background_task graph
                           WHERE graph.tenant_id = task.tenant_id
                             AND graph.generation_id = task.generation_id
                             AND graph.task_type = 'GRAPH') AS graph_tasks,
                          (SELECT min(graph.id) FROM {schema}.background_task graph
                           WHERE graph.tenant_id = task.tenant_id
                             AND graph.generation_id = task.generation_id
                             AND graph.task_type = 'GRAPH') AS graph_task_id
                   FROM {schema}.background_task task
                   JOIN {schema}.document doc ON doc.tenant_id = task.tenant_id
                     AND doc.id = task.document_id
                   WHERE task.tenant_id = %s AND task.id = %s""",
                (identity.tenant_id, int(body["task_id"])),
            ).fetchone()
        assert row["stage"] == "EMBEDDING"
        assert row["document_status"] == "DRAFT"
        assert row["chunks"] > 0 and row["embedded"] == 0
        assert row["graph_tasks"] == 1

        class FakeGraphStore:
            def __init__(self, settings):
                pass

            async def prepare(self):
                pass

            async def write_chunk(self, scope, chunk, extraction):
                return set(), set()

            async def close(self):
                pass

        async def empty_extraction(settings, content):
            return Extraction(entities=[], relations=[])

        monkeypatch.setattr(graph, "GraphStore", FakeGraphStore)
        monkeypatch.setattr(graph, "extract_chunk", empty_extraction)
        graph_settings = settings.model_copy(
            update={
                "neo4j_enabled": True,
                "deepseek_api_key": SecretStr("test"),
            }
        )
        graph_worker = graph.GraphWorker(graph_settings)
        saved_extraction = graph_worker.queue.save_extraction
        cache_checked = []

        def save_and_check(lease, chunk, extraction):
            saved_extraction(lease, chunk, extraction)
            assert graph_worker.queue.cached_extraction(lease, chunk) == extraction
            assert (
                graph_worker.queue.cached_extraction(
                    lease, {**chunk, "id": chunk["id"] + 100000, "chunk_index": 99}
                )
                == extraction
            )
            assert (
                graph_worker.queue.cached_extraction(
                    lease, {**chunk, "content": chunk["content"] + "修改"}
                )
                is None
            )
            other_model_queue = GraphQueue(
                graph_worker.engine,
                graph_settings.model_copy(update={"graph_extraction_model": "other-model"}),
            )
            assert other_model_queue.cached_extraction(lease, chunk) is None
            cache_checked.append(chunk["id"])

        graph_worker.queue.save_extraction = save_and_check
        await graph_worker.run(once=True, task_id=row["graph_task_id"])
        assert len(cache_checked) == row["chunks"]
        with database_connection(settings) as connection:
            projection = (
                UploadRepository(connection, settings.db_schema)
                .query(
                    """SELECT status FROM {schema}.graph_projection
                   WHERE tenant_id = %s AND generation_id = %s""",
                    (identity.tenant_id, int(body["generation_id"])),
                )
                .fetchone()
            )
        assert projection["status"] == "READY"
    finally:
        release.set()
        await processing
        worker.engine.dispose()
    verify_publication(uploaded)


def expire(settings, lease):
    with database_connection(settings) as connection:
        UploadRepository(connection, settings.db_schema).query(
            """UPDATE {schema}.background_task SET lease_expires_at =
               clock_timestamp() - interval '1 second'
               WHERE tenant_id = %s AND id = %s""",
            (lease.tenant_id, lease.task_id),
        )


def test_expired_lease_recovered_and_old_worker_cannot_write(uploaded):
    settings, identity, body, client = uploaded
    queue = IngestionQueue(settings)
    first = queue.claim("worker-first", int(body["task_id"]))
    expire(settings, first)
    second = queue.claim("worker-second", first.task_id)
    assert second.epoch == first.epoch + 1
    with pytest.raises(LeaseLostError):
        queue.heartbeat(first, "EMBEDDING", 50)
    queue.fail(first, "INVALID_EMBEDDING", False, {})
    assert task_response(uploaded)["status"] == "RUNNING"
    queue.fail(second, "PARSE_FAILED", False, {})
    assert task_response(uploaded)["status"] == "FAILED"
    response = client.post(f"/api/document-tasks/{body['task_id']}/retry")
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "PENDING"
    third = queue.claim("worker-third", first.task_id)
    assert third.epoch == second.epoch + 1
    with database_connection(settings) as connection:
        rows = (
            UploadRepository(connection, settings.db_schema)
            .query(
                """SELECT attempt_no, status FROM {schema}.task_attempt
               WHERE tenant_id = %s AND task_id = %s ORDER BY attempt_no""",
                (identity.tenant_id, first.task_id),
            )
            .fetchall()
        )
        assert [row["attempt_no"] for row in rows] == [1, 2, 3]
        assert [row["status"] for row in rows] == ["LOST", "FAILED", "RUNNING"]
    queue.fail(third, "PARSE_FAILED", False, {})


def test_transient_failure_waits_then_retries_without_marking_ready(uploaded):
    settings, _, body, _ = uploaded
    queue = IngestionQueue(settings)
    lease = queue.claim("retry-worker", int(body["task_id"]))
    queue.fail(lease, "PROCESSING_UNAVAILABLE", True, {})
    status = task_response(uploaded)
    assert status["status"] == "RETRY_WAIT" and status["chunk_count"] == 0
    assert status["document_status"] == "DRAFT"
    assert queue.claim("retry-worker", lease.task_id) is None


def test_expired_last_attempt_becomes_failed(uploaded):
    settings, _, body, _ = uploaded
    queue = IngestionQueue(settings)
    lease = queue.claim("crashed-worker", int(body["task_id"]))
    with database_connection(settings) as connection:
        UploadRepository(connection, settings.db_schema).query(
            "UPDATE {schema}.background_task SET max_attempts = 1 WHERE tenant_id = %s AND id = %s",
            (lease.tenant_id, lease.task_id),
        )
    expire(settings, lease)
    assert queue.claim("replacement-worker", lease.task_id) is None
    assert task_response(uploaded)["status"] == "FAILED"
    assert task_response(uploaded)["error_code"] == "ATTEMPTS_EXHAUSTED"


@pytest.mark.asyncio
async def test_stale_revision_never_embeds_or_publishes(uploaded, monkeypatch):
    settings, identity, body, _ = uploaded

    async def forbidden_embedding(*args, **kwargs):
        pytest.fail("旧修订不能调用模型")

    monkeypatch.setattr(ingestion, "embed_chunks", forbidden_embedding)
    with database_connection(settings) as connection:
        UploadRepository(connection, settings.db_schema).query(
            "UPDATE {schema}.document SET desired_revision_id = NULL "
            "WHERE tenant_id = %s AND id = %s",
            (identity.tenant_id, int(body["document_id"])),
        )
    worker = ingestion.IngestionWorker(settings)
    try:
        lease = worker.queue.claim(worker.worker_id, int(body["task_id"]))
        await worker.process(lease)
        status = task_response(uploaded)
        assert status["status"] == "FAILED"
        assert status["error_code"] == "REVISION_UNAVAILABLE"
        assert status["chunk_count"] == 0 and status["document_status"] == "DRAFT"
    finally:
        worker.engine.dispose()


@pytest.mark.asyncio
async def test_partial_vector_write_rolls_back_before_retry(uploaded, monkeypatch):
    settings, identity, body, _ = uploaded
    monkeypatch.setattr(ingestion, "embed_chunks", fake_embeddings)
    original_save = publication.complete_embedded_chunks

    def fail_after_insert(session, scope, chunks):
        original_save(session, scope, chunks)
        raise RuntimeError("模拟分块写入后发布失败")

    monkeypatch.setattr(publication, "complete_embedded_chunks", fail_after_insert)
    worker = ingestion.IngestionWorker(settings)
    try:
        lease = worker.queue.claim(worker.worker_id, int(body["task_id"]))
        await worker.process(lease)
        assert task_response(uploaded)["status"] == "RETRY_WAIT"
        with database_connection(settings) as connection:
            row = (
                UploadRepository(connection, settings.db_schema)
                .query(
                    """SELECT status, (SELECT count(*) FROM {schema}.document_chunk
                   WHERE tenant_id = %s AND generation_id = %s) AS chunk_count,
                   (SELECT count(*) FROM {schema}.document_chunk WHERE tenant_id = %s
                     AND generation_id = %s AND embedding IS NOT NULL) AS embedded_count
                   FROM {schema}.index_generation WHERE tenant_id = %s AND id = %s""",
                    (
                        identity.tenant_id,
                        int(body["generation_id"]),
                        identity.tenant_id,
                        int(body["generation_id"]),
                        identity.tenant_id,
                        int(body["generation_id"]),
                    ),
                )
                .fetchone()
            )
            assert row["status"] == "BUILDING" and row["chunk_count"] > 0
            assert row["embedded_count"] == 0
            graph_task = (
                UploadRepository(connection, settings.db_schema)
                .query(
                    """SELECT status FROM {schema}.background_task
                   WHERE tenant_id = %s AND generation_id = %s AND task_type = 'GRAPH'""",
                    (identity.tenant_id, int(body["generation_id"])),
                )
                .fetchone()
            )
            assert graph_task is not None and graph_task["status"] != "CANCELLED"
    finally:
        worker.engine.dispose()


@pytest.mark.skipif(
    os.getenv("RUN_INGESTION_MODEL_INTEGRATION") != "1", reason="需显式允许调用真实向量模型"
)
@pytest.mark.asyncio
async def test_real_embedding_model_pipeline(uploaded):
    _, _, body, _ = uploaded
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "zhiqing_rag.workers.ingestion",
        "--once",
        "--task-id",
        body["task_id"],
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, errors = await asyncio.wait_for(process.communicate(), timeout=150)
        assert process.returncode == 0, errors.decode(errors="replace")
        verify_publication(uploaded)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
