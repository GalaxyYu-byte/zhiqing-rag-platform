"""显式启用的 PostgreSQL/MinIO 上传实测，完成后清理本测试生成的记录和对象。"""

import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.infrastructure.object_storage import document_object_storage
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.documents.repository import UploadRepository
from zhiqing_rag.modules.documents.router import get_upload_identity


def create_upload_test_app():
    """存储专项实测显式注入数据库验证的身份；完整登录另由认证集成测试覆盖。"""
    settings = get_settings()
    with database_connection(settings) as connection:
        identity = UploadRepository(connection, settings.db_schema).identity(
            settings.dev_upload_user_email, settings.dev_upload_tenant_code
        )
    app = create_app()
    app.dependency_overrides[get_upload_identity] = lambda: identity
    return app


@pytest.mark.skipif(
    os.getenv("RUN_UPLOAD_INTEGRATION") != "1", reason="需要显式启用真实数据库与对象存储实测"
)
def test_upload_to_postgres_and_minio():
    settings = get_settings()
    filename = f"upload-integration-{uuid4().hex}.md"
    content = "# 上传接口联调\n\n实际校验原文件保存和数据库草稿修订。\n".encode()
    with database_connection(settings) as connection:
        repository = UploadRepository(connection, settings.db_schema)
        identity = repository.identity(
            settings.dev_upload_user_email, settings.dev_upload_tenant_code
        )
        bases = repository.writable_bases(identity)
        assert bases, "开发身份必须拥有至少一个可写知识库"
    try:
        with TestClient(create_upload_test_app(), client=("127.0.0.1", 54321)) as client:
            options = client.get("/api/knowledge-bases")
            assert options.status_code == 200, options.text
            response = client.post(
                "/api/documents/upload",
                files={"file": (filename, content, "text/markdown")},
                data={"knowledge_base_id": bases[0].id},
            )
            assert response.status_code == 202, response.text
            body = response.json()
            with database_connection(settings) as connection:
                repository = UploadRepository(connection, settings.db_schema)
                row = repository.query(
                    """SELECT d.status, d.metadata, d.desired_revision_id, r.*
                       FROM {schema}.document d JOIN {schema}.document_revision r
                         ON r.tenant_id = d.tenant_id AND r.document_id = d.id
                       WHERE d.tenant_id = %s AND d.id = %s AND r.id = %s""",
                    (identity.tenant_id, int(body["document_id"]), int(body["revision_id"])),
                ).fetchone()
                assert row["status"] == "DRAFT"
                assert row["desired_revision_id"] == int(body["revision_id"])
                assert row["metadata"]["chunk_config"]["max_tokens"] == settings.upload_chunk_size
                assert (
                    row["metadata"]["chunk_config"]["overlap_tokens"]
                    == settings.upload_chunk_overlap
                )
                assert row["file_sha256"] == body["file_sha256"]
                assert row["original_filename"] == filename
            with document_object_storage(settings) as storage:
                stored = storage.client.get_object(row["bucket"], row["object_key"])
                try:
                    assert stored.read() == content
                finally:
                    stored.close()
                    stored.release_conn()
    finally:
        # UUID 文件名 + 当前租户限制范围，只清理本测试创建的记录。
        with database_connection(settings) as connection:
            repository = UploadRepository(connection, settings.db_schema)
            rows = repository.query(
                """SELECT id, document_id, bucket, object_key FROM {schema}.document_revision
                   WHERE tenant_id = %s AND original_filename = %s""",
                (identity.tenant_id, filename),
            ).fetchall()
            for row in rows:
                repository.query(
                    """UPDATE {schema}.document SET desired_revision_id = NULL,
                       active_generation_id = NULL, status = 'DRAFT'
                       WHERE tenant_id = %s AND id = %s AND title = %s""",
                    (identity.tenant_id, row["document_id"], filename),
                )
                repository.query(
                    "DELETE FROM {schema}.task_attempt WHERE tenant_id = %s AND task_id IN "
                    "(SELECT id FROM {schema}.background_task WHERE tenant_id = %s "
                    "AND document_id = %s)",
                    (identity.tenant_id, identity.tenant_id, row["document_id"]),
                )
                repository.query(
                    "DELETE FROM {schema}.background_task WHERE tenant_id = %s "
                    "AND document_id = %s",
                    (identity.tenant_id, row["document_id"]),
                )
                repository.query(
                    "DELETE FROM {schema}.document_chunk WHERE tenant_id = %s AND document_id = %s",
                    (identity.tenant_id, row["document_id"]),
                )
                repository.query(
                    "DELETE FROM {schema}.index_generation WHERE tenant_id = %s "
                    "AND document_id = %s",
                    (identity.tenant_id, row["document_id"]),
                )
                repository.query(
                    "DELETE FROM {schema}.document_revision WHERE tenant_id = %s AND id = %s",
                    (identity.tenant_id, row["id"]),
                )
                repository.query(
                    "DELETE FROM {schema}.document WHERE tenant_id = %s AND id = %s AND title = %s",
                    (identity.tenant_id, row["document_id"], filename),
                )
        with document_object_storage(settings) as storage:
            for row in rows:
                assert row["object_key"].startswith(settings.minio_object_prefix)
                storage.client.remove_object(row["bucket"], row["object_key"])


@pytest.mark.skipif(
    os.getenv("RUN_UPLOAD_INTEGRATION") != "1", reason="需要显式启用真实数据库与对象存储实测"
)
def test_update_creates_revision_and_cancels_superseded_task():
    filename = f"update-integration-{uuid4().hex}.md"
    with TestClient(create_upload_test_app(), client=("127.0.0.1", 54321)) as client:
        document_id = None
        try:
            with database_connection(get_settings()) as connection:
                repository = UploadRepository(connection, get_settings().db_schema)
                identity = repository.identity(
                    get_settings().dev_upload_user_email,
                    get_settings().dev_upload_tenant_code,
                )
                kb_id = repository.writable_bases(identity)[0].id
            first = client.post(
                "/api/documents/upload",
                files={"file": (filename, b"# First version", "text/markdown")},
                data={"knowledge_base_id": kb_id},
            )
            assert first.status_code == 202, first.text
            old = first.json()
            document_id = old["document_id"]
            updated = client.post(
                f"/api/documents/{document_id}/revisions",
                files={"file": (filename, b"# Second version", "text/markdown")},
                data={"expected_revision_id": old["revision_id"]},
            )
            assert updated.status_code == 202, updated.text
            new = updated.json()
            assert new["document_id"] == document_id
            assert new["revision_id"] != old["revision_id"]
            with database_connection(get_settings()) as connection:
                repository = UploadRepository(connection, get_settings().db_schema)
                rows = repository.query(
                    """SELECT revision_no, file_sha256 FROM {schema}.document_revision
                       WHERE tenant_id = %s AND document_id = %s ORDER BY revision_no""",
                    (identity.tenant_id, int(document_id)),
                ).fetchall()
                assert [row["revision_no"] for row in rows] == [1, 2]
                assert rows[0]["file_sha256"] != rows[1]["file_sha256"]
                document = repository.query(
                    """SELECT desired_revision_id FROM {schema}.document
                       WHERE tenant_id = %s AND id = %s""",
                    (identity.tenant_id, int(document_id)),
                ).fetchone()
                assert document["desired_revision_id"] == int(new["revision_id"])
                old_task = repository.query(
                    """SELECT status FROM {schema}.background_task
                       WHERE tenant_id = %s AND id = %s""",
                    (identity.tenant_id, int(old["task_id"])),
                ).fetchone()
                assert old_task["status"] in {"CANCELLED", "SUCCEEDED"}
            stale = client.post(
                f"/api/documents/{document_id}/revisions",
                files={"file": (filename, b"# Third version", "text/markdown")},
                data={"expected_revision_id": old["revision_id"]},
            )
            assert stale.status_code == 409
            unchanged = client.post(
                f"/api/documents/{document_id}/revisions",
                files={"file": (filename, b"# Second version", "text/markdown")},
                data={"expected_revision_id": new["revision_id"]},
            )
            assert unchanged.status_code == 409
            assert unchanged.json()["detail"]["code"] == "FILE_UNCHANGED"
            restored = client.post(
                f"/api/documents/{document_id}/revisions",
                files={"file": (filename, b"# First version", "text/markdown")},
                data={"expected_revision_id": new["revision_id"]},
            )
            assert restored.status_code == 202, restored.text
            assert restored.json()["document_id"] == document_id
            page = client.get("/api/document-tasks/page", params={"search": filename})
            assert page.status_code == 200, page.text
            assert page.json()["total"] == 1
            assert page.json()["items"][0]["revision_no"] == 3
            history = client.get(f"/api/documents/{document_id}/revisions")
            assert history.status_code == 200, history.text
            assert [item["revision_no"] for item in history.json()] == [3, 2, 1]
        finally:
            if document_id is not None:
                assert client.delete(f"/api/documents/{document_id}").status_code == 204
