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
        with TestClient(create_app(), client=("127.0.0.1", 54321)) as client:
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
