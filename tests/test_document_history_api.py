from datetime import datetime

import pytest
from fastapi import HTTPException

from zq_rag_app.api import document as document_api
from zq_rag_app.core.security import DEFAULT_ADMIN_USER, UserContext
from zq_rag_app.models.document import Document, DocumentVersion
from zq_rag_app.services.permission_service import PermissionLevel


class _FakeScalarResult:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class _HistorySession:
    def __init__(self, document, versions, active_task=None):
        self.document = document
        self.versions = versions
        self.statement = None
        self.scalar_values = iter([document, active_task])

    async def scalar(self, statement):
        self.statement = statement
        return next(self.scalar_values)

    async def scalars(self, statement):
        self.statement = statement
        return _FakeScalarResult(self.versions)


def _document() -> Document:
    return Document(
        id=7,
        kb_id=2,
        file_name="当前版.txt",
        file_type="TXT",
        file_size=7,
        minio_path="rag-documents/current.txt",
        status="DONE",
        error_msg=None,
        chunk_count=2,
        token_count=10,
        version=2,
        uploaded_by=1,
        uploaded_at=datetime(2026, 9, 11, 9, 0, 0),
        indexed_at=datetime(2026, 9, 11, 9, 1, 0),
        is_deleted=False,
    )


@pytest.mark.asyncio
async def test_history_marks_only_ready_non_current_versions_as_restorable():
    current = DocumentVersion(
        id=12,
        doc_id=7,
        version=2,
        file_name="当前版.txt",
        file_type="TXT",
        file_size=7,
        file_hash="b" * 64,
        minio_path="rag-documents/current.txt",
        operation_type="UPDATE",
        source_version=None,
        status="READY",
        uploaded_by=1,
        created_at=datetime(2026, 9, 11, 9, 0, 0),
        indexed_at=datetime(2026, 9, 11, 9, 1, 0),
        error_msg=None,
    )
    history = DocumentVersion(
        id=11,
        doc_id=7,
        version=1,
        file_name="历史版.txt",
        file_type="TXT",
        file_size=5,
        file_hash="a" * 64,
        minio_path="rag-documents/history.txt",
        operation_type="UPLOAD",
        source_version=None,
        status="READY",
        uploaded_by=1,
        created_at=datetime(2026, 9, 10, 9, 0, 0),
        indexed_at=datetime(2026, 9, 10, 9, 1, 0),
        error_msg=None,
    )
    session = _HistorySession(_document(), [current, history])

    response = await document_api.get_document_versions(
        doc_id=7,
        current_user=DEFAULT_ADMIN_USER,
        session=session,
    )

    assert response.current_version == 2
    assert response.items[0].is_current is True
    assert response.items[0].can_restore is False
    assert response.items[1].is_current is False
    assert response.items[1].can_restore is True


@pytest.mark.asyncio
async def test_history_disables_restore_while_another_task_is_active():
    history = DocumentVersion(
        id=11,
        doc_id=7,
        version=1,
        file_name="历史版.txt",
        file_type="TXT",
        file_size=5,
        file_hash="a" * 64,
        minio_path="rag-documents/history.txt",
        operation_type="UPLOAD",
        source_version=None,
        status="READY",
        uploaded_by=1,
        created_at=datetime(2026, 9, 10, 9, 0, 0),
        indexed_at=datetime(2026, 9, 10, 9, 1, 0),
        error_msg=None,
    )
    session = _HistorySession(_document(), [history], active_task=99)

    response = await document_api.get_document_versions(
        doc_id=7,
        current_user=DEFAULT_ADMIN_USER,
        session=session,
    )

    assert response.items[0].can_restore is False


@pytest.mark.asyncio
async def test_restore_requires_write_permission(monkeypatch):
    async def deny_permission(*args, **kwargs):
        del args, kwargs
        return False

    monkeypatch.setattr(
        document_api,
        "has_knowledge_base_permission",
        deny_permission,
    )
    user = UserContext(
        user_id=9,
        username="reader",
        department_id="DEV",
        role="USER",
    )
    session = _HistorySession(_document(), [])

    with pytest.raises(HTTPException) as exc_info:
        await document_api._require_document_permission(
            session,
            doc_id=7,
            current_user=user,
            required=PermissionLevel.WRITE,
        )

    assert exc_info.value.status_code == 403
