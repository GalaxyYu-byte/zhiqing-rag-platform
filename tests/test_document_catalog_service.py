from datetime import datetime
import hashlib
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from zq_rag_app.services.document_catalog_service import (
    DocumentUpdateConflictError,
    DocumentUploadValidationError,
    create_document_restore,
    create_document_update,
    create_uploaded_documents,
)
from zq_rag_app.models.document import Document, DocumentVersion


class _FakeObjectStore:
    def __init__(self):
        self.objects = {}
        self.removed = []

    def put_object(self, bucket, object_name, stream, length, **kwargs):
        self.objects[(bucket, object_name)] = stream.read(length)

    def remove_object(self, bucket, object_name):
        self.removed.append((bucket, object_name))
        self.objects.pop((bucket, object_name), None)


class _FakeSession:
    def __init__(self):
        self.documents = []
        self.versions = []
        self.rolled_back = False

    async def scalar(self, statement):
        return SimpleNamespace(id=1, is_deleted=False)

    def add_all(self, documents):
        for item in documents:
            if isinstance(item, Document):
                self.documents.append(item)
            elif isinstance(item, DocumentVersion):
                self.versions.append(item)

    async def flush(self):
        for document in self.documents:
            document.id = 100 + self.documents.index(document)

    async def commit(self):
        return None

    async def rollback(self):
        self.rolled_back = True

    async def refresh(self, document):
        document.id = 100 + self.documents.index(document)
        document.uploaded_at = datetime(2026, 9, 9, 10, 0, 0)
        document.indexed_at = None
        document.error_msg = None


class _FakeUpdateSession:
    def __init__(self, *, current_hash: str | None = None):
        self.document = Document(
            id=7,
            kb_id=2,
            file_name="旧版.txt",
            file_type="TXT",
            file_size=3,
            minio_path="rag-documents/kb/2/old.txt",
            status="DONE",
            chunk_count=1,
            token_count=2,
            version=1,
            uploaded_by=3,
            is_deleted=False,
        )
        self.current_version = DocumentVersion(
            id=10,
            doc_id=7,
            version=1,
            file_name="旧版.txt",
            file_type="TXT",
            file_size=3,
            file_hash=current_hash,
            minio_path="rag-documents/kb/2/old.txt",
            status="READY",
            uploaded_by=3,
        )
        self.scalar_values = iter([self.document, None, self.current_version, 1])
        self.added = []
        self.committed = False
        self.rolled_back = False

    async def scalar(self, statement):
        del statement
        return next(self.scalar_values)

    def add_all(self, values):
        self.added.extend(values)

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True

    async def refresh(self, value):
        if isinstance(value, DocumentVersion):
            value.id = 11
            value.created_at = datetime(2026, 9, 11, 10, 0, 0)
        else:
            value.id = 21
            value.created_at = datetime(2026, 9, 11, 10, 0, 0)
            value.started_at = None
            value.finished_at = None
            value.heartbeat_at = None
            value.lease_expires_at = None
            value.error_msg = None


class _FakeRestoreSession:
    def __init__(self, *, source_status="READY", active_task=None):
        self.document = Document(
            id=7,
            kb_id=2,
            file_name="当前版.txt",
            file_type="TXT",
            file_size=7,
            minio_path="rag-documents/kb/2/current.txt",
            status="DONE",
            chunk_count=2,
            token_count=10,
            version=2,
            uploaded_by=3,
            is_deleted=False,
        )
        self.source = DocumentVersion(
            id=10,
            doc_id=7,
            version=1,
            file_name="历史版.txt",
            file_type="TXT",
            file_size=5,
            file_hash="a" * 64,
            minio_path="rag-documents/kb/2/history.txt",
            operation_type="UPLOAD",
            status=source_status,
            uploaded_by=3,
        )
        self.scalar_values = iter([self.document, active_task, self.source, 2])
        self.added = []
        self.committed = False
        self.rolled_back = False

    async def scalar(self, statement):
        del statement
        return next(self.scalar_values)

    def add_all(self, values):
        self.added.extend(values)

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True

    async def refresh(self, value):
        value.id = 20 if isinstance(value, DocumentVersion) else 30
        value.created_at = datetime(2026, 9, 11, 12, 0, 0)
        if not isinstance(value, DocumentVersion):
            value.started_at = None
            value.finished_at = None
            value.heartbeat_at = None
            value.lease_expires_at = None
            value.error_msg = None


@pytest.mark.asyncio
async def test_upload_stores_object_and_creates_document():
    session = _FakeSession()
    store = _FakeObjectStore()
    upload = UploadFile(
        filename="员工手册.pdf",
        file=BytesIO(b"fake pdf content"),
        size=16,
        headers={"content-type": "application/pdf"},
    )

    documents = await create_uploaded_documents(
        session,
        files=[upload],
        kb_id=1,
        uploaded_by=9,
        object_store=store,
    )

    assert len(documents) == 1
    assert documents[0].file_name == "员工手册.pdf"
    assert documents[0].file_type == "PDF"
    assert documents[0].status == "PENDING"
    assert documents[0].minio_path.startswith("rag-documents/kb/1/")
    assert next(iter(store.objects.values())) == b"fake pdf content"
    assert len(session.versions) == 1
    assert session.versions[0].doc_id == documents[0].id
    assert len(session.versions[0].file_hash) == 64


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_file_before_minio_write():
    session = _FakeSession()
    store = _FakeObjectStore()
    upload = UploadFile(
        filename="payload.exe",
        file=BytesIO(b"unsafe"),
        size=6,
    )

    with pytest.raises(DocumentUploadValidationError, match="不支持"):
        await create_uploaded_documents(
            session,
            files=[upload],
            kb_id=1,
            uploaded_by=9,
            object_store=store,
        )

    assert store.objects == {}


@pytest.mark.asyncio
async def test_update_creates_candidate_version_without_switching_active_document():
    session = _FakeUpdateSession()
    store = _FakeObjectStore()
    upload = UploadFile(filename="新版.txt", file=BytesIO(b"new content"), size=11)

    result = await create_document_update(
        session,
        doc_id=7,
        file=upload,
        expected_version=1,
        uploaded_by=9,
        object_store=store,
    )

    assert result.document.version == 1
    assert result.document.minio_path.endswith("old.txt")
    assert result.version.version == 2
    assert result.version.status == "PENDING"
    assert result.version.operation_type == "UPDATE"
    assert result.task.task_type == "UPDATE"
    assert result.task.doc_version == 2
    assert session.committed is True
    assert next(iter(store.objects.values())) == b"new content"


@pytest.mark.asyncio
async def test_update_rejects_file_with_same_hash_as_active_version():
    content = b"same content"
    session = _FakeUpdateSession(
        current_hash=hashlib.sha256(content).hexdigest()
    )
    store = _FakeObjectStore()
    upload = UploadFile(filename="相同.txt", file=BytesIO(content), size=len(content))

    with pytest.raises(DocumentUpdateConflictError, match="完全一致"):
        await create_document_update(
            session,
            doc_id=7,
            file=upload,
            expected_version=1,
            uploaded_by=9,
            object_store=store,
        )

    assert store.objects == {}


@pytest.mark.asyncio
async def test_restore_creates_new_candidate_from_ready_history_without_switching():
    session = _FakeRestoreSession()

    result = await create_document_restore(
        session,
        doc_id=7,
        source_version=1,
        expected_current_version=2,
        restored_by=9,
    )

    assert result.document.version == 2
    assert result.document.minio_path.endswith("current.txt")
    assert result.version.version == 3
    assert result.version.operation_type == "RESTORE"
    assert result.version.source_version == 1
    assert result.version.minio_path.endswith("history.txt")
    assert result.task.task_type == "RESTORE"
    assert result.task.doc_version == 3
    assert session.committed is True


@pytest.mark.asyncio
async def test_restore_rejects_stale_expected_current_version():
    session = _FakeRestoreSession()

    with pytest.raises(DocumentUpdateConflictError, match="当前版本为 V2"):
        await create_document_restore(
            session,
            doc_id=7,
            source_version=1,
            expected_current_version=1,
            restored_by=9,
        )

    assert session.added == []


@pytest.mark.asyncio
async def test_restore_rejects_when_an_index_task_is_active():
    session = _FakeRestoreSession(active_task=99)

    with pytest.raises(DocumentUpdateConflictError, match="已有正在执行"):
        await create_document_restore(
            session,
            doc_id=7,
            source_version=1,
            expected_current_version=2,
            restored_by=9,
        )

    assert session.added == []


@pytest.mark.asyncio
async def test_restore_rejects_non_ready_history_version():
    session = _FakeRestoreSession(source_status="FAILED")

    with pytest.raises(DocumentUpdateConflictError, match="只有 READY"):
        await create_document_restore(
            session,
            doc_id=7,
            source_version=1,
            expected_current_version=2,
            restored_by=9,
        )

    assert session.added == []
