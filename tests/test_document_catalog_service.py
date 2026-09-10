from datetime import datetime
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from zq_rag_app.services.document_catalog_service import (
    DocumentUploadValidationError,
    create_uploaded_documents,
)


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
        self.rolled_back = False

    async def scalar(self, statement):
        return SimpleNamespace(id=1, is_deleted=False)

    def add_all(self, documents):
        self.documents.extend(documents)

    async def commit(self):
        return None

    async def rollback(self):
        self.rolled_back = True

    async def refresh(self, document):
        document.id = 100 + self.documents.index(document)
        document.uploaded_at = datetime(2026, 9, 9, 10, 0, 0)
        document.indexed_at = None
        document.error_msg = None


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
