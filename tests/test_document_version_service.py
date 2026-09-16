from datetime import datetime

import pytest

from zq_rag_app.models.document import Document, DocumentVersion, IndexTask
from zq_rag_app.models.graph import GraphTask
from zq_rag_app.services import document_service
from zq_rag_app.services.document_service import create_index_task
from zq_rag_app.services.document_service import DocumentIndexService, _DocumentSnapshot


class _FakeIndexSession:
    def __init__(self, document, current_version):
        self.scalar_values = iter([document, None, current_version, 1])
        self.added = []
        self.committed = False

    async def scalar(self, statement):
        del statement
        return next(self.scalar_values)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.committed = True

    async def refresh(self, task):
        task.id = 30
        task.created_at = datetime(2026, 9, 11, 12, 0, 0)


class _CompletionSession:
    def __init__(self, task, document, version):
        self.task = task
        self.document = document
        self.version = version
        self.executed = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def get(self, model, object_id, **kwargs):
        del object_id, kwargs
        if model is IndexTask:
            return self.task
        if model is Document:
            return self.document
        return None

    async def scalar(self, statement):
        del statement
        return self.version

    async def execute(self, statement):
        self.executed.append(statement)

    async def commit(self):
        self.committed = True


class _CompletionFactory:
    def __init__(self, session):
        self.session = session

    def __call__(self):
        return self.session


class _GraphCompletionSession(_CompletionSession):
    def __init__(self, task, document, version):
        super().__init__(task, document, version)
        self.scalar_values = iter([version, None])
        self.added = []

    async def scalar(self, statement):
        del statement
        return next(self.scalar_values)

    def add(self, value):
        self.added.append(value)

    async def flush(self):
        graph_task = next(
            value for value in self.added if isinstance(value, GraphTask)
        )
        graph_task.id = 91


@pytest.mark.asyncio
async def test_reindex_builds_candidate_before_switching_active_version():
    document = Document(
        id=7,
        kb_id=2,
        file_name="手册.pdf",
        file_type="PDF",
        file_size=100,
        minio_path="rag-documents/kb/2/handbook.pdf",
        status="DONE",
        version=1,
        uploaded_by=9,
        is_deleted=False,
    )
    current_version = DocumentVersion(
        id=11,
        doc_id=7,
        version=1,
        file_name=document.file_name,
        file_type=document.file_type,
        file_size=document.file_size,
        file_hash="a" * 64,
        minio_path=document.minio_path,
        status="READY",
        uploaded_by=9,
    )
    session = _FakeIndexSession(document, current_version)

    task, created = await create_index_task(
        session,
        doc_id=document.id,
        task_type="REINDEX",
    )

    candidate = next(
        value for value in session.added if isinstance(value, DocumentVersion)
    )
    assert created is True
    assert document.version == 1
    assert document.status == "DONE"
    assert candidate.version == 2
    assert candidate.operation_type == "REINDEX"
    assert candidate.minio_path == document.minio_path
    assert task.doc_version == 2
    assert isinstance(task, IndexTask)
    assert session.committed is True


@pytest.mark.asyncio
async def test_completion_atomically_switches_candidate_to_active_version():
    document = Document(
        id=7,
        kb_id=2,
        file_name="旧版.txt",
        file_type="TXT",
        file_size=3,
        minio_path="rag-documents/old.txt",
        status="DONE",
        version=1,
        uploaded_by=3,
        is_deleted=False,
    )
    version = DocumentVersion(
        id=12,
        doc_id=7,
        version=2,
        file_name="新版.txt",
        file_type="TXT",
        file_size=8,
        file_hash="b" * 64,
        minio_path="rag-documents/new.txt",
        status="PROCESSING",
        uploaded_by=9,
    )
    task = IndexTask(
        id=30,
        doc_id=7,
        doc_version=2,
        task_type="UPDATE",
        status="PROCESSING",
        stage="PERSISTING",
        worker_id="test-worker",
    )
    session = _CompletionSession(task, document, version)
    service = DocumentIndexService(
        session_factory=_CompletionFactory(session),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="test-worker",
    )

    await service._complete_task(
        task_id=30,
        document=_DocumentSnapshot(
            id=7,
            kb_id=2,
            file_name="新版.txt",
            minio_path="rag-documents/new.txt",
            version=2,
        ),
        chunk_count=4,
        token_count=80,
    )

    assert document.version == 2
    assert document.file_name == "新版.txt"
    assert document.minio_path == "rag-documents/new.txt"
    assert version.status == "READY"
    assert task.status == "DONE"
    assert document.chunk_count == 4
    assert session.committed is True


@pytest.mark.asyncio
async def test_vector_completion_creates_graph_task_when_enabled(monkeypatch):
    monkeypatch.setattr(
        document_service.settings,
        "graph_extraction_enabled",
        True,
    )
    document = Document(
        id=7,
        kb_id=2,
        file_name="手册.txt",
        file_type="TXT",
        file_size=8,
        minio_path="rag-documents/manual.txt",
        status="PROCESSING",
        version=1,
        uploaded_by=3,
        is_deleted=False,
    )
    version = DocumentVersion(
        id=12,
        doc_id=7,
        version=1,
        file_name="手册.txt",
        file_type="TXT",
        file_size=8,
        minio_path="rag-documents/manual.txt",
        status="PROCESSING",
        uploaded_by=3,
    )
    task = IndexTask(
        id=30,
        doc_id=7,
        doc_version=1,
        task_type="INDEX",
        status="PROCESSING",
        stage="PERSISTING",
        worker_id="test-worker",
    )
    session = _GraphCompletionSession(task, document, version)
    service = DocumentIndexService(
        session_factory=_CompletionFactory(session),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="test-worker",
    )

    graph_task_id = await service._complete_task(
        task_id=30,
        document=_DocumentSnapshot(
            id=7,
            kb_id=2,
            file_name="手册.txt",
            minio_path="rag-documents/manual.txt",
            version=1,
        ),
        chunk_count=2,
        token_count=30,
    )

    graph_task = next(
        value for value in session.added if isinstance(value, GraphTask)
    )
    assert graph_task_id == 91
    assert graph_task.doc_id == 7
    assert graph_task.doc_version == 1
    assert graph_task.status == "PENDING"


@pytest.mark.asyncio
async def test_failed_candidate_does_not_break_active_version():
    document = Document(
        id=7,
        kb_id=2,
        file_name="正式版.txt",
        file_type="TXT",
        file_size=3,
        minio_path="rag-documents/active.txt",
        status="DONE",
        version=1,
        uploaded_by=3,
        is_deleted=False,
    )
    version = DocumentVersion(
        id=12,
        doc_id=7,
        version=2,
        file_name="失败版.txt",
        file_type="TXT",
        file_size=8,
        minio_path="rag-documents/failed.txt",
        status="PROCESSING",
        uploaded_by=9,
    )
    task = IndexTask(
        id=30,
        doc_id=7,
        doc_version=2,
        task_type="UPDATE",
        status="PROCESSING",
        stage="EMBEDDING",
        worker_id="test-worker",
        retry_count=0,
        max_retry=0,
    )
    session = _CompletionSession(task, document, version)
    service = DocumentIndexService(
        session_factory=_CompletionFactory(session),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="test-worker",
    )

    should_retry, _ = await service._record_failure(30, ValueError("bad file"))

    assert should_retry is False
    assert task.status == "FAILED"
    assert version.status == "FAILED"
    assert document.status == "DONE"
    assert document.version == 1
    assert document.minio_path == "rag-documents/active.txt"


@pytest.mark.asyncio
async def test_failed_restore_candidate_keeps_current_version_available():
    document = Document(
        id=7,
        kb_id=2,
        file_name="当前版.txt",
        file_type="TXT",
        file_size=8,
        minio_path="rag-documents/current.txt",
        status="DONE",
        version=2,
        uploaded_by=3,
        is_deleted=False,
    )
    version = DocumentVersion(
        id=13,
        doc_id=7,
        version=3,
        file_name="历史版.txt",
        file_type="TXT",
        file_size=5,
        minio_path="rag-documents/history.txt",
        operation_type="RESTORE",
        source_version=1,
        status="PROCESSING",
        uploaded_by=9,
    )
    task = IndexTask(
        id=31,
        doc_id=7,
        doc_version=3,
        task_type="RESTORE",
        status="PROCESSING",
        stage="EMBEDDING",
        worker_id="test-worker",
        retry_count=0,
        max_retry=0,
    )
    session = _CompletionSession(task, document, version)
    service = DocumentIndexService(
        session_factory=_CompletionFactory(session),  # type: ignore[arg-type]
        embedding_service=object(),  # type: ignore[arg-type]
        worker_id="test-worker",
    )

    should_retry, _ = await service._record_failure(
        31, ValueError("bad history")
    )

    assert should_retry is False
    assert version.status == "FAILED"
    assert document.status == "DONE"
    assert document.version == 2
    assert document.minio_path == "rag-documents/current.txt"
