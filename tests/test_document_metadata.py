"""业务元数据维护的权限、并发版本、白名单、上传字段和审计测试。"""

import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest
import test_document_upload_api as upload_tests
from fastapi.testclient import TestClient
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from zhiqing_rag.main import create_app
from zhiqing_rag.modules.documents.repository import UploadRepository
from zhiqing_rag.modules.documents.router import get_upload_identity, get_upload_service
from zhiqing_rag.modules.documents.schemas import DocumentMetadataPatch, UploadError, UploadIdentity

UPDATED = datetime(2026, 10, 11, 1, 0, tzinfo=UTC)
CURRENT = UploadIdentity(1, 101, 1, 10, 2)


@pytest.fixture
def upload_api():
    yield from upload_tests.api.__wrapped__()


class CodeConflict(UniqueViolation):
    @property
    def diag(self):
        return SimpleNamespace(constraint_name="uq_document_code")


class MetadataRepository(UploadRepository):
    def __init__(self, connection):
        self.inner = connection
        self.statements = []
        self.commits = 0
        super().__init__(self, "zhiqing_rag")

    @contextmanager
    def transaction(self):
        self.inner.execute("SAVEPOINT metadata_test")
        try:
            yield
            self.inner.execute("RELEASE SAVEPOINT metadata_test")
        except Exception:
            self.inner.execute("ROLLBACK TO SAVEPOINT metadata_test")
            self.inner.execute("RELEASE SAVEPOINT metadata_test")
            raise

    def commit(self):
        self.inner.commit()
        self.commits += 1

    def query(self, statement, parameters=()):
        self.statements.append(statement)
        statement = re.sub(r"\s+FOR (?:UPDATE|SHARE)(?: OF [a-zA-Z_, ]+)?\s*$", "", statement)
        statement = statement.replace("{schema}", "zhiqing_rag").replace("%s", "?")
        statement = statement.replace("IS NOT DISTINCT FROM", "IS")
        statement = statement.replace("clock_timestamp()", "'2026-10-11T02:00:00+00:00'")
        parameters = tuple(json.dumps(value.obj) if isinstance(value, Jsonb)
                           else value.isoformat() if isinstance(value, (datetime, date)) else value
                           for value in parameters)
        try:
            cursor = self.inner.execute(statement, parameters)
        except sqlite3.IntegrityError as error:
            if "document.tenant_id, document.kb_id, document.document_code" in str(error):
                raise CodeConflict() from error
            raise
        rows = [dict(row) for row in cursor.fetchall()]
        for row in rows:
            if row.get("updated_at"):
                row["updated_at"] = datetime.fromisoformat(row["updated_at"])
            if row.get("event_date"):
                row["event_date"] = date.fromisoformat(row["event_date"])
        return SimpleNamespace(fetchone=lambda: rows[0] if rows else None, fetchall=lambda: rows)


@pytest.fixture
def metadata_repository():
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("ATTACH DATABASE ':memory:' AS zhiqing_rag")
    definitions = {
        "tenant": "id INTEGER, status TEXT",
        "user_account": "id INTEGER, is_active BOOLEAN",
        "tenant_member": "id INTEGER, tenant_id INTEGER, user_id INTEGER, department_id INTEGER, "
                         "clearance INTEGER, status TEXT",
        "department": "id INTEGER, tenant_id INTEGER, name TEXT, is_active BOOLEAN",
        "knowledge_base": "id INTEGER, tenant_id INTEGER, status TEXT, deleted_at TEXT",
        "knowledge_base_grant": "id INTEGER, tenant_id INTEGER, kb_id INTEGER, user_id INTEGER, "
                                "department_id INTEGER, permission TEXT",
        "document_access_grant": "id INTEGER, tenant_id INTEGER, document_id INTEGER, "
                                 "user_id INTEGER, department_id INTEGER, permission TEXT",
        "document": "id INTEGER, tenant_id INTEGER, kb_id INTEGER, title TEXT, document_code TEXT, "
                    "business_status TEXT, event_date TEXT, department_id INTEGER, "
                    "confidentiality INTEGER, created_by INTEGER, updated_at TEXT, status TEXT, "
                    "deleted_at TEXT",
        "audit_event": "tenant_id INTEGER, actor_member_id INTEGER, action TEXT, "
                       "target_type TEXT, target_id TEXT, details TEXT",
    }
    for table, definition in definitions.items():
        connection.execute(f"CREATE TABLE zhiqing_rag.{table} ({definition})")
    connection.executescript("""
        CREATE UNIQUE INDEX zhiqing_rag.uq_document_code
          ON document(tenant_id, kb_id, document_code) WHERE deleted_at IS NULL;
        INSERT INTO zhiqing_rag.tenant VALUES (1, 'ACTIVE'), (2, 'ACTIVE');
        INSERT INTO zhiqing_rag.user_account VALUES (1, true), (2, true), (3, true), (4, true);
        INSERT INTO zhiqing_rag.department VALUES (10, 1, '财务部', true), (20, 1, '研发部', true);
        INSERT INTO zhiqing_rag.tenant_member VALUES
          (101, 1, 1, 10, 2, 'ACTIVE'), (102, 1, 2, 20, 2, 'ACTIVE'),
          (103, 1, 3, 10, 2, 'ACTIVE'), (104, 1, 4, 10, 0, 'ACTIVE');
        INSERT INTO zhiqing_rag.knowledge_base VALUES
          (1, 1, 'ACTIVE', null), (2, 2, 'ACTIVE', null);
        INSERT INTO zhiqing_rag.knowledge_base_grant VALUES
          (1, 1, 1, 1, null, 'ADMIN'), (2, 1, 1, 2, null, 'WRITE'),
          (3, 1, 1, 3, null, 'READ'), (4, 1, 1, 4, null, 'WRITE');
        INSERT INTO zhiqing_rag.document VALUES
          (11, 1, 1, '采购制度', 'A-001', 'UNKNOWN', null, 10, 1, 101,
           '2026-10-11T01:00:00+00:00', 'PUBLISHED', null),
          (12, 1, 1, '研发制度', 'A-002', 'DRAFT', null, 20, 1, 102,
           '2026-10-11T01:00:00+00:00', 'DRAFT', null),
          (21, 2, 2, '其他租户', null, 'UNKNOWN', null, null, 0, 201,
           '2026-10-11T01:00:00+00:00', 'PUBLISHED', null);
    """)
    repository = MetadataRepository(connection)
    yield repository
    connection.close()


@pytest.fixture
def metadata_api(metadata_repository):
    app = create_app()
    app.dependency_overrides[get_upload_identity] = lambda: CURRENT
    app.dependency_overrides[get_upload_service] = lambda: SimpleNamespace(
        repository=metadata_repository,
    )
    with TestClient(app) as client:
        yield client


def patch(**values):
    return DocumentMetadataPatch(expected_updated_at=UPDATED, **values)


def test_metadata_read_returns_live_business_values_and_permission(metadata_api):
    response = metadata_api.get("/api/documents/11/metadata")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "document_id": "11", "knowledge_base_id": "1", "title": "采购制度",
        "document_code": "A-001", "business_status": "UNKNOWN", "event_date": None,
        "department_id": "10", "department_name": "财务部",
        "updated_at": "2026-10-11T01:00:00Z", "can_write": True,
    }


@pytest.mark.parametrize("current,document_id", [
    (CURRENT, 21), (CURRENT, 12), (UploadIdentity(1, 102, 2, 20, 2), 11),
    (UploadIdentity(1, 104, 4, 10, 0), 11),
])
def test_metadata_never_exposes_foreign_tenant_department_or_high_clearance(
    metadata_repository, current, document_id,
):
    with pytest.raises(UploadError) as error:
        metadata_repository.document_metadata(current, document_id)
    assert error.value.status_code == 404


def test_read_only_member_can_view_but_cannot_patch(metadata_repository):
    current = UploadIdentity(1, 103, 3, 10, 2)
    assert not metadata_repository.document_metadata(current, 11).can_write
    with pytest.raises(UploadError) as error:
        metadata_repository.update_document_metadata(current, 11, patch(title="新标题"))
    assert error.value.status_code == 403
    assert metadata_repository.document_metadata(CURRENT, 11).title == "采购制度"


def test_explicit_read_share_does_not_grant_metadata_write(metadata_repository):
    metadata_repository.query(
        "INSERT INTO {schema}.document_access_grant VALUES (1, 1, 12, 1, null, 'READ')",
    )
    assert not metadata_repository.document_metadata(CURRENT, 12).can_write
    with pytest.raises(UploadError) as error:
        metadata_repository.update_document_metadata(CURRENT, 12, patch(title="越权"))
    assert error.value.status_code == 404


@pytest.mark.parametrize("status", ["WITHDRAWN", "DELETED"])
def test_unavailable_document_has_no_metadata(metadata_repository, status):
    metadata_repository.query("UPDATE {schema}.document SET status = %s WHERE id = 11", (status,))
    with pytest.raises(UploadError) as error:
        metadata_repository.document_metadata(CURRENT, 11)
    assert error.value.status_code == 404


def test_patch_changes_only_business_columns_and_records_before_after(
    metadata_api, metadata_repository,
):
    response = metadata_api.patch("/api/documents/11/metadata", json={
        "expected_updated_at": UPDATED.isoformat(), "title": " 新采购制度 ",
        "document_code": " NEW-001 ", "business_status": "EFFECTIVE", "event_date": "2026-10-01",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["title"] == "新采购制度" and result["document_code"] == "NEW-001"
    assert result["business_status"] == "EFFECTIVE" and result["event_date"] == "2026-10-01"
    assert result["department_id"] == "10"
    row = metadata_repository.query("SELECT * FROM {schema}.audit_event").fetchone()
    assert row["actor_member_id"] == 101 and row["action"] == "DOCUMENT_METADATA_UPDATE"
    assert json.loads(row["details"])["previous"]["business_status"] == "UNKNOWN"
    assert json.loads(row["details"])["current"]["event_date"] == "2026-10-01"
    statements = metadata_repository.statements
    kb_lock = next(i for i, text in enumerate(statements) if "FOR SHARE" in text)
    identity_lock = next(i for i, text in enumerate(statements) if "FOR SHARE OF m, t, u" in text)
    document_lock = next(i for i, text in enumerate(statements) if "FOR UPDATE OF doc" in text)
    assert kb_lock < identity_lock < document_lock
    assert metadata_repository.commits == 1


def test_stale_patch_does_not_overwrite_or_audit(metadata_repository):
    metadata_repository.update_document_metadata(CURRENT, 11, patch(title="新标题"))
    with pytest.raises(UploadError) as error:
        metadata_repository.update_document_metadata(CURRENT, 11, patch(title="过期标题"))
    assert error.value.code == "METADATA_CONFLICT"
    assert metadata_repository.document_metadata(CURRENT, 11).title == "新标题"
    assert len(metadata_repository.query("SELECT * FROM {schema}.audit_event").fetchall()) == 1


def test_duplicate_code_rolls_back_and_returns_friendly_conflict(metadata_api, metadata_repository):
    response = metadata_api.patch("/api/documents/11/metadata", json={
        "expected_updated_at": UPDATED.isoformat(), "title": "不应保存", "document_code": "A-002",
    })
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "DOCUMENT_CODE_CONFLICT"
    assert metadata_repository.document_metadata(CURRENT, 11).title == "采购制度"
    assert not metadata_repository.query("SELECT * FROM {schema}.audit_event").fetchall()


def test_clear_optional_values_and_noop_skip_audit(metadata_repository):
    metadata_repository.update_document_metadata(CURRENT, 11, patch(business_status="UNKNOWN"))
    assert not metadata_repository.query("SELECT * FROM {schema}.audit_event").fetchall()
    result = metadata_repository.update_document_metadata(
        CURRENT, 11, patch(document_code=" ", event_date=None),
    )
    assert result.document_code is None and result.event_date is None


def test_identity_changes_are_checked_before_metadata_write(metadata_repository):
    metadata_repository.query("UPDATE {schema}.tenant_member SET clearance = 1 WHERE id = 101")
    with pytest.raises(UploadError) as error:
        metadata_repository.update_document_metadata(CURRENT, 11, patch(title="被拒绝"))
    assert error.value.code == "IDENTITY_UNAVAILABLE"
    assert not metadata_repository.query("SELECT * FROM {schema}.audit_event").fetchall()


@pytest.mark.parametrize("values", [
    {}, {"title": " "}, {"title": None}, {"business_status": None},
    {"business_status": "SUCCEEDED"},
    {"document_code": "x" * 101}, {"document_code": "A\nB"}, {"event_date": "2026-02-30"},
    {"department_id": "20"}, {"confidentiality": 0}, {"knowledge_base_id": "2"},
    {"expected_updated_at": "2026-10-11T01:00:00"},
])
def test_patch_rejects_invalid_values_and_auth_fields(metadata_api, metadata_repository, values):
    response = metadata_api.patch("/api/documents/11/metadata", json={
        "expected_updated_at": UPDATED.isoformat(), **values,
    })
    assert response.status_code == 422
    assert not metadata_repository.query("SELECT * FROM {schema}.audit_event").fetchall()


def test_upload_accepts_optional_metadata(upload_api):
    client, repository, _ = upload_api
    response = upload_tests.upload(
        client, document_code=" DOC-001 ", business_status="EFFECTIVE", event_date="2026-10-01",
    )
    assert response.status_code == 202
    parameters = repository.saved[0][1]
    assert parameters.document_code == "DOC-001" and parameters.business_status == "EFFECTIVE"
    assert parameters.event_date == date(2026, 10, 1)


@pytest.mark.parametrize("values", [
    {"document_code": "x" * 101}, {"document_code": "A\nB"},
    {"business_status": "PUBLISHED"}, {"event_date": "2026-02-30"},
])
def test_invalid_upload_metadata_never_stores_file(upload_api, values):
    client, repository, storage = upload_api
    assert upload_tests.upload(client, **values).status_code == 422
    assert not repository.saved and not storage.objects


def test_upload_duplicate_code_removes_uncommitted_object(upload_api, monkeypatch):
    client, repository, storage = upload_api

    def duplicate(*args, **kwargs):
        raise CodeConflict()

    monkeypatch.setattr(repository, "persist", duplicate)
    response = upload_tests.upload(client, document_code="A-001")
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "DOCUMENT_CODE_CONFLICT"
    assert storage.removed == [(storage.bucket, storage.objects[0][0])]
    assert not repository.saved and repository.commits == 0


@pytest.mark.parametrize("method,body", [("GET", None), ("PATCH", {"title": "标题"})])
def test_metadata_requires_login_before_opening_services(monkeypatch, method, body):
    import zhiqing_rag.modules.documents.router as documents

    def unexpected(*args, **kwargs):
        pytest.fail("匿名请求不得建立业务连接")

    monkeypatch.setattr(documents, "database_connection", unexpected)
    with TestClient(create_app()) as client:
        response = client.request(method, "/api/documents/11/metadata", json=body)
    assert response.status_code == 401
