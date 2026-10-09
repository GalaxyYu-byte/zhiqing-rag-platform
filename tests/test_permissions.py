"""统一SQL/SQLAlchemy授权语义和权限管理事务的可运行边界测试。"""

import json
import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import aliased
from sqlalchemy.schema import CreateTable

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.documents.document import Document
from zhiqing_rag.modules.documents.document_access_grant import DocumentAccessGrant
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.modules.knowledge import router as permission_router
from zhiqing_rag.modules.knowledge.permission_repository import PermissionRepository
from zhiqing_rag.modules.knowledge.permission_schemas import (
    DocumentGrantRequest,
    GrantRequest,
)
from zhiqing_rag.modules.knowledge.permissions import (
    document_access_condition,
    document_access_sql,
    document_scope_sql,
)


def identity(user=1001, member=101, department=10, clearance=2, tenant=1):
    return SimpleNamespace(user_id=user, member_id=member, department_id=department,
                           clearance=clearance, tenant_id=tenant, role="ADMIN")


class Result:
    def __init__(self, result):
        self.rows = [dict(row) for row in result.mappings()] if result.returns_rows else []

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        result, self.rows = self.rows, []
        return result


class Connection:
    def __init__(self, connection):
        self.inner = connection

    @contextmanager
    def transaction(self):
        with (self.inner.begin_nested() if self.inner.in_transaction() else self.inner.begin()):
            yield

    def commit(self):
        self.inner.commit()


class SqliteRepository(PermissionRepository):
    def __init__(self, connection):
        super().__init__(Connection(connection), "zhiqing_rag")
        self.statements = []

    def query(self, statement, parameters=()):
        self.statements.append(statement)
        statement = re.sub(r"\s+FOR (?:UPDATE|SHARE)(?: OF [a-zA-Z_, ]+)?\s*$", "", statement)
        statement = statement.replace("{schema}", "zhiqing_rag").replace("%s", "?")
        parameters = tuple(json.dumps(value.obj) if isinstance(value, Jsonb) else value
                           for value in parameters)
        return Result(self.connection.inner.exec_driver_sql(statement, parameters))


@pytest.fixture
def repository():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    with engine.connect() as connection:
        connection.exec_driver_sql("ATTACH DATABASE ':memory:' AS zhiqing_rag")
        definitions = {
            "user_account": "id INTEGER PRIMARY KEY, email TEXT, display_name TEXT, "
                            "is_active BOOLEAN",
            "tenant": "id INTEGER PRIMARY KEY, status TEXT, auth_version INTEGER",
            "department": "id INTEGER PRIMARY KEY, tenant_id INTEGER, name TEXT, is_active BOOLEAN",
            "tenant_member": "id INTEGER PRIMARY KEY, tenant_id INTEGER, user_id INTEGER, "
                             "department_id INTEGER, clearance INTEGER, status TEXT, "
                             "auth_version INTEGER",
            "knowledge_base": "id INTEGER PRIMARY KEY, tenant_id INTEGER, name TEXT, status TEXT, "
                              "deleted_at TEXT, acl_version INTEGER, updated_at TEXT",
            "document": "id INTEGER PRIMARY KEY, tenant_id INTEGER, kb_id INTEGER, title TEXT, "
                        "department_id INTEGER, confidentiality INTEGER, created_by INTEGER, "
                        "status TEXT, deleted_at TEXT, active_generation_id INTEGER, "
                        "updated_at TEXT",
            "knowledge_base_grant": "id INTEGER PRIMARY KEY, tenant_id INTEGER, kb_id INTEGER, "
                                    "user_id INTEGER, department_id INTEGER, permission TEXT, "
                                    "granted_by INTEGER, updated_at TEXT",
            "document_access_grant": "id INTEGER PRIMARY KEY, tenant_id INTEGER, "
                                     "document_id INTEGER, "
                                     "user_id INTEGER, department_id INTEGER, permission TEXT, "
                                     "granted_by INTEGER, updated_at TEXT",
            "audit_event": "id INTEGER PRIMARY KEY, tenant_id INTEGER, actor_member_id INTEGER, "
                           "action TEXT, target_type TEXT, target_id TEXT, details TEXT",
        }
        for table, definition in definitions.items():
            connection.exec_driver_sql(f"CREATE TABLE zhiqing_rag.{table} ({definition})")
        connection.exec_driver_sql(
            "INSERT INTO zhiqing_rag.tenant VALUES (1,'ACTIVE',1),(2,'ACTIVE',1)"
        )
        connection.exec_driver_sql("""INSERT INTO zhiqing_rag.department VALUES
            (10,1,'A',true),(20,1,'B',true),(21,1,'B子部门',true),
            (30,1,'停用部门',false),(200,2,'隔离部门',true)""")
        connection.exec_driver_sql("""INSERT INTO zhiqing_rag.user_account VALUES
            (1001,'a@test','A用户',true),(1002,'b@test','B用户',true),
            (1003,'low@test','低密级',true),(1004,'none@test','无部门',true),
            (1005,'child@test','子部门',true),(1006,'disabled@test','停用',false)""")
        connection.exec_driver_sql("""INSERT INTO zhiqing_rag.tenant_member VALUES
            (101,1,1001,10,2,'ACTIVE',1),(102,1,1002,20,2,'ACTIVE',1),
            (103,1,1003,10,1,'ACTIVE',1),(104,1,1004,NULL,2,'ACTIVE',1),
            (105,1,1005,21,2,'ACTIVE',1),(106,1,1006,10,2,'ACTIVE',1),
            (201,2,1001,200,2,'ACTIVE',1)""")
        connection.exec_driver_sql("""INSERT INTO zhiqing_rag.knowledge_base
            (id,tenant_id,name,status,deleted_at,acl_version) VALUES
            (1,1,'KB','ACTIVE',NULL,1),(2,2,'隔离KB','ACTIVE',NULL,1),
            (3,1,'归档KB','ARCHIVED',NULL,1),(4,1,'无授权KB','ACTIVE',NULL,1)""")
        connection.exec_driver_sql("""INSERT INTO zhiqing_rag.knowledge_base_grant
            (id,tenant_id,kb_id,user_id,department_id,permission,granted_by) VALUES
            (1,1,1,1001,NULL,'ADMIN',101),(2,1,1,NULL,10,'READ',101),
            (3,1,1,1002,NULL,'WRITE',101),(4,1,1,1004,NULL,'READ',101),
            (5,2,2,1001,NULL,'ADMIN',201),(6,1,1,NULL,20,'READ',101)""")
        rows = [
            (1, 1, 1, 10, 1, 101, "PUBLISHED", None, 11),
            (2, 1, 1, 20, 1, 102, "PUBLISHED", None, 12),
            (3, 1, 1, None, 1, 101, "PUBLISHED", None, 13),
            (4, 1, 1, None, 1, 102, "PUBLISHED", None, 14),
            (5, 1, 1, 20, 0, 102, "PUBLISHED", None, 15),
            (6, 1, 1, 10, 2, 101, "PUBLISHED", None, 16),
            (7, 2, 2, 200, 0, 201, "PUBLISHED", None, 17),
            (8, 1, 3, 10, 0, 101, "PUBLISHED", None, 18),
            (9, 1, 4, 10, 0, 101, "PUBLISHED", None, 19),
            (10, 1, 1, 10, 1, 101, "DRAFT", None, None),
            (11, 1, 1, 10, 1, 101, "WITHDRAWN", None, 11),
            (12, 1, 1, 10, 1, 101, "DELETED", "2026-10-02", None),
            (13, 1, 1, 10, 1, 101, "PUBLISHED", None, None),
            (14, 1, 1, 10, 1, 102, "PUBLISHED", None, 14),
        ]
        connection.exec_driver_sql("""INSERT INTO zhiqing_rag.document
            (id,tenant_id,kb_id,department_id,confidentiality,created_by,status,deleted_at,
             active_generation_id,title) VALUES (?,?,?,?,?,?,?,?,?, '测试文档')""", rows)
        connection.commit()
        yield SqliteRepository(connection)
    engine.dispose()


def sql_visible(repository, current, **options):
    predicate, parameters = document_access_sql(current, **options)
    rows = repository.query("""SELECT doc.id FROM {schema}.document doc
        JOIN {schema}.knowledge_base kb ON kb.tenant_id=doc.tenant_id AND kb.id=doc.kb_id
        WHERE """ + predicate, parameters).fetchall()
    return {row["id"] for row in rows}


def sa_visible(repository, current, **options):
    return set(repository.connection.inner.scalars(select(Document.id).where(
        document_access_condition(current, **options),
    )))


@pytest.mark.parametrize("current,expected", [
    (identity(), {1, 3, 5, 6, 14}),
    (identity(user=1002, member=102, department=20), {2, 4, 5}),
    (identity(user=1003, member=103, clearance=1), {1, 5, 14}),
    (identity(user=1004, member=104, department=None), {5}),
    (identity(user=1005, member=105, department=21), set()),
])
def test_read_policy_matches_real_sql_and_sqlalchemy(repository, current, expected):
    assert sql_visible(repository, current, published_only=True) == expected
    assert sa_visible(repository, current) == expected


def test_default_listing_and_deleted_maintenance_do_not_include_withdrawn(repository):
    assert sql_visible(repository, identity()) == {1, 3, 5, 6, 10, 13, 14}
    assert sql_visible(repository, identity(), include_deleted=True) == {1, 3, 5, 6, 10, 12, 13, 14}
    assert sa_visible(repository, identity(), published_only=False,
                      include_deleted=True) == {1, 3, 5, 6, 10, 12, 13, 14}
    assert 12 not in sql_visible(repository, identity(), include_deleted=True, published_only=True)


def test_explicit_read_share_never_escalates_to_write_even_with_kb_write(repository):
    repository.query("""INSERT INTO {schema}.document_access_grant
        (tenant_id,document_id,user_id,permission,granted_by)
        VALUES (1,1,1002,'READ',101)""")
    current = identity(user=1002, member=102, department=20)
    assert 1 in sql_visible(repository, current)
    assert 1 in sa_visible(repository, current)
    assert 1 not in sql_visible(repository, current, permission="WRITE")
    assert 1 not in sa_visible(repository, current, permission="WRITE")
    repository.query("UPDATE {schema}.document_access_grant SET permission='WRITE'")
    assert 1 in sql_visible(repository, current, permission="WRITE")
    assert 1 in sa_visible(repository, current, permission="WRITE")


def test_explicit_share_requires_kb_permission_clearance_and_exact_department(repository):
    repository.query("""INSERT INTO {schema}.document_access_grant
        (tenant_id,document_id,department_id,permission,granted_by)
        VALUES (1,6,20,'WRITE',101),(1,9,20,'WRITE',101)""")
    current = identity(user=1002, member=102, department=20)
    assert 6 in sql_visible(repository, current)
    assert 9 not in sql_visible(repository, current)
    assert 6 not in sql_visible(
        repository, identity(user=1002, member=102, department=20, clearance=1),
    )
    assert 6 not in sql_visible(repository, identity(user=1005, member=105, department=21))
    assert 7 not in sql_visible(repository, current)


def test_author_moved_departments_loses_earlier_department_scope(repository):
    moved = identity(department=20)
    assert 1 not in sql_visible(repository, moved)
    assert 1 not in sa_visible(repository, moved)
    assert 3 in sql_visible(repository, moved)
    assert 3 in sa_visible(repository, moved)


def test_no_department_documents_are_private_to_author_or_explicit_sharing(repository):
    current = identity(user=1004, member=104, department=None)
    assert 3 not in sql_visible(repository, current)
    repository.query("""INSERT INTO {schema}.document_access_grant
        (tenant_id,document_id,user_id,permission,granted_by) VALUES (1,3,1004,'READ',101)""")
    assert 3 in sql_visible(repository, current)
    assert 3 in sa_visible(repository, current)
    assert 3 not in sql_visible(repository, current, permission="WRITE")


def test_admin_does_not_bypass_document_scope_or_missing_kb_acl(repository):
    assert 2 not in sql_visible(repository, identity(), permission="ADMIN")
    assert 9 not in sql_visible(repository, identity(), permission="READ")
    repository.query("""INSERT INTO {schema}.document_access_grant
        (tenant_id,document_id,user_id,permission,granted_by) VALUES (1,2,1001,'READ',101)""")
    assert 2 not in sql_visible(repository, identity(), permission="ADMIN")
    repository.query("UPDATE {schema}.document_access_grant SET permission='WRITE'")
    assert 2 in sql_visible(repository, identity(), permission="ADMIN")
    assert 2 in sa_visible(repository, identity(), permission="ADMIN")


def test_sqlalchemy_supports_document_alias(repository):
    alias = aliased(Document, name="owned")
    condition = document_access_condition(identity(), document=alias)
    result = set(repository.connection.inner.scalars(select(alias.id).where(condition)))
    assert result == {1, 3, 5, 6, 14}


@pytest.mark.parametrize("bad", ["doc; DROP TABLE x", "doc.id", "x--", "", "with space"])
def test_internal_alias_injection_is_rejected(bad):
    with pytest.raises(ValueError):
        document_scope_sql(identity(), alias=bad)
    with pytest.raises(ValueError):
        document_access_sql(identity(), kb_alias=bad)


def test_unknown_permission_fails_closed():
    with pytest.raises(ValueError):
        document_access_sql(identity(), permission="OWNER")


def test_document_grant_ddl_has_tenant_fk_cascade_and_subject_constraints():
    ddl = str(CreateTable(DocumentAccessGrant.__table__).compile(dialect=postgresql.dialect()))
    assert "ON DELETE CASCADE" in ddl
    assert "num_nonnulls(user_id, department_id) = 1" in ddl
    assert "permission IN ('READ','WRITE')" in ddl
    assert "FOREIGN KEY(tenant_id, user_id)" in ddl
    assert "FOREIGN KEY(tenant_id, granted_by)" in ddl
    assert len(DocumentAccessGrant.__table__.indexes) == 4


@pytest.mark.parametrize("body", [
    {"permission": "READ"}, {"user_id": "1", "department_id": "2", "permission": "READ"},
    {"user_id": "0", "permission": "READ"}, {"user_id": "1", "permission": "OWNER"},
])
def test_grant_request_requires_one_valid_subject(body):
    with pytest.raises(ValidationError):
        GrantRequest(**body)
    with pytest.raises(ValidationError):
        DocumentGrantRequest(user_id="1", permission="ADMIN")


def test_kb_permission_views_have_only_current_active_tenant_subjects(repository):
    view = repository.knowledge_base_permissions(identity(), 1)
    assert view.knowledge_base.id == "1"
    assert {user.id for user in view.subjects.users} == {"1001", "1002", "1003", "1004", "1005"}
    assert {department.id for department in view.subjects.departments} == {"10", "20", "21"}
    assert all(isinstance(grant.id, str) for grant in view.grants)


def assert_error(code, operation, *args):
    with pytest.raises(AuthError) as result:
        operation(*args)
    assert result.value.code == code


def test_management_requires_kb_admin_and_document_write_scope(repository):
    b = identity(user=1002, member=102, department=20)
    assert_error("RESOURCE_NOT_FOUND", repository.knowledge_base_permissions, b, 1)
    assert_error("RESOURCE_NOT_FOUND", repository.document_permissions, identity(), 2)
    assert_error("RESOURCE_NOT_FOUND", repository.knowledge_base_permissions, identity(), 2)
    assert_error("RESOURCE_NOT_FOUND", repository.knowledge_base_permissions, identity(), 4)


def test_last_available_admin_cannot_be_removed_or_downgraded(repository):
    assert_error("LAST_ADMIN_REQUIRED", repository.delete_knowledge_base_grant, identity(), 1, 1)
    assert_error("LAST_ADMIN_REQUIRED", repository.upsert_knowledge_base_grant,
                 identity(), 1, GrantRequest(user_id=1001, permission="READ"))
    view = repository.knowledge_base_permissions(identity(), 1)
    assert next(grant.permission for grant in view.grants if grant.id == "1") == "ADMIN"
    assert view.knowledge_base.acl_version == 1
    assert not repository.query("SELECT * FROM {schema}.audit_event").fetchall()


def test_deactivated_admin_does_not_count_as_available(repository):
    repository.query("""INSERT INTO {schema}.knowledge_base_grant
        (tenant_id,kb_id,user_id,permission,granted_by) VALUES (1,1,1006,'ADMIN',101)""")
    assert_error("LAST_ADMIN_REQUIRED", repository.delete_knowledge_base_grant, identity(), 1, 1)


def test_kb_changes_are_audited_versioned_and_upsert_is_idempotent(repository):
    grant = GrantRequest(user_id=1002, permission="ADMIN")
    result = repository.upsert_knowledge_base_grant(identity(), 1, grant)
    assert result.knowledge_base.acl_version == 2
    assert next(item.permission for item in result.grants if item.user_id == "1002") == "ADMIN"
    repeat = repository.upsert_knowledge_base_grant(identity(), 1, grant)
    assert repeat.knowledge_base.acl_version == 2
    result = repository.delete_knowledge_base_grant(identity(), 1, 1)
    assert result.knowledge_base.acl_version == 3
    audit = repository.query(
        "SELECT action, details FROM {schema}.audit_event ORDER BY id"
    ).fetchall()
    assert len(audit) == 2
    assert json.loads(audit[0]["details"])["previous_permission"] == "WRITE"
    assert json.loads(audit[1]["details"])["acl_version"] == 3


def test_doc_share_revoke_and_classification_only_change_acl_metadata(repository):
    grant = DocumentGrantRequest(user_id=1002, permission="WRITE")
    result = repository.upsert_document_grant(identity(), 1, grant)
    assert result.grants[0].user_id == "1002"
    assert 1 in sql_visible(repository, identity(user=1002, member=102, department=20),
                            permission="WRITE")
    repository.delete_document_grant(identity(), 1, int(result.grants[0].id))
    assert 1 not in sql_visible(repository, identity(user=1002, member=102, department=20))
    changed = repository.classify_document(identity(), 1, 0)
    assert changed.document.confidentiality == 0
    row = repository.query("SELECT acl_version FROM {schema}.knowledge_base WHERE id=1").fetchone()
    assert row["acl_version"] == 4
    doc = repository.query(
        "SELECT active_generation_id FROM {schema}.document WHERE id=1"
    ).fetchone()
    assert doc["active_generation_id"] == 11
    assert len(repository.query("SELECT * FROM {schema}.audit_event").fetchall()) == 3


def test_classification_above_clearance_denied_without_mutation(repository):
    repository.query("UPDATE {schema}.tenant_member SET clearance=1 WHERE id=101")
    actor = identity(clearance=1)
    assert_error("CLEARANCE_EXCEEDED", repository.classify_document, actor, 1, 2)
    assert repository.document_permissions(actor, 1).document.confidentiality == 1
    assert not repository.query("SELECT * FROM {schema}.audit_event").fetchall()


def test_subject_cross_tenant_or_inactive_denied_and_grant_unknown_not_found(repository):
    for grant in (GrantRequest(user_id=1006, permission="READ"),
                  GrantRequest(user_id=9999, permission="READ"),
                  GrantRequest(department_id=200, permission="READ"),
                  GrantRequest(department_id=30, permission="READ")):
        assert_error("PERMISSION_SUBJECT_INVALID", repository.upsert_knowledge_base_grant,
                     identity(), 1, grant)
    assert_error("RESOURCE_NOT_FOUND", repository.delete_document_grant, identity(), 1, 9999)


def test_actor_permission_changes_rechecked_inside_transaction(repository):
    repository.query("UPDATE {schema}.tenant_member SET clearance=1 WHERE id=101")
    assert_error("AUTHORIZATION_CHANGED", repository.upsert_document_grant,
                 identity(), 1, DocumentGrantRequest(user_id=1002, permission="READ"))


def test_mutation_locks_kb_before_identity_and_document(repository):
    repository.upsert_document_grant(identity(), 1,
                                     DocumentGrantRequest(user_id=1002, permission="READ"))
    kb_lock = next(index for index, statement in enumerate(repository.statements)
                   if "knowledge_base WHERE" in statement and "FOR UPDATE" in statement)
    identity_lock = next(index for index, statement in enumerate(repository.statements)
                         if "FOR SHARE OF m, t, u" in statement)
    doc_lock = next(index for index, statement in enumerate(repository.statements)
                    if "FOR UPDATE OF doc" in statement)
    assert kb_lock < identity_lock < doc_lock


def test_http_permission_contract_and_csrf(repository):
    app = FastAPI()
    app.include_router(permission_router.router)
    app.dependency_overrides[require_identity] = lambda: identity()
    app.dependency_overrides[permission_router.get_permission_repository] = lambda: repository
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)

    @app.exception_handler(AuthError)
    async def handle(request: Request, error: AuthError):
        return JSONResponse(status_code=error.status_code,
                            content={"detail": {"code": error.code, "message": str(error)}})

    with TestClient(app) as client:
        response = client.get("/api/knowledge-bases/1/permissions")
        assert response.status_code == 200
        assert response.json()["knowledge_base"]["id"] == "1"
        forbidden = client.post("/api/documents/1/permissions", json={
            "user_id": "1002", "permission": "READ",
        })
        assert forbidden.status_code == 403
        shared = client.post("/api/documents/1/permissions", headers={
            "X-Requested-With": "XMLHttpRequest",
        }, json={"user_id": "1002", "permission": "READ"})
        assert shared.status_code == 200
        assert shared.json()["document"]["department_id"] == "10"
        assert shared.json()["grants"][0]["user_id"] == "1002"
