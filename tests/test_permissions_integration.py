"""独立 PostgreSQL / Redis 命名空间中的知识权限接口与统一查询门禁实测。"""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from pydantic import SecretStr
from redis import Redis
from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.documents import router as document_router
from zhiqing_rag.modules.documents.schemas import UploadIdentity
from zhiqing_rag.modules.identity.passwords import hash_password
from zhiqing_rag.modules.schema import SCHEMA

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PERMISSION_INTEGRATION") != "1",
    reason="需要显式启用真实 PostgreSQL / Redis 隔离知识权限实测",
)

PASSWORD = "Permission-test-passphrase-2026"
CONTENT = b"# Permission integration\n\nIsolated document content.\n"


class MemoryObjectStorage:
    """仅替代对象存储；身份、业务权限与持久化查询仍访问真实 PostgreSQL。"""

    bucket = "permission-test"

    def __init__(self):
        self.objects = {}
        self.reads = 0
        self.writes = 0
        self.removals = 0
        self.before_write = None
        self.client = SimpleNamespace(get_object=self.get_object)

    def get_object(self, bucket, key):
        self.reads += 1
        stream = BytesIO(self.objects[(bucket, key)])
        stream.release_conn = lambda: None
        return stream

    def put(self, key, path, mime_type):
        self.writes += 1
        self.objects[(self.bucket, key)] = path.read_bytes()
        if self.before_write:
            self.before_write()

    def remove(self, bucket, key):
        self.removals += 1
        self.objects.pop((bucket, key), None)


def clone_permission_tables(schema):
    # 导入实体以注册外键；只克隆权限接口实际使用的表及其完整依赖闭包。
    from zhiqing_rag.modules import chat, documents, identity, knowledge, retrieval  # noqa: F401
    from zhiqing_rag.modules.documents.document_access_grant import DocumentAccessGrant

    required = {
        "user_account", "tenant", "department", "tenant_member", "auth_session",
        "audit_event", "knowledge_base", "knowledge_base_grant", "document",
        "document_revision", "index_generation", "embedding_profile", "background_task",
        "outbox_event", "task_attempt", "graph_projection", DocumentAccessGrant.__tablename__,
    }
    pending = list(required)
    while pending:
        table = Base.metadata.tables[f"{SCHEMA}.{pending.pop()}"]
        for foreign_key in table.foreign_keys:
            dependency = foreign_key.column.table.name
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    assert "document_chunk" not in required, "权限测试无需向量或 BM25 索引"
    metadata = MetaData(schema=schema)
    tables = {}
    for name in sorted(required):
        tables[name] = Base.metadata.tables[f"{SCHEMA}.{name}"].to_metadata(
            metadata, schema=schema, referred_schema_fn=lambda *_args: schema,
        )
    return metadata, tables


@pytest.fixture
def permission_environment(monkeypatch):
    suffix = uuid4().hex
    schema = f"permission_test_{suffix}"
    original = get_settings()
    prefix = f"{original.redis_key_prefix}permission-test:{suffix}:"
    settings = original.model_copy(update={
        "app_env": "test", "db_schema": schema, "redis_key_prefix": prefix,
        "jwt_secret_key": SecretStr(uuid4().hex + uuid4().hex),
        "refresh_cookie_secure": False,
    })
    engine = create_engine(settings.database_url)
    cache = Redis(
        host=settings.redis_host, port=settings.redis_port, db=settings.redis_db,
        password=settings.redis_password.get_secret_value() or None,
        socket_connect_timeout=settings.service_timeout_seconds,
        socket_timeout=settings.service_timeout_seconds, decode_responses=True,
    )
    storage = MemoryObjectStorage()
    schema_created = False
    cache_connected = False
    try:
        assert cache.ping(), "Redis 必须可用"
        cache_connected = True
        metadata, tables = clone_permission_tables(schema)
        password_hash = hash_password(PASSWORD)
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
            metadata.create_all(connection)

            def insert(table_name, **values):
                return connection.execute(
                    tables[table_name].insert().values(**values).returning(tables[table_name].c.id),
                ).scalar_one()

            tenants = [insert("tenant", code=f"T{index}", name=f"权限租户 {index}")
                       for index in range(2)]
            departments = {
                "a": insert("department", tenant_id=tenants[0], code="A", name="部门 A"),
                "b": insert("department", tenant_id=tenants[0], code="B", name="部门 B"),
                "foreign": insert("department", tenant_id=tenants[1], code="F", name="外租户部门"),
            }
            departments["child"] = insert(
                "department", tenant_id=tenants[0], code="A_CHILD", name="A 下属部门",
                parent_id=departments["a"],
            )
            users = {}
            definitions = [
                ("owner", "a", 2, "ADMIN"), ("reader", "a", 1, "MEMBER"),
                ("writer_b", "b", 2, "MEMBER"), ("reader_b", "b", 2, "MEMBER"),
                ("admin_b", "b", 2, "ADMIN"), ("low", "b", 0, "MEMBER"),
                ("child", "child", 2, "MEMBER"), ("none_owner", None, 2, "MEMBER"),
                ("none_other", None, 2, "MEMBER"), ("outsider", "b", 2, "ADMIN"),
                ("foreign", "foreign", 2, "ADMIN"),
            ]
            for name, department, clearance, role in definitions:
                user_id = insert(
                    "user_account", email=f"{name}-{suffix}@permission.test",
                    display_name=name, password_hash=password_hash,
                )
                tenant_id = tenants[1] if name == "foreign" else tenants[0]
                department_id = departments[department] if department else None
                member_id = insert(
                    "tenant_member", tenant_id=tenant_id, user_id=user_id,
                    department_id=department_id, clearance=clearance, role=role,
                )
                users[name] = SimpleNamespace(
                    user_id=user_id, member_id=member_id, tenant_id=tenant_id,
                    department_id=department_id, clearance=clearance,
                    email=f"{name}-{suffix}@permission.test",
                )
            kb_id = insert(
                "knowledge_base", tenant_id=tenants[0], code="SHARED", name="部门知识库",
                created_by=users["owner"].member_id,
            )
            other_kb_id = insert(
                "knowledge_base", tenant_id=tenants[0], code="OTHER", name="独立知识库",
                created_by=users["owner"].member_id,
            )
            foreign_kb_id = insert(
                "knowledge_base", tenant_id=tenants[1], code="FOREIGN", name="外租户知识库",
                created_by=users["foreign"].member_id,
            )
            grants = {}
            for name, permission in {
                "owner": "ADMIN", "reader": "READ", "writer_b": "WRITE",
                "reader_b": "READ", "admin_b": "ADMIN", "low": "WRITE",
                "none_owner": "ADMIN", "none_other": "READ",
            }.items():
                grants[name] = insert(
                    "knowledge_base_grant", tenant_id=tenants[0], kb_id=kb_id,
                    user_id=users[name].user_id, permission=permission,
                    granted_by=users["owner"].member_id,
                )
            grants["department_a"] = insert(
                "knowledge_base_grant", tenant_id=tenants[0], kb_id=kb_id,
                department_id=departments["a"], permission="READ",
                granted_by=users["owner"].member_id,
            )
            for tenant_id, selected_kb, actor in (
                (tenants[0], other_kb_id, users["owner"]),
                (tenants[1], foreign_kb_id, users["foreign"]),
            ):
                insert("knowledge_base_grant", tenant_id=tenant_id, kb_id=selected_kb,
                       user_id=actor.user_id, permission="ADMIN", granted_by=actor.member_id)
            profile_id = insert(
                "embedding_profile", code="PERMISSION_TEST", provider="dashscope",
                model_name=settings.embedding_model, model_revision="test", dimensions=1024,
            )
            document_definitions = [
                ("department", "owner", kb_id, 1), ("secret", "owner", kb_id, 2),
                ("public", "owner", kb_id, 0), ("none", "none_owner", kb_id, 1),
                ("department_b", "admin_b", kb_id, 1),
                ("other_kb", "owner", other_kb_id, 0),
                ("foreign", "foreign", foreign_kb_id, 1),
            ]
            documents = {}
            for name, owner, selected_kb, confidentiality in document_definitions:
                actor = users[owner]
                document_id = insert(
                    "document", tenant_id=actor.tenant_id, kb_id=selected_kb,
                    title=f"{name}.md", department_id=actor.department_id,
                    confidentiality=confidentiality, created_by=actor.member_id,
                )
                key = f"{schema}/{name}.md"
                storage.objects[(storage.bucket, key)] = CONTENT
                revision_id = insert(
                    "document_revision", tenant_id=actor.tenant_id, document_id=document_id,
                    revision_no=1, original_filename=f"{name}.md", mime_type="text/markdown",
                    file_size=len(CONTENT), file_sha256=sha256(CONTENT).hexdigest(),
                    bucket=storage.bucket, object_key=key, uploaded_by=actor.member_id,
                )
                now = datetime.now(UTC)
                generation_id = insert(
                    "index_generation", tenant_id=actor.tenant_id, document_id=document_id,
                    revision_id=revision_id, embedding_profile_id=profile_id, generation_no=1,
                    parser_version="test", cleaner_version="test", chunker_version="test",
                    status="READY", chunk_count=1, ready_at=now,
                )
                connection.execute(tables["document"].update().where(
                    tables["document"].c.id == document_id,
                ).values(status="PUBLISHED", desired_revision_id=revision_id,
                         active_generation_id=generation_id))
                task_id = insert(
                    "background_task", tenant_id=actor.tenant_id, task_type="INDEX",
                    document_id=document_id, revision_id=revision_id, generation_id=generation_id,
                    idempotency_key=f"permission-test:{name}", status="SUCCEEDED",
                    finished_at=now, progress=100, stage="COMPLETED",
                    payload={"pipeline": "document-ingestion-v1"},
                )
                projection_id = insert(
                    "graph_projection", tenant_id=actor.tenant_id, document_id=document_id,
                    revision_id=revision_id, generation_id=generation_id, projection_version=1,
                    graph_namespace=f"permission-test-{suffix}", extractor_version="test",
                    model_name="test", status="READY", ready_at=now,
                )
                documents[name] = SimpleNamespace(
                    id=document_id, revision_id=revision_id, generation_id=generation_id,
                    task_id=task_id, projection_id=projection_id, kb_id=selected_kb,
                    tenant_id=actor.tenant_id,
                )
        schema_created = True

        @contextmanager
        def isolated_storage(_settings):
            yield storage

        monkeypatch.setattr(document_router, "document_object_storage", isolated_storage)
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings
        yield SimpleNamespace(
            settings=settings, schema=schema, prefix=prefix, redis=cache, engine=engine,
            app=app, tables=tables, users=users, departments=departments, grants=grants,
            documents=documents, kb_id=kb_id, other_kb_id=other_kb_id,
            foreign_kb_id=foreign_kb_id, storage=storage,
        )
    finally:
        assert schema == f"permission_test_{suffix}" and len(suffix) == 32
        try:
            if schema_created:
                with engine.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        finally:
            try:
                if cache_connected:
                    keys = [key for key in cache.scan_iter(match=prefix + "*")
                            if key.startswith(prefix)]
                    if keys:
                        cache.delete(*keys)
            finally:
                cache.close()
                engine.dispose()


def authenticated_client(environment, stack, name):
    user = environment.users[name]
    client = stack.enter_context(TestClient(environment.app))
    client.headers.update({"Origin": environment.settings.cors_origins[0],
                           "X-Requested-With": "XMLHttpRequest"})
    login = client.post("/api/auth/login", json={"email": user.email, "password": PASSWORD})
    assert login.status_code == 200, login.text
    selected = client.post("/api/auth/select-tenant", json={
        "selection_token": login.json()["selection_token"], "tenant_id": str(user.tenant_id),
    })
    assert selected.status_code == 200, selected.text
    client.headers["Authorization"] = "Bearer " + selected.json()["access_token"]
    return client


def query(environment, statement, parameters=(), *, all_rows=False):
    with database_connection(environment.settings) as connection:
        cursor = connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(environment.schema)), parameters,
        )
        if cursor.description is None:
            return None
        return cursor.fetchall() if all_rows else cursor.fetchone()


def content_url(document):
    return f"/api/documents/{document.id}/content?revision_id={document.revision_id}"


def listed_document_ids(client):
    response = client.get("/api/document-tasks/page", params={"limit": 100})
    assert response.status_code == 200, response.text
    return {str(item["document_id"]) for item in response.json()["items"]}


def denied(response):
    assert response.status_code in {403, 404}, response.text


def share(client, document, *, user_id=None, department_id=None, permission="READ"):
    body = {"permission": permission}
    if user_id is not None:
        body["user_id"] = str(user_id)
    if department_id is not None:
        body["department_id"] = str(department_id)
    response = client.post(f"/api/documents/{document.id}/permissions", json=body)
    assert response.status_code == 200, response.text
    matching = [grant for grant in response.json()["grants"]
                if grant["user_id"] == (str(user_id) if user_id is not None else None)
                and grant["department_id"] == (
                    str(department_id) if department_id is not None else None)]
    assert len(matching) == 1
    return matching[0]


def update_document(client, document):
    return client.post(
        f"/api/documents/{document.id}/revisions",
        files={"file": ("updated.md", b"# Updated isolated content", "text/markdown")},
        data={"expected_revision_id": str(document.revision_id)},
    )


def test_read_permission_opens_same_department_lists_preview_and_download(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        reader = authenticated_client(environment, stack, "reader")
        assert str(document.id) in listed_document_ids(reader)
        assert str(environment.documents["secret"].id) not in listed_document_ids(reader)
        assert reader.get(content_url(document)).content == CONTENT
        preview = reader.get(
            f"/api/documents/{document.id}/preview", params={"revision_id": document.revision_id},
        )
        assert preview.status_code == 200, preview.text
        assert reader.get(f"/api/documents/{document.id}/revisions").status_code == 200
        assert reader.get(f"/api/document-tasks/{document.task_id}").status_code == 200
        denied(update_document(reader, document))
        denied(reader.post(f"/api/document-tasks/{document.task_id}/retry"))
        denied(reader.delete(f"/api/documents/{document.id}"))
        assert environment.storage.writes == 0 and environment.storage.removals == 0


def test_kb_write_does_not_cross_department_document_scope(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        writer = authenticated_client(environment, stack, "writer_b")
        assert str(document.id) not in listed_document_ids(writer)
        denied(writer.get(content_url(document)))
        denied(update_document(writer, document))
        denied(writer.delete(f"/api/documents/{document.id}"))
        assert environment.storage.reads == 0 and environment.storage.writes == 0


def test_read_share_cannot_write_even_with_kb_write(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        writer = authenticated_client(environment, stack, "writer_b")
        share(owner, document, department_id=environment.departments["b"])
        assert writer.get(content_url(document)).status_code == 200
        assert str(document.id) in listed_document_ids(writer)
        denied(update_document(writer, document))
        denied(writer.post(f"/api/document-tasks/{document.task_id}/retry"))
        denied(writer.delete(f"/api/documents/{document.id}"))
        assert environment.storage.writes == 0 and environment.storage.removals == 0


def test_write_share_allows_revision_without_changing_owner_department(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        writer = authenticated_client(environment, stack, "writer_b")
        share(owner, document, user_id=environment.users["writer_b"].user_id, permission="WRITE")
        result = update_document(writer, document)
        assert result.status_code == 202, result.text
        assert result.json()["document_id"] == str(document.id)
        row = query(environment, "SELECT * FROM {schema}.document WHERE id = %s", (document.id,))
        assert row["department_id"] == environment.departments["a"]
        assert row["created_by"] == environment.users["owner"].member_id
        assert row["confidentiality"] == 1
        denied(writer.get(f"/api/documents/{document.id}/permissions"))


def test_share_revocation_is_immediate_preserves_access_login_and_index(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        reader = authenticated_client(environment, stack, "reader_b")
        before = query(environment, "SELECT acl_version FROM {schema}.knowledge_base WHERE id=%s",
                       (environment.kb_id,))["acl_version"]
        grant = share(owner, document, user_id=environment.users["reader_b"].user_id)
        assert reader.get(content_url(document)).status_code == 200
        revoke = owner.delete(f"/api/documents/{document.id}/permissions/{grant['id']}")
        assert revoke.status_code == 200, revoke.text
        denied(reader.get(content_url(document)))
        assert reader.get("/api/auth/me").status_code == 200
        after = query(environment, "SELECT acl_version FROM {schema}.knowledge_base WHERE id=%s",
                      (environment.kb_id,))["acl_version"]
        assert after > before
        row = query(environment, "SELECT active_generation_id FROM {schema}.document WHERE id=%s",
                    (document.id,))
        assert row["active_generation_id"] == document.generation_id
        assert query(environment, "SELECT count(*) AS n FROM {schema}.index_generation")["n"] == 7
        assert query(environment, "SELECT count(*) AS n FROM {schema}.audit_event")["n"] >= 2


def test_share_requires_kb_permission_and_cannot_override_clearance(permission_environment):
    environment = permission_environment
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        outsider = authenticated_client(environment, stack, "outsider")
        low = authenticated_client(environment, stack, "low")
        share(owner, environment.documents["department"],
              user_id=environment.users["outsider"].user_id)
        share(owner, environment.documents["secret"], user_id=environment.users["low"].user_id,
              permission="WRITE")
        denied(outsider.get(content_url(environment.documents["department"])))
        denied(low.get(content_url(environment.documents["secret"])))
        denied(update_document(low, environment.documents["secret"]))
        assert outsider.get("/api/auth/me").status_code == 200


def test_public_still_requires_acl_and_parent_department_does_not_inherit(permission_environment):
    environment = permission_environment
    with ExitStack() as stack:
        outsider = authenticated_client(environment, stack, "outsider")
        child = authenticated_client(environment, stack, "child")
        writer = authenticated_client(environment, stack, "writer_b")
        owner = authenticated_client(environment, stack, "owner")
        denied(outsider.get(content_url(environment.documents["public"])))
        denied(child.get(content_url(environment.documents["public"])))
        assert writer.get(content_url(environment.documents["public"])).status_code == 200
        assert listed_document_ids(child) == set()
        explicit_kb_read = owner.post(
            f"/api/knowledge-bases/{environment.kb_id}/permissions",
            json={"user_id": str(environment.users["child"].user_id), "permission": "READ"},
        )
        assert explicit_kb_read.status_code == 200, explicit_kb_read.text
        assert child.get(content_url(environment.documents["public"])).status_code == 200
        denied(child.get(content_url(environment.documents["department"])))


def test_null_department_allows_only_author_or_explicit_share(permission_environment):
    environment = permission_environment
    document = environment.documents["none"]
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "none_owner")
        other = authenticated_client(environment, stack, "none_other")
        assert owner.get(content_url(document)).status_code == 200
        denied(other.get(content_url(document)))
        share(owner, document, user_id=environment.users["none_other"].user_id)
        assert other.get(content_url(document)).status_code == 200


def test_permissions_admin_cannot_cross_document_department_or_tenant(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        admin_b = authenticated_client(environment, stack, "admin_b")
        kb_view = admin_b.get(f"/api/knowledge-bases/{environment.kb_id}/permissions")
        assert kb_view.status_code == 200
        denied(admin_b.get(f"/api/documents/{document.id}/permissions"))
        denied(admin_b.post(f"/api/documents/{document.id}/permissions", json={
            "user_id": str(environment.users["reader_b"].user_id), "permission": "READ",
        }))
        denied(owner.get(f"/api/documents/{environment.documents['foreign'].id}/permissions"))
        cross_tenant = owner.post(f"/api/documents/{document.id}/permissions", json={
            "user_id": str(environment.users["foreign"].user_id), "permission": "READ",
        })
        assert cross_tenant.status_code in {400, 403, 404, 422}, cross_tenant.text
        shares = query(environment, "SELECT count(*) AS n FROM {schema}.document_access_grant")
        assert shares["n"] == 0


def test_last_knowledge_base_admin_cannot_be_removed(permission_environment):
    environment = permission_environment
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        endpoint = f"/api/knowledge-bases/{environment.kb_id}/permissions"
        response = owner.get(endpoint)
        assert response.status_code == 200, response.text
        for grant in response.json()["grants"]:
            if grant["permission"] == "ADMIN" and grant["user_id"] != str(
                environment.users["owner"].user_id
            ):
                removed = owner.delete(f"{endpoint}/{grant['id']}")
                assert removed.status_code == 200, removed.text
        final = owner.delete(f"{endpoint}/{environment.grants['owner']}")
        assert final.status_code == 409, final.text
        current = owner.get(endpoint)
        assert current.status_code == 200
        assert any(grant["permission"] == "ADMIN" for grant in current.json()["grants"])


def test_classification_changes_recheck_existing_tokens_without_reembedding(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        reader = authenticated_client(environment, stack, "reader")
        assert reader.get(content_url(document)).status_code == 200
        changed = owner.post(f"/api/documents/{document.id}/classification",
                             json={"confidentiality": 2})
        assert changed.status_code == 200, changed.text
        denied(reader.get(content_url(document)))
        row = query(environment, "SELECT active_generation_id, confidentiality "
                    "FROM {schema}.document WHERE id=%s", (document.id,))
        assert row["active_generation_id"] == document.generation_id and row["confidentiality"] == 2
        assert query(environment, "SELECT count(*) AS n FROM {schema}.index_generation")["n"] == 7
        assert query(environment, "SELECT count(*) AS n FROM {schema}.audit_event")["n"] >= 1


def test_new_upload_defaults_to_department_classification_not_user_clearance(
    permission_environment,
):
    environment = permission_environment
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        uploaded = owner.post(
            "/api/documents/upload", files={"file": ("new.md", CONTENT, "text/markdown")},
            data={"knowledge_base_id": str(environment.kb_id)},
        )
        assert uploaded.status_code == 202, uploaded.text
        row = query(environment, "SELECT confidentiality, department_id FROM {schema}.document "
                    "WHERE id=%s", (int(uploaded.json()["document_id"]),))
        assert environment.users["owner"].clearance == 2 and row["confidentiality"] == 1
        assert row["department_id"] == environment.departments["a"]


def test_orm_authorization_matches_sql_and_rejects_cross_department_graph_scope(
    permission_environment,
):
    from zhiqing_rag.modules.documents import Document
    from zhiqing_rag.modules.knowledge.permissions import document_access_condition

    environment = permission_environment
    user = environment.users["writer_b"]
    identity = UploadIdentity(user.tenant_id, user.member_id, user.user_id,
                              user.department_id, user.clearance)
    translated = environment.engine.execution_options(
        schema_translate_map={SCHEMA: environment.schema},
    )
    with Session(translated) as session:
        visible = set(session.scalars(select(Document.id).where(
            document_access_condition(identity),
        )))
    with ExitStack() as stack:
        writer = authenticated_client(environment, stack, "writer_b")
        assert {str(value) for value in visible} == listed_document_ids(writer)
        assert environment.documents["department"].id not in visible
        assert environment.documents["public"].id in visible


def test_department_transfer_removes_authors_implicit_department_access(permission_environment):
    environment = permission_environment
    document = environment.documents["department"]
    with ExitStack() as stack:
        owner_before = authenticated_client(environment, stack, "owner")
        assert owner_before.get(content_url(document)).status_code == 200
        query(environment, "UPDATE {schema}.tenant_member SET department_id=%s, "
              "auth_version=auth_version+1 WHERE id=%s",
              (environment.departments["b"], environment.users["owner"].member_id))
        assert owner_before.get("/api/auth/me").status_code == 401
        owner_after = authenticated_client(environment, stack, "owner")
        assert owner_after.get(
            f"/api/knowledge-bases/{environment.kb_id}/permissions",
        ).status_code == 200
        denied(owner_after.get(content_url(document)))
        denied(owner_after.get(f"/api/documents/{document.id}/permissions"))
        assert str(document.id) not in listed_document_ids(owner_after)
        row = query(environment, "SELECT department_id, created_by FROM {schema}.document "
                    "WHERE id=%s", (document.id,))
        assert row["department_id"] == environment.departments["a"]
        assert row["created_by"] == environment.users["owner"].member_id


def test_authorized_graph_bridge_uses_live_acl_and_current_published_generation(
    permission_environment,
):
    from zhiqing_rag.modules.retrieval.graph_access import authorized_graph_projection

    environment = permission_environment
    document = environment.documents["department"]
    user = environment.users["writer_b"]
    identity = UploadIdentity(user.tenant_id, user.member_id, user.user_id,
                              user.department_id, user.clearance)
    translated = environment.engine.execution_options(
        schema_translate_map={SCHEMA: environment.schema},
    )

    def visible_projection():
        with Session(translated) as session:
            return authorized_graph_projection(session, identity, document.id)

    assert visible_projection() is None
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        grant = share(owner, document, user_id=user.user_id)
        projection = visible_projection()
        assert projection is not None and projection.generation_id == document.generation_id
        suffix = environment.schema.removeprefix("permission_test_")
        assert projection.graph_namespace.endswith(suffix)
        query(environment, "UPDATE {schema}.document SET status='DRAFT' WHERE id=%s",
              (document.id,))
        assert visible_projection() is None
        query(environment, "UPDATE {schema}.document SET status='PUBLISHED' WHERE id=%s",
              (document.id,))
        revoked = owner.delete(f"/api/documents/{document.id}/permissions/{grant['id']}")
        assert revoked.status_code == 200, revoked.text
        assert visible_projection() is None


def test_worker_rechecks_write_scope_after_department_transfer_and_share_revocation(
    permission_environment,
):
    from zhiqing_rag.modules.documents.task_queue import ProcessingError
    from zhiqing_rag.workers.publication import verify_work_access

    environment = permission_environment
    document = environment.documents["department"]
    lease = SimpleNamespace(tenant_id=document.tenant_id, document_id=document.id,
                            revision_id=document.revision_id)
    translated = environment.engine.execution_options(
        schema_translate_map={SCHEMA: environment.schema},
    )
    verify_work_access(translated, lease)
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        reader = authenticated_client(environment, stack, "reader")
        promoted = owner.post(f"/api/knowledge-bases/{environment.kb_id}/permissions", json={
            "user_id": str(environment.users["reader"].user_id), "permission": "ADMIN",
        })
        assert promoted.status_code == 200, promoted.text
        query(environment, "UPDATE {schema}.tenant_member SET department_id=%s, "
              "auth_version=auth_version+1 WHERE id=%s",
              (environment.departments["b"], environment.users["owner"].member_id))
        with pytest.raises(ProcessingError, match="ACCESS_REVOKED"):
            verify_work_access(translated, lease)
        share(reader, document, user_id=environment.users["owner"].user_id)
        with pytest.raises(ProcessingError, match="ACCESS_REVOKED"):
            verify_work_access(translated, lease)
        grant = share(reader, document, user_id=environment.users["owner"].user_id,
                      permission="WRITE")
        verify_work_access(translated, lease)
        revoked = reader.delete(f"/api/documents/{document.id}/permissions/{grant['id']}")
        assert revoked.status_code == 200, revoked.text
        with pytest.raises(ProcessingError, match="ACCESS_REVOKED"):
            verify_work_access(translated, lease)
        assert query(environment, "SELECT active_generation_id FROM {schema}.document "
                     "WHERE id=%s", (document.id,))["active_generation_id"] == (
            document.generation_id
        )


def test_worker_takes_knowledge_base_lock_before_document_lock(permission_environment):
    from zhiqing_rag.workers.publication import verify_work_access

    environment = permission_environment
    document = environment.documents["department"]
    lease = SimpleNamespace(tenant_id=document.tenant_id, document_id=document.id,
                            revision_id=document.revision_id)
    worker_engine = create_engine(environment.settings.database_url, connect_args={
        "options": "-c lock_timeout=8000 -c statement_timeout=12000",
    })
    translated = worker_engine.execution_options(
        schema_translate_map={SCHEMA: environment.schema},
    )
    worker_started = Event()
    worker_pids = []

    def verify_in_thread():
        with translated.connect() as connection:
            worker_pids.append(connection.exec_driver_sql("SELECT pg_backend_pid()").scalar_one())
            connection.rollback()
            worker_started.set()
            verify_work_access(connection, lease)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with database_connection(environment.settings) as holder, holder.transaction():
                holder.execute(sql.SQL(
                    "SELECT id FROM {}.knowledge_base WHERE id=%s FOR UPDATE",
                ).format(sql.Identifier(environment.schema)), (document.kb_id,)).fetchone()
                future = executor.submit(verify_in_thread)
                assert worker_started.wait(3), "Worker 未能连接隔离数据库"
                deadline = monotonic() + 5
                while True:
                    blocked = holder.execute(
                        "SELECT %s = ANY(pg_blocking_pids(%s)) AS blocked",
                        (holder.info.backend_pid, worker_pids[0]),
                    ).fetchone()["blocked"]
                    if blocked:
                        break
                    if future.done():
                        future.result()
                        pytest.fail("Worker 未等待知识库权限变更锁")
                    assert monotonic() < deadline, "Worker 未在预期时间阻塞于知识库锁"
                    sleep(0.02)
                # 复现权限管理的 KB→doc 顺序；Worker 若先持 doc SHARE，此处 NOWAIT 必败。
                locked_document = holder.execute(sql.SQL(
                    "SELECT id FROM {}.document WHERE id=%s FOR UPDATE NOWAIT",
                ).format(sql.Identifier(environment.schema)), (document.id,)).fetchone()
                assert locked_document["id"] == document.id
            # 上层事务释放 KB 后，Worker 应完成验权，而不是等到超时或数据库死锁检测。
            future.result(timeout=10)
    finally:
        worker_engine.dispose()


def test_worker_uses_current_revision_uploader_after_original_author_transfers_department(
    permission_environment,
):
    from zhiqing_rag.modules.documents.task_queue import ProcessingError
    from zhiqing_rag.workers.publication import verify_work_access

    environment = permission_environment
    document = environment.documents["department"]
    translated = environment.engine.execution_options(
        schema_translate_map={SCHEMA: environment.schema},
    )
    with ExitStack() as stack:
        owner = authenticated_client(environment, stack, "owner")
        manager = authenticated_client(environment, stack, "reader")
        writer = authenticated_client(environment, stack, "writer_b")
        promoted = owner.post(f"/api/knowledge-bases/{environment.kb_id}/permissions", json={
            "user_id": str(environment.users["reader"].user_id), "permission": "ADMIN",
        })
        assert promoted.status_code == 200, promoted.text
        grant = share(owner, document, user_id=environment.users["writer_b"].user_id,
                      permission="WRITE")
        query(environment, "UPDATE {schema}.tenant_member SET department_id=%s, "
              "auth_version=auth_version+1 WHERE id=%s",
              (environment.departments["b"], environment.users["owner"].member_id))
        old_lease = SimpleNamespace(tenant_id=document.tenant_id, document_id=document.id,
                                    revision_id=document.revision_id)
        with pytest.raises(ProcessingError, match="ACCESS_REVOKED"):
            verify_work_access(translated, old_lease)
        uploaded = update_document(writer, document)
        assert uploaded.status_code == 202, uploaded.text
        result = uploaded.json()
        assert result["document_id"] == str(document.id)
        revision_id = int(result["revision_id"])
        assert revision_id != document.revision_id
        revision = query(environment, "SELECT uploaded_by FROM {schema}.document_revision "
                         "WHERE document_id=%s AND id=%s", (document.id, revision_id))
        assert revision["uploaded_by"] == environment.users["writer_b"].member_id
        original_author = query(environment, "SELECT created_by, department_id FROM "
                                "{schema}.document WHERE id=%s", (document.id,))
        assert original_author["created_by"] == environment.users["owner"].member_id
        assert original_author["department_id"] == environment.departments["a"]
        current_lease = SimpleNamespace(tenant_id=document.tenant_id, document_id=document.id,
                                        revision_id=revision_id)
        verify_work_access(translated, current_lease)
        revoked = manager.delete(f"/api/documents/{document.id}/permissions/{grant['id']}")
        assert revoked.status_code == 200, revoked.text
        with pytest.raises(ProcessingError, match="ACCESS_REVOKED"):
            verify_work_access(translated, current_lease)
