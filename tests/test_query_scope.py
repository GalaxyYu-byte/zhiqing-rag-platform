"""真实 SQL 门禁在独立 SQLite 数据集上的租户、部门、密级和批次验证。"""

import re
import sqlite3
from types import SimpleNamespace

import pytest

from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_routing.models import PlanQueryBody
from zhiqing_rag.query_routing.scope import QueryScopeRepository

IDENTITY = SimpleNamespace(tenant_id=1, user_id=3, member_id=4, department_id=10, clearance=1)


class SQLiteScopeRepository(QueryScopeRepository):
    def query(self, statement, parameters=()):
        # 仅适配占位符和 ANY 数组，授权谓词直接复用生产实现。
        statement = statement.replace("{schema}", "zhiqing_rag")
        statement = statement.replace("= ANY(%s)", "IN (%s)")
        values = iter(parameters)
        flattened = []

        def replace(match):
            value = next(values)
            if isinstance(value, list):
                flattened.extend(value)
                return ",".join("?" for _ in value)
            flattened.append(value)
            return "?"

        statement = re.sub("%s", replace, statement)
        return self.connection.execute(statement, flattened)


@pytest.fixture
def repository():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("ATTACH DATABASE ':memory:' AS zhiqing_rag")
    definitions = {
        "knowledge_base": "id INTEGER, tenant_id INTEGER, status TEXT, deleted_at TEXT",
        "knowledge_base_grant": "tenant_id INTEGER, kb_id INTEGER, user_id INTEGER, "
        "department_id INTEGER, permission TEXT",
        "document": "id INTEGER, tenant_id INTEGER, kb_id INTEGER, status TEXT, deleted_at TEXT, "
        "department_id INTEGER, confidentiality INTEGER, created_by INTEGER, "
        "active_generation_id INTEGER",
        "document_access_grant": "tenant_id INTEGER, document_id INTEGER, user_id INTEGER, "
        "department_id INTEGER, permission TEXT",
        "index_generation": "id INTEGER, tenant_id INTEGER, document_id INTEGER, status TEXT",
    }
    for name, fields in definitions.items():
        connection.execute(f"CREATE TABLE zhiqing_rag.{name} ({fields})")
    connection.executemany(
        "INSERT INTO zhiqing_rag.knowledge_base VALUES (?,?,?,?)",
        [
            (11, 1, "ACTIVE", None),
            (12, 1, "ACTIVE", None),
            (13, 1, "INACTIVE", None),
            (21, 2, "ACTIVE", None),
        ],
    )
    connection.executemany(
        "INSERT INTO zhiqing_rag.knowledge_base_grant VALUES (?,?,?,?,?)",
        [
            (1, 11, 3, None, "READ"),
            (1, 12, None, 10, "WRITE"),
            (1, 13, 3, None, "ADMIN"),
            (2, 21, 3, None, "READ"),
        ],
    )
    rows = [
        (100, 1, 11, "PUBLISHED", None, 10, 1, 4, 200),
        (101, 1, 11, "PUBLISHED", None, 20, 1, 5, 201),
        (102, 1, 11, "PUBLISHED", None, 20, 1, 5, 202),
        (103, 1, 11, "PUBLISHED", None, 20, 0, 5, 203),
        (104, 1, 11, "PUBLISHED", None, 10, 2, 4, 204),
        (105, 1, 11, "DRAFT", None, 10, 1, 4, None),
        (106, 1, 11, "WITHDRAWN", None, 10, 1, 4, 206),
        (107, 1, 11, "DELETED", "deleted", 10, 1, 4, 207),
        (108, 1, 11, "PUBLISHED", None, 10, 1, 4, 208),
        (109, 1, 11, "PUBLISHED", None, 10, 1, 4, 200),
        (110, 1, 12, "PUBLISHED", None, 10, 1, 4, 210),
        (210, 2, 21, "PUBLISHED", None, 10, 1, 4, 300),
    ]
    connection.executemany("INSERT INTO zhiqing_rag.document VALUES (?,?,?,?,?,?,?,?,?)", rows)
    connection.execute("INSERT INTO zhiqing_rag.document_access_grant VALUES (1,102,3,NULL,'READ')")
    generations = [(row[8], row[1], row[0], "READY") for row in rows if row[8] and row[0] != 109]
    connection.executemany("INSERT INTO zhiqing_rag.index_generation VALUES (?,?,?,?)", generations)
    connection.execute("UPDATE zhiqing_rag.index_generation SET status='BUILDING' WHERE id=208")
    try:
        yield SQLiteScopeRepository(connection, "zhiqing_rag")
    finally:
        connection.close()


def test_scope_combines_kb_acl_department_document_acl_and_clearance(repository):
    scope = repository.resolve(IDENTITY, PlanQueryBody(query="问题"))
    assert scope.knowledge_base_ids == (11, 12)
    assert scope.document_ids == (100, 102, 103, 105, 108, 109, 110)
    assert [item.document_id for item in scope.published_documents] == [100, 102, 103, 110]
    assert [item.generation_id for item in scope.published_documents] == [200, 202, 203, 210]


@pytest.mark.parametrize(
    "body",
    [
        {"knowledge_base_ids": ["21"]},
        {"knowledge_base_ids": ["13"]},
        {"document_ids": ["101"]},
        {"document_ids": ["104"]},
        {"document_ids": ["106"]},
        {"document_ids": ["107"]},
        {"knowledge_base_ids": ["11"], "document_ids": ["110"]},
    ],
)
def test_inaccessible_explicit_resources_fail_instead_of_silent_intersection(repository, body):
    with pytest.raises(AuthError) as error:
        repository.resolve(IDENTITY, PlanQueryBody(query="问题", **body))
    assert error.value.status_code == 403


def test_explicit_readable_document_narrows_scope(repository):
    scope = repository.resolve(
        IDENTITY,
        PlanQueryBody(
            query="问题",
            knowledge_base_ids=["11"],
            document_ids=["102"],
        ),
    )
    assert scope.knowledge_base_ids == (11,)
    assert scope.document_ids == (102,)


def test_no_grants_returns_empty_scope_and_cannot_approve_requested_document(repository):
    identity = SimpleNamespace(
        tenant_id=1, user_id=999, member_id=999, department_id=None, clearance=1
    )
    scope = repository.resolve(identity, PlanQueryBody(query="你好"))
    assert not scope.knowledge_base_ids and not scope.document_ids
    with pytest.raises(AuthError):
        repository.resolve(identity, PlanQueryBody(query="问题", document_ids=["100"]))
