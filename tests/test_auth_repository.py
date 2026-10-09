"""在内存 SQL 数据库验证合并 JOIN 的租户绑定和状态过滤，不访问业务库。"""

import sqlite3
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from zhiqing_rag.modules.identity.auth_repository import AuthRepository
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.auth_service import authorization_from_session


class MemoryConnection:
    def __init__(self):
        self.database = sqlite3.connect(":memory:")
        self.database.execute("ATTACH DATABASE ':memory:' AS auth_test")
        self.database.row_factory = sqlite3.Row
        self.calls = []

    def execute(self, statement, parameters):
        rendered = statement.as_string(None)
        self.calls.append((rendered, parameters))
        rows = self.database.execute(
            rendered.replace("%s", "?"),
            tuple(str(value) if isinstance(value, UUID) else value for value in parameters),
        )
        return MemoryCursor(rows)


class MemoryCursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        result = self.rows.fetchone()
        if result is None:
            return None
        row = dict(result)
        # sqlite 不原生返回 PostgreSQL 的 bool/timestamptz，适配至 psycopg 行类型。
        if "authorization_active" in row and row["authorization_active"] is not None:
            row["authorization_active"] = bool(row["authorization_active"])
        for name in ("expires_at", "revoked_at"):
            if row.get(name) is not None:
                row[name] = datetime.fromisoformat(row[name])
        return row


@pytest.fixture
def authorization():
    connection = MemoryConnection()
    connection.database.executescript("""
        CREATE TABLE auth_test.user_account (
            id INTEGER PRIMARY KEY, email TEXT, display_name TEXT, is_active BOOLEAN);
        CREATE TABLE auth_test.tenant (
            id INTEGER PRIMARY KEY, code TEXT, name TEXT, status TEXT, auth_version INTEGER);
        CREATE TABLE auth_test.tenant_member (
            id INTEGER PRIMARY KEY, user_id INTEGER, tenant_id INTEGER, department_id INTEGER,
            role TEXT, clearance INTEGER, status TEXT, auth_version INTEGER);
        CREATE TABLE auth_test.department (
            id INTEGER, tenant_id INTEGER, is_active BOOLEAN);
        CREATE TABLE auth_test.auth_session (
            session_key TEXT, tenant_id INTEGER, tenant_member_id INTEGER,
            revoked_at TEXT, expires_at TEXT);
        INSERT INTO auth_test.user_account VALUES (2, 'user@test', '用户', 1);
        INSERT INTO auth_test.tenant VALUES (1, 'unit', '租户', 'ACTIVE', 3);
        INSERT INTO auth_test.tenant_member VALUES (10, 2, 1, 7, 'MEMBER', 2, 'ACTIVE', 4);
        INSERT INTO auth_test.department VALUES (7, 1, 1);
    """)
    session_key = uuid4()
    expires_at = datetime.now(UTC) + timedelta(hours=1)
    connection.database.execute(
        "INSERT INTO auth_test.auth_session VALUES (?, 1, 10, NULL, ?)",
        (str(session_key), expires_at.isoformat()),
    )
    yield AuthRepository(connection, "auth_test"), session_key, expires_at
    connection.database.close()


def validate(repository, session_key):
    return authorization_from_session(repository, session_key, (2, 1, 10), 4, 3)


def test_session_and_authorization_use_one_query_with_current_fields(authorization):
    repository, session_key, expires_at = authorization
    row = validate(repository, session_key)
    assert (row["department_id"], row["clearance"], row["role"]) == (7, 2, "MEMBER")
    assert row["expires_at"] == expires_at
    assert len(repository.connection.calls) == 1
    statement, parameters = repository.connection.calls[0]
    assert parameters == (session_key,)
    assert str(session_key) not in statement
    assert "refresh_token_hash" not in statement and "password_hash" not in statement


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE user_account SET is_active = 0",
        "UPDATE tenant SET status = 'DISABLED'",
        "UPDATE tenant_member SET status = 'DISABLED'",
        "UPDATE department SET is_active = 0",
        "DELETE FROM department",
        "UPDATE department SET tenant_id = 99",
        "DELETE FROM user_account",
        "DELETE FROM tenant",
        "UPDATE tenant_member SET auth_version = 5",
        "UPDATE tenant SET auth_version = 4",
    ],
)
def test_disabled_or_changed_authorization_fails_without_losing_session(authorization, change):
    repository, session_key, _ = authorization
    verb, rest = change.split(" ", 1)
    statement = (
        f"{verb} auth_test.{rest}"
        if verb == "UPDATE"
        else change.replace(
            "FROM ",
            "FROM auth_test.",
            1,
        )
    )
    repository.connection.database.execute(statement)
    with pytest.raises(AuthError) as error:
        validate(repository, session_key)
    assert error.value.code == "AUTHORIZATION_CHANGED"
    assert len(repository.connection.calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        "DELETE FROM auth_test.auth_session",
        "DELETE FROM auth_test.tenant_member",
        "UPDATE auth_test.auth_session SET tenant_id = 99",
        "UPDATE auth_test.auth_session SET tenant_member_id = 99",
        "UPDATE auth_test.auth_session SET expires_at = '2000-01-01T00:00:00+00:00'",
        "UPDATE auth_test.auth_session SET revoked_at = '2000-01-01T00:00:00+00:00'",
    ],
)
def test_invalid_session_fails_before_authorization(authorization, change):
    repository, session_key, _ = authorization
    repository.connection.database.execute(change)
    repository.connection.database.execute("UPDATE auth_test.tenant SET status = 'DISABLED'")
    with pytest.raises(AuthError) as error:
        validate(repository, session_key)
    assert error.value.code == "SESSION_REVOKED"


@pytest.mark.parametrize("expected", [(99, 1, 10), (2, 99, 10), (2, 1, 99)])
def test_valid_session_cannot_authenticate_another_token_identity(authorization, expected):
    repository, session_key, _ = authorization
    with pytest.raises(AuthError) as error:
        authorization_from_session(repository, session_key, expected, 4, 3)
    assert error.value.code == "ACCESS_TOKEN_INVALID"


def test_null_department_is_allowed(authorization):
    repository, session_key, _ = authorization
    repository.connection.database.execute(
        "UPDATE auth_test.tenant_member SET department_id = NULL"
    )
    repository.connection.database.execute("DELETE FROM auth_test.department")
    assert validate(repository, session_key)["department_id"] is None


def test_repeated_checks_read_changed_state_instead_of_reusing_previous_row(authorization):
    repository, session_key, _ = authorization
    assert validate(repository, session_key)["clearance"] == 2
    repository.connection.database.execute(
        "UPDATE auth_test.tenant_member SET department_id = NULL, clearance = 0"
    )
    assert validate(repository, session_key)["clearance"] == 0
    repository.connection.database.execute(
        "UPDATE auth_test.auth_session SET revoked_at = '2000-01-01T00:00:00+00:00'"
    )
    with pytest.raises(AuthError) as error:
        validate(repository, session_key)
    assert error.value.code == "SESSION_REVOKED"
    assert len(repository.connection.calls) == 3
