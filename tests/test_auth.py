"""认证密码/JWT、Redis原子轮换、跨租户授权和HTTP安全边界。"""

import json
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import jwt
import psycopg
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from starlette.requests import Request

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.identity import router as auth_router
from zhiqing_rag.modules.identity.auth_schemas import AuthError, LoginRequest
from zhiqing_rag.modules.identity.auth_service import AuthService, identity_from_access
from zhiqing_rag.modules.identity.auth_store import (
    CONSUME_SCRIPT,
    RATE_SCRIPT,
    ROTATE_SCRIPT,
    AuthStore,
    token_hash,
)
from zhiqing_rag.modules.identity.auth_tokens import (
    ACCESS_TTL,
    REFRESH_TTL,
    decode_access_token,
    issue_access_token,
)
from zhiqing_rag.modules.identity.passwords import hash_password, verify_password

PASSWORD = "Strong-Unit-Test-Password!"
PASSWORD_HASH = hash_password(PASSWORD)


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.dead = False
        self.lock = threading.RLock()

    def _check(self):
        if self.dead:
            raise RedisConnectionError("secret redis password")

    def get(self, key):
        self._check()
        return self.values.get(key)

    def set(self, key, value, *, ex, nx=False):
        self._check()
        with self.lock:
            if nx and key in self.values:
                return False
            self.values[key] = value
            return True

    def delete(self, *keys):
        self._check()
        for key in keys:
            self.values.pop(key, None)

    def eval(self, script, key_count, *values):
        self._check()
        with self.lock:
            keys, args = values[:key_count], values[key_count:]
            if script == RATE_SCRIPT:
                count = int(self.values.get(keys[0], 0)) + 1
                self.values[keys[0]] = str(count)
                return count
            if script == CONSUME_SCRIPT:
                if self.values.get(keys[0]) != args[0]:
                    return 0
                self.values.pop(keys[0])
                return 1
            if script == ROTATE_SCRIPT:
                if self.values.get(keys[0]) != args[0] or keys[1] in self.values:
                    return 0
                self.values[keys[1]] = args[1]
                self.values.pop(keys[0])
                return 1
            raise AssertionError("Unknown script")


class FakeConnection:
    def __init__(self, repository):
        self.repository = repository
        self.lock = threading.RLock()
        self.fail_commit = False

    @contextmanager
    def transaction(self):
        with self.lock:
            snapshot = deepcopy(self.repository.sessions)
            try:
                yield
            except Exception:
                self.repository.sessions = snapshot
                raise

    def commit(self):
        if self.fail_commit:
            raise psycopg.OperationalError("secret database password")


class FakeRepository:
    def __init__(self):
        self.user = {
            "id": 9007199254740993, "email": "member@example.test", "display_name": "用户",
            "password_hash": PASSWORD_HASH, "is_active": True,
        }
        self.rows = {
            tid: {
                "user_id": self.user["id"], "email": self.user["email"],
                "display_name": self.user["display_name"], "tenant_id": tid,
                "member_id": tid + 10, "department_id": None, "role": "MEMBER",
                "clearance": 1, "auth_version": 1, "tenant_auth_version": 1,
                "tenant_code": f"T{tid}", "tenant_name": f"租户{tid}",
            } for tid in (1, 2)
        }
        self.sessions = {}
        self.next_session_id = 1
        self.connection = FakeConnection(self)

    def user_by_email(self, email):
        return dict(self.user) if email == self.user["email"] else None

    def lock_user(self, user_id):
        return dict(self.user) if user_id == self.user["id"] else None

    def revocation_stamp(self, user_id):
        values = [row for row in self.sessions.values()
                  if row["user_id"] == user_id and row["revoked_at"]]
        return (f"{max(row['revoked_at'] for row in values).isoformat()}:"
                f"{max(row['id'] for row in values)}:{len(values)}") if values else ""

    def tenants(self, user_id):
        return [{"id": row["tenant_id"], "code": row["tenant_code"],
                 "name": row["tenant_name"], "role": row["role"]}
                for row in self.rows.values() if row["user_id"] == user_id]

    def identity(self, user_id, tenant_id, member_id=None):
        row = self.rows.get(tenant_id)
        if (not row or not self.user["is_active"] or row["user_id"] != user_id or
                (member_id is not None and row["member_id"] != member_id)):
            return None
        return dict(row)

    def session(self, session_key, *, lock=False):
        row = self.sessions.get(session_key)
        return dict(row) if row else None

    def session_identity(self, session_key):
        session = self.session(session_key)
        row = self.identity(
            session["user_id"], session["tenant_id"], session["tenant_member_id"],
        ) if session else None
        return session, row

    def create_session(self, row, session_key, token_hash, created_at, expires_at,
                       user_agent, ip_address):
        self.sessions[session_key] = {
            "id": self.next_session_id,
            "session_key": session_key, "user_id": row["user_id"],
            "tenant_id": row["tenant_id"], "tenant_member_id": row["member_id"],
            "refresh_token_hash": token_hash, "created_at": created_at,
            "expires_at": expires_at, "revoked_at": None, "rotation": 0,
        }
        self.next_session_id += 1

    def rotate(self, session_key, token_hash, now):
        self.sessions[session_key]["refresh_token_hash"] = token_hash
        self.sessions[session_key]["rotation"] += 1

    def revoke_user(self, user_id, now):
        revoked = []
        for row in self.sessions.values():
            if row["user_id"] == user_id and row["revoked_at"] is None:
                row["revoked_at"] = now
                revoked.append(dict(row))
        return revoked


@pytest.fixture
def auth():
    settings = Settings(_env_file=None, app_env="test", jwt_secret_key="u" * 40)
    repository = FakeRepository()
    redis = FakeRedis()
    store = AuthStore(redis, "auth-test:")
    return AuthService(repository, store, settings), repository, redis


def sign_in(service, tenant_id=1):
    login = service.login("member@example.test", PASSWORD, "127.0.0.1")
    return service.select_tenant(login.selection_token, tenant_id, "test-client", "127.0.0.1")


def assert_error(code, function, *args, **kwargs):
    with pytest.raises(AuthError) as result:
        function(*args, **kwargs)
    assert result.value.code == code
    return result.value


def test_password_seed_format_and_invalid_hashes():
    assert verify_password(PASSWORD, PASSWORD_HASH)
    assert not verify_password("wrong", PASSWORD_HASH)
    for encoded in ("bad", PASSWORD_HASH.replace("16384", "1073741824"),
                    PASSWORD_HASH.replace("scrypt", "sha256"),
                    "scrypt$16384$8$1$invalid!$invalid!"):
        assert not verify_password(PASSWORD, encoded)
    assert LoginRequest(email=" Member@Example.Test ", password=PASSWORD).email == \
        "member@example.test"


def test_login_has_only_tenant_ticket_and_preserves_bigint_ids(auth):
    service, _, _ = auth
    login = service.login("member@example.test", PASSWORD, "127.0.0.1")
    assert login.user.id == "9007199254740993"
    assert [tenant.id for tenant in login.tenants] == ["1", "2"]
    assert "access_token" not in login.model_dump()
    assert_error("ACCESS_TOKEN_INVALID", decode_access_token, service.settings,
                 login.selection_token)


def test_bad_unknown_and_disabled_user_have_same_login_failure(auth):
    service, repository, _ = auth
    assert_error("LOGIN_FAILED", service.login, "unknown@example.test", PASSWORD, "ip1")
    assert_error("LOGIN_FAILED", service.login, "member@example.test", "wrong", "ip2")
    repository.user["is_active"] = False
    assert_error("LOGIN_FAILED", service.login, "member@example.test", PASSWORD, "ip3")


def test_selection_cannot_cross_tenant_and_is_single_use(auth):
    service, repository, _ = auth
    login = service.login("member@example.test", PASSWORD, "127.0.0.1")
    assert_error("TENANT_FORBIDDEN", service.select_tenant, login.selection_token, 999, None, None)
    result, refresh, ttl = service.select_tenant(login.selection_token, 1, None, None)
    assert result.tenant.id == "1"
    assert ttl == REFRESH_TTL
    assert refresh not in result.model_dump_json()
    assert len(repository.sessions) == 1
    assert_error("SELECTION_TOKEN_INVALID", service.select_tenant,
                 login.selection_token, 2, None, None)


def test_selection_ticket_invalid_after_password_change(auth):
    service, repository, _ = auth
    login = service.login("member@example.test", PASSWORD, "ip")
    repository.user["password_hash"] = hash_password("changed-password")
    assert_error("SELECTION_TOKEN_INVALID", service.select_tenant,
                 login.selection_token, 1, None, None)


def test_refresh_rotates_hash_fixed_expiry_and_keeps_previous_access(auth):
    service, repository, redis = auth
    result, refresh, _ = sign_in(service)
    initial = next(iter(repository.sessions.values())).copy()
    new_result, new_refresh, ttl = service.refresh(refresh)
    current = next(iter(repository.sessions.values()))
    assert new_refresh != refresh
    assert current["rotation"] == 1
    assert current["expires_at"] == initial["expires_at"]
    assert 0 < ttl <= REFRESH_TTL
    assert current["refresh_token_hash"] == token_hash(new_refresh)
    assert all(refresh not in key and new_refresh not in key for key in redis.values)
    assert_error("REFRESH_TOKEN_INVALID", service.refresh, refresh)
    claims = decode_access_token(service.settings, result.access_token)
    assert identity_from_access(repository, claims).tenant_id == 1
    assert new_result.user == result.user


def test_two_simultaneous_refreshes_only_one_succeeds(auth):
    service, repository, _ = auth
    _, refresh, _ = sign_in(service)
    outcomes = []
    barrier = threading.Barrier(2)

    def perform():
        barrier.wait()
        try:
            outcomes.append(service.refresh(refresh))
        except AuthError as error:
            outcomes.append(error.code)

    threads = [threading.Thread(target=perform) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(isinstance(result, tuple) for result in outcomes) == 1
    assert "REFRESH_TOKEN_INVALID" in outcomes
    assert next(iter(repository.sessions.values()))["rotation"] == 1


def test_logout_revokes_all_tenants_devices_and_previous_selection(auth):
    service, repository, _ = auth
    first, first_refresh, _ = sign_in(service, 1)
    second, second_refresh, _ = sign_in(service, 2)
    old_login = service.login("member@example.test", PASSWORD, "ip")
    service.logout(first.access_token, first_refresh)
    for result in (first, second):
        assert_error("SESSION_REVOKED", identity_from_access, repository,
                     decode_access_token(service.settings, result.access_token))
    for refresh in (first_refresh, second_refresh):
        assert_error("REFRESH_TOKEN_INVALID", service.refresh, refresh)
    assert_error("SELECTION_TOKEN_INVALID", service.select_tenant,
                 old_login.selection_token, 1, None, None)
    # 退出完成后重新输入密码的登录可建立新会话。
    _, refresh, _ = sign_in(service)
    service.refresh(refresh)


def test_logout_with_refresh_when_access_expired(auth):
    service, repository, _ = auth
    result, refresh, _ = sign_in(service)
    claims = decode_access_token(service.settings, result.access_token)
    row = repository.rows[1]
    expired = issue_access_token(service.settings, row, claims["sid"],
                                 int(time.time()) - ACCESS_TTL - 1)
    service.logout(expired, refresh)
    assert all(row["revoked_at"] for row in repository.sessions.values())


def test_old_revoked_access_cannot_logout_new_sessions(auth):
    service, repository, _ = auth
    first, refresh, _ = sign_in(service)
    service.logout(first.access_token, refresh)
    second, _, _ = sign_in(service)
    assert_error("SESSION_REVOKED", service.logout, first.access_token, None)
    assert identity_from_access(repository,
                                decode_access_token(service.settings, second.access_token))


@pytest.mark.parametrize("version", ["auth_version", "tenant_auth_version"])
def test_authorization_versions_checked_on_access_and_refresh(auth, version):
    service, repository, _ = auth
    result, refresh, _ = sign_in(service)
    repository.rows[1][version] += 1
    assert_error("AUTHORIZATION_CHANGED", identity_from_access, repository,
                 decode_access_token(service.settings, result.access_token))
    assert_error("AUTHORIZATION_CHANGED", service.refresh, refresh)


def test_user_disabled_immediately_denies_access_refresh(auth):
    service, repository, _ = auth
    result, refresh, _ = sign_in(service)
    repository.user["is_active"] = False
    assert_error("AUTHORIZATION_CHANGED", identity_from_access, repository,
                 decode_access_token(service.settings, result.access_token))
    assert_error("AUTHORIZATION_CHANGED", service.refresh, refresh)
    # 停用账号仍可退出，以清除全部会话。
    service.logout(result.access_token, refresh)


def test_redis_missing_or_down_fails_closed_without_secret(auth):
    service, _, redis = auth
    _, refresh, _ = sign_in(service)
    redis.values.pop(service.store.key("refresh", token_hash(refresh)))
    assert_error("REFRESH_TOKEN_INVALID", service.refresh, refresh)
    redis.dead = True
    error = assert_error("AUTH_STORE_UNAVAILABLE", service.refresh, refresh)
    assert error.status_code == 503
    assert "secret" not in str(error)
    assert_error("AUTH_STORE_UNAVAILABLE", service.login, "member@example.test", PASSWORD, "ip")


def test_logout_remains_effective_when_redis_cleanup_fails(auth):
    service, repository, redis = auth
    result, refresh, _ = sign_in(service)
    redis.dead = True
    service.logout(result.access_token, refresh)
    assert_error("SESSION_REVOKED", identity_from_access, repository,
                 decode_access_token(service.settings, result.access_token))


def test_failed_db_commit_discards_new_refresh(auth):
    service, repository, redis = auth
    _, refresh, _ = sign_in(service)
    repository.connection.fail_commit = True
    with pytest.raises(psycopg.Error):
        service.refresh(refresh)
    assert not any(key.startswith(service.store.prefix + "refresh:") for key in redis.values)
    assert_error("REFRESH_TOKEN_INVALID", service.refresh, refresh)


def test_login_rate_limit_and_no_raw_identifiers_in_redis(auth):
    service, _, redis = auth
    for _ in range(10):
        assert_error("LOGIN_FAILED", service.login, "member@example.test", "wrong", "127.0.0.1")
    error = assert_error("LOGIN_RATE_LIMITED", service.login,
                         "member@example.test", PASSWORD, "127.0.0.1")
    assert error.status_code == 429
    assert all("member@example.test" not in key and "127.0.0.1" not in key
               for key in redis.values)


def test_jwt_strict_signature_audience_type_claims_and_expired_code(auth):
    service, repository, _ = auth
    result, _, _ = sign_in(service)
    settings = service.settings
    payload = jwt.decode(result.access_token, options={"verify_signature": False})
    for change in ({"aud": "another-app"}, {"iss": "another-issuer"}, {"type": "refresh"},
                   {"av": True}, {"tid": "0"}, {"sid": "invalid"}):
        bad = jwt.encode({**payload, **change}, settings.jwt_secret_key.get_secret_value(),
                         algorithm="HS256")
        assert_error("ACCESS_TOKEN_INVALID", decode_access_token, settings, bad)
    forged = jwt.encode(payload, "different-signing-key-with-32-bytes", algorithm="HS256")
    assert_error("ACCESS_TOKEN_INVALID", decode_access_token, settings, forged)
    expired = issue_access_token(settings, repository.rows[1], uuid4(),
                                 int(time.time()) - ACCESS_TTL - 1)
    claims = decode_access_token(settings, expired)
    assert_error("SESSION_REVOKED", identity_from_access, repository, claims)
    valid = decode_access_token(settings, result.access_token)
    valid["exp"] = int(time.time()) - 1
    assert_error("ACCESS_TOKEN_EXPIRED", identity_from_access, repository, valid)


def test_session_absolute_expiry_denies_access_even_fresh_jwt(auth):
    service, repository, _ = auth
    result, refresh, _ = sign_in(service)
    next(iter(repository.sessions.values()))["expires_at"] = (
        datetime.now(UTC) - timedelta(seconds=1)
    )
    assert_error("SESSION_REVOKED", identity_from_access, repository,
                 decode_access_token(service.settings, result.access_token))
    assert_error("SESSION_REVOKED", service.refresh, refresh)


def request(headers=()):
    return Request({"type": "http", "headers": list(headers), "method": "GET",
                    "path": "/api/auth/me", "client": ("127.0.0.1", 123)})


def test_missing_access_rejected_without_infrastructure_and_empty_config(monkeypatch):
    def unexpected_connection(settings):
        raise AssertionError("DB must not be opened")

    monkeypatch.setattr(auth_router, "database_connection", unexpected_connection)
    assert_error("AUTH_REQUIRED", auth_router.require_identity, request(),
                 Settings(_env_file=None, jwt_secret_key=""))


def test_cookie_operations_require_csrf_header_and_allowed_origin(auth):
    settings = auth[0].settings
    assert_error("CSRF_VALIDATION_FAILED", auth_router.require_cookie_request, request(), settings)
    assert_error("ORIGIN_FORBIDDEN", auth_router.require_cookie_request,
                 request([(b"origin", b"https://evil.test"),
                          (b"x-requested-with", b"XMLHttpRequest")]), settings)
    auth_router.require_cookie_request(
        request([(b"x-requested-with", b"XMLHttpRequest")]), settings,
    )


@pytest.mark.parametrize("headers,code", [
    ({}, "CSRF_VALIDATION_FAILED"),
    ({"X-Requested-With": "XMLHttpRequest", "Origin": "https://evil.test"},
     "ORIGIN_FORBIDDEN"),
])
def test_http_csrf_fails_before_database_connection(auth, monkeypatch, headers, code):
    def unexpected_connection(settings):
        raise AssertionError("CSRF failure must not open DB")

    monkeypatch.setattr(auth_router, "database_connection", unexpected_connection)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: auth[0].settings
    with TestClient(app) as client:
        response = client.post("/api/auth/login", headers=headers, json={
            "email": "member@example.test", "password": PASSWORD,
        })
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == code


def test_http_missing_signing_key_fails_before_database_connection(monkeypatch):
    def unexpected_connection(settings):
        raise AssertionError("Missing signing key must not open DB")

    monkeypatch.setattr(auth_router, "database_connection", unexpected_connection)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, jwt_secret_key="")
    with TestClient(app) as client:
        response = client.post("/api/auth/login", headers={
            "X-Requested-With": "XMLHttpRequest",
        }, json={"email": "member@example.test", "password": PASSWORD})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "AUTH_CONFIG_INVALID"


def test_http_database_failure_is_sanitized(auth, monkeypatch):
    @contextmanager
    def unavailable(settings):
        raise psycopg.OperationalError("secret db connection password")
        yield

    monkeypatch.setattr(auth_router, "database_connection", unavailable)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: auth[0].settings
    with TestClient(app) as client:
        response = client.post("/api/auth/login", headers={
            "X-Requested-With": "XMLHttpRequest",
        }, json={"email": "member@example.test", "password": PASSWORD})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DATABASE_UNAVAILABLE"
    assert "secret" not in response.text


@pytest.fixture
def api(auth, monkeypatch):
    service, repository, _ = auth
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: service.settings
    app.dependency_overrides[auth_router.get_auth_service] = lambda: service

    @contextmanager
    def connection(settings):
        yield repository.connection

    monkeypatch.setattr(auth_router, "database_connection", connection)
    monkeypatch.setattr(auth_router, "AuthRepository", lambda conn, schema: repository)
    with TestClient(app) as client:
        yield client, service, repository


def test_http_login_selection_cookie_refresh_me_and_logout(api):
    client, _, repository = api
    login = client.post("/api/auth/login", json={
        "email": " MEMBER@EXAMPLE.TEST ", "password": PASSWORD,
    })
    assert login.status_code == 200
    assert login.headers["cache-control"] == "no-store"
    assert "set-cookie" not in login.headers
    response = client.post("/api/auth/select-tenant", json={
        "selection_token": login.json()["selection_token"], "tenant_id": "2",
    })
    assert response.status_code == 200
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Path=/api/auth" in cookie and "SameSite=lax" in cookie
    assert "Max-Age=604800" in cookie
    payload = response.json()
    assert payload["expires_in"] == ACCESS_TTL and payload["tenant"]["id"] == "2"
    assert "refresh_token" not in payload
    header = {"Authorization": "Bearer " + payload["access_token"]}
    assert client.get("/api/auth/me", headers=header).json()["tenant"]["id"] == "2"
    refreshed = client.post("/api/auth/refresh")
    assert refreshed.status_code == 200
    assert refreshed.headers["set-cookie"] != cookie
    assert client.post("/api/auth/logout", headers=header).status_code == 204
    assert client.get("/api/auth/me", headers=header).json()["detail"]["code"] == "SESSION_REVOKED"
    assert all(session["revoked_at"] for session in repository.sessions.values())


@pytest.mark.parametrize("path,body", [
    ("/api/auth/login", {"email": "bad", "password": "sensitive" * 200}),
    ("/api/auth/select-tenant", {"selection_token": "sensitive", "tenant_id": -1}),
])
def test_http_validation_does_not_echo_password_or_tokens(api, path, body):
    response = api[0].post(path, json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_AUTH_REQUEST"
    assert "sensitive" not in response.text


def test_http_unauthenticated_me_and_missing_refresh_are_deterministic(api):
    client = api[0]
    assert client.get("/api/auth/me").json()["detail"]["code"] == "AUTH_REQUIRED"
    assert client.post("/api/auth/refresh").json()["detail"]["code"] == "REFRESH_TOKEN_INVALID"


def test_empty_signing_key_is_not_replaced_by_fallback(auth):
    service, _, _ = auth
    service.settings.jwt_secret_key = Settings(_env_file=None, jwt_secret_key="").jwt_secret_key
    assert_error("AUTH_CONFIG_INVALID", service.login, "member@example.test", PASSWORD, "ip")


def test_corrupt_redis_refresh_payload_denied_without_server_error(auth):
    service, _, redis = auth
    _, refresh, _ = sign_in(service)
    redis.values[service.store.key("refresh", token_hash(refresh))] = json.dumps({"user_id": True})
    assert_error("REFRESH_TOKEN_INVALID", service.refresh, refresh)


def test_cookie_secure_flag_is_honored():
    response = SimpleNamespace(headers={}, set_cookie=lambda **kwargs: captured.update(kwargs))
    captured = {}
    auth_router.set_refresh_cookie(
        response, Settings(_env_file=None, refresh_cookie_secure=True), "opaque", 100,
    )
    assert captured["secure"] and captured["httponly"]


def test_access_session_cannot_be_rebound_to_different_member(auth):
    service, repository, _ = auth
    result, _, _ = sign_in(service)
    claims = decode_access_token(service.settings, result.access_token)
    claims["mid"] += 1
    assert_error("ACCESS_TOKEN_INVALID", identity_from_access, repository, claims)
