"""显式启用的真实 PostgreSQL / Redis 认证实测，仅创建并清理唯一测试命名空间。"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from redis import Redis
from sqlalchemy import MetaData, create_engine
from sqlalchemy.schema import CreateSchema, DropSchema

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.identity import AuthSession, Department, Tenant, TenantMember, User
from zhiqing_rag.modules.identity.auth_repository import AuthRepository
from zhiqing_rag.modules.identity.passwords import hash_password

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_AUTH_INTEGRATION") != "1",
    reason="需要显式启用真实 PostgreSQL / Redis 隔离认证实测",
)

PASSWORD = "Auth-test-passphrase-2026"


@pytest.fixture
def auth_environment():
    suffix = uuid4().hex
    schema = f"auth_test_{suffix}"
    original = get_settings()
    prefix = f"{original.redis_key_prefix}auth-test:{suffix}:"
    settings = original.model_copy(update={
        "app_env": "test",
        "db_schema": schema,
        "redis_key_prefix": prefix,
        "jwt_secret_key": SecretStr(uuid4().hex + uuid4().hex),
        "refresh_cookie_secure": False,
    })
    engine = create_engine(settings.database_url)
    cache = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password.get_secret_value() or None,
        socket_connect_timeout=settings.service_timeout_seconds,
        socket_timeout=settings.service_timeout_seconds,
        decode_responses=True,
    )
    schema_created = False
    cache_connected = False
    try:
        assert cache.ping(), "Redis 必须可用"
        cache_connected = True
        metadata = MetaData(schema=schema)
        tables = {}
        for model in (User, Tenant, Department, TenantMember, AuthSession):
            tables[model.__tablename__] = model.__table__.to_metadata(
                metadata, schema=schema, referred_schema_fn=lambda *_args: schema,
            )
        with engine.begin() as connection:
            connection.execute(CreateSchema(schema))
            metadata.create_all(connection)
            user_id = connection.execute(
                tables["user_account"].insert().values(
                    email=f"user-{suffix}@auth.test", password_hash=hash_password(PASSWORD),
                    display_name="认证实测用户",
                ).returning(tables["user_account"].c.id),
            ).scalar_one()
            tenant_ids = []
            department_ids = []
            member_ids = []
            for index in range(2):
                tenant_id = connection.execute(
                    tables["tenant"].insert().values(
                        code=f"TEST_{index}", name=f"认证测试租户 {index}",
                    ).returning(tables["tenant"].c.id),
                ).scalar_one()
                department_id = connection.execute(
                    tables["department"].insert().values(
                        tenant_id=tenant_id, code="TEST_DEPT", name="认证测试部门",
                    ).returning(tables["department"].c.id),
                ).scalar_one()
                member_id = connection.execute(
                    tables["tenant_member"].insert().values(
                        tenant_id=tenant_id, user_id=user_id, department_id=department_id,
                        clearance=index, role="MEMBER",
                    ).returning(tables["tenant_member"].c.id),
                ).scalar_one()
                tenant_ids.append(tenant_id)
                department_ids.append(department_id)
                member_ids.append(member_id)
        schema_created = True
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings
        yield SimpleNamespace(
            settings=settings, app=app, redis=cache, prefix=prefix, user_id=user_id,
            email=f"user-{suffix}@auth.test", tenant_ids=tenant_ids,
            department_ids=department_ids, member_ids=member_ids,
        )
    finally:
        # 所有名称均由本fixture生成；不访问项目业务schema，不执行FLUSHDB。
        assert schema == f"auth_test_{suffix}" and len(suffix) == 32
        try:
            if schema_created:
                with engine.begin() as connection:
                    connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        finally:
            try:
                if cache_connected:
                    keys = [
                        key for key in cache.scan_iter(match=prefix + "*")
                        if key.startswith(prefix)
                    ]
                    if keys:
                        cache.delete(*keys)
            finally:
                cache.close()
                engine.dispose()


def client_for(environment, stack):
    client = stack.enter_context(TestClient(environment.app))
    client.headers.update({
        "Origin": environment.settings.cors_origins[0],
        "X-Requested-With": "XMLHttpRequest",
    })
    return client


def login_and_select(environment, client, tenant_index=0):
    login = client.post(
        "/api/auth/login", json={"email": environment.email, "password": PASSWORD},
    )
    assert login.status_code == 200, login.text
    assert {tenant["id"] for tenant in login.json()["tenants"]} == {
        str(value) for value in environment.tenant_ids
    }
    selected = client.post(
        "/api/auth/select-tenant",
        json={
            "selection_token": login.json()["selection_token"],
            "tenant_id": str(environment.tenant_ids[tenant_index]),
        },
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["expires_in"] == 900
    assert selected.json()["tenant"]["id"] == str(environment.tenant_ids[tenant_index])
    assert "refresh_token" not in selected.json()
    return selected.json()["access_token"]


def me(client, token):
    return client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})


def query(environment, statement, parameters=(), *, all_rows=False):
    with database_connection(environment.settings) as connection:
        result = AuthRepository(connection, environment.settings.db_schema).query(
            statement, parameters,
        )
        if result.description is None:
            return None
        return result.fetchall() if all_rows else result.fetchone()


def test_real_login_rotation_fixed_expiry_and_all_device_logout(auth_environment):
    environment = auth_environment
    with ExitStack() as stack:
        device_a = client_for(environment, stack)
        device_b = client_for(environment, stack)
        access_a = login_and_select(environment, device_a)
        access_b = login_and_select(environment, device_b, 1)
        assert me(device_a, access_a).json()["tenant"]["id"] == str(environment.tenant_ids[0])
        assert me(device_b, access_b).json()["tenant"]["id"] == str(environment.tenant_ids[1])
        before = query(environment, "SELECT * FROM {schema}.auth_session ORDER BY id",
                       all_rows=True)
        assert len(before) == 2 and before[0]["session_key"] != before[1]["session_key"]
        assert before[0]["expires_at"] - before[0]["created_at"] == timedelta(days=7)
        old_refresh = device_a.cookies.get(environment.settings.refresh_cookie_name)
        response = device_a.post("/api/auth/refresh")
        assert response.status_code == 200, response.text
        assert device_a.cookies.get(environment.settings.refresh_cookie_name) != old_refresh
        after = query(environment, "SELECT * FROM {schema}.auth_session ORDER BY id",
                      all_rows=True)
        assert after[0]["expires_at"] == before[0]["expires_at"]
        assert after[0]["rotation"] == 1
        assert after[0]["refresh_token_hash"] != before[0]["refresh_token_hash"]
        replayer = client_for(environment, stack)
        replay = replayer.post(
            "/api/auth/refresh",
            headers={"Cookie": f"{environment.settings.refresh_cookie_name}={old_refresh}"},
        )
        assert replay.status_code == 401, replay.text
        logout = device_a.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {response.json()['access_token']}"},
        )
        assert logout.status_code == 204, logout.text
        assert me(device_a, access_a).status_code == 401
        assert me(device_b, access_b).status_code == 401
        assert device_b.post("/api/auth/refresh").status_code == 401
        assert not device_a.cookies.get(environment.settings.refresh_cookie_name)
        sessions = query(environment, "SELECT revoked_at FROM {schema}.auth_session",
                         all_rows=True)
        assert all(session["revoked_at"] is not None for session in sessions)


@pytest.mark.parametrize("change", ["user", "tenant", "member", "department",
                                    "tenant_version", "member_version"])
def test_real_access_checks_current_status_and_versions(auth_environment, change):
    environment = auth_environment
    with ExitStack() as stack:
        client = client_for(environment, stack)
        access = login_and_select(environment, client)
        assert me(client, access).status_code == 200
        changes = {
            "user": ("user_account", "is_active = false", environment.user_id),
            "tenant": ("tenant", "status = 'DISABLED'", environment.tenant_ids[0]),
            "member": ("tenant_member", "status = 'DISABLED'", environment.member_ids[0]),
            "department": ("department", "is_active = false", environment.department_ids[0]),
            "tenant_version": ("tenant", "auth_version = auth_version + 1",
                               environment.tenant_ids[0]),
            "member_version": ("tenant_member", "auth_version = auth_version + 1",
                               environment.member_ids[0]),
        }
        table, assignment, identity_id = changes[change]
        # 表名及赋值仅来自上面的固定测试映射，值始终使用参数绑定。
        query(environment, f"UPDATE {{schema}}.{table} SET {assignment} WHERE id = %s",
              (identity_id,))
        assert me(client, access).status_code == 401
        refresh = client.post("/api/auth/refresh")
        assert refresh.status_code == 401, refresh.text


def test_real_refresh_fails_after_redis_record_is_lost(auth_environment):
    environment = auth_environment
    with ExitStack() as stack:
        client = client_for(environment, stack)
        login_and_select(environment, client)
        keys = list(environment.redis.scan_iter(match=environment.prefix + "*"))
        assert keys and all(key.startswith(environment.prefix) for key in keys)
        environment.redis.delete(*keys)
        response = client.post("/api/auth/refresh")
        assert response.status_code == 401, response.text


def test_real_repeated_logout_in_one_second_invalidates_pending_selection(
    auth_environment, monkeypatch
):
    import zhiqing_rag.modules.identity.auth_service as service_module

    environment = auth_environment
    fixed_time = datetime.fromtimestamp(int(time.time()), UTC)
    monkeypatch.setattr(service_module, "current_time", lambda: fixed_time)
    with ExitStack() as stack:
        device = client_for(environment, stack)
        first_access = login_and_select(environment, device)
        first_logout = device.post(
            "/api/auth/logout", headers={"Authorization": f"Bearer {first_access}"},
        )
        assert first_logout.status_code == 204
        second_access = login_and_select(environment, device)
        waiting_device = client_for(environment, stack)
        pending_login = waiting_device.post(
            "/api/auth/login", json={"email": environment.email, "password": PASSWORD},
        )
        assert pending_login.status_code == 200
        second_logout = device.post(
            "/api/auth/logout", headers={"Authorization": f"Bearer {second_access}"},
        )
        assert second_logout.status_code == 204
        pending_selection = waiting_device.post(
            "/api/auth/select-tenant",
            json={
                "selection_token": pending_login.json()["selection_token"],
                "tenant_id": str(environment.tenant_ids[0]),
            },
        )
        assert pending_selection.status_code == 401, pending_selection.text
        sessions = query(environment, "SELECT revoked_at FROM {schema}.auth_session",
                         all_rows=True)
        assert len(sessions) == 2 and all(row["revoked_at"] is not None for row in sessions)


def test_real_concurrent_refresh_accepts_one_rotation(auth_environment):
    environment = auth_environment
    with ExitStack() as stack:
        signed_in = client_for(environment, stack)
        login_and_select(environment, signed_in)
        old_refresh = signed_in.cookies.get(environment.settings.refresh_cookie_name)
        clients = [client_for(environment, stack), client_for(environment, stack)]
        ready = Barrier(2)

        def refresh_with_old_cookie(client):
            ready.wait(timeout=10)
            return client.post(
                "/api/auth/refresh",
                headers={
                    "Cookie": f"{environment.settings.refresh_cookie_name}={old_refresh}",
                },
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(refresh_with_old_cookie, clients))
        assert sorted(response.status_code for response in responses) == [200, 401]
        loser = next(response for response in responses if response.status_code == 401)
        assert loser.json()["detail"]["code"] == "REFRESH_TOKEN_INVALID"
        winner_index = next(index for index, response in enumerate(responses)
                            if response.status_code == 200)
        winner = clients[winner_index]
        assert me(winner, responses[winner_index].json()["access_token"]).status_code == 200
        another_refresh = winner.post("/api/auth/refresh")
        assert another_refresh.status_code == 200, another_refresh.text
        session = query(environment, "SELECT rotation FROM {schema}.auth_session")
        assert session["rotation"] == 2
