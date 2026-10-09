"""连接复用、独占借用、故障恢复和事务边界；不连接业务数据库。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database import connection as database


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, sql):
        assert sql == ";"  # psycopg 方言用空语句探测连接。
        assert self.connection.autocommit
        if self.connection.closed:
            raise psycopg.OperationalError("connection closed")
        self.connection.pings += 1

    def close(self):
        pass


class FakeConnection:
    def __init__(self):
        self.autocommit = False
        self.closed = False
        self.broken = False
        self.pings = 0
        self.commits = 0
        self.rollbacks = 0
        self.pending = []
        self.persisted = []
        self.fail_commit = False
        self.fail_rollback = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        if self.fail_commit:
            raise psycopg.OperationalError("commit failed")
        self.commits += 1
        self.persisted.extend(self.pending)
        self.pending.clear()

    def rollback(self):
        if self.fail_rollback:
            raise psycopg.OperationalError("rollback failed")
        self.rollbacks += 1
        self.pending.clear()

    def close(self):
        self.closed = True


@pytest.fixture
def setup_pool(monkeypatch):
    database.close_database_pools()
    settings = Settings(_env_file=None, db_pool_size=2, db_max_overflow=0, db_pool_timeout=1)
    connections = []
    arguments = []

    def connect(**kwargs):
        arguments.append(kwargs)
        connection = FakeConnection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(database.psycopg, "connect", connect)
    yield settings, connections, arguments
    database.close_database_pools()


def test_pool_is_lazy_and_reuses_committed_connection(setup_pool):
    settings, connections, arguments = setup_pool
    database._get_pool(settings)
    assert connections == []
    with database.database_connection(settings) as first:
        first.pending.append("first transaction")
    with database.database_connection(settings) as second:
        assert second is first
        assert second.pending == []
        assert second.persisted == ["first transaction"]
        assert not second.autocommit
    assert len(connections) == 1
    assert first.commits == 2
    assert first.pings == 1
    assert not first.closed
    assert arguments[0]["row_factory"] is dict_row
    assert arguments[0]["options"] == "-c statement_timeout=15000 -c lock_timeout=5000"


@pytest.mark.parametrize("error", [ValueError("query failed"), KeyboardInterrupt()])
def test_failed_transaction_rolls_back_before_reuse(setup_pool, error):
    settings, connections, _ = setup_pool
    with pytest.raises(type(error)) as raised:
        with database.database_connection(settings) as connection:
            connection.pending.append("must not commit")
            raise error
    assert raised.value is error
    with database.database_connection(settings) as reused:
        assert reused is connection
        assert reused.pending == []
        assert reused.persisted == []
    assert len(connections) == 1


def test_broken_idle_connection_is_replaced(setup_pool):
    settings, connections, _ = setup_pool
    with database.database_connection(settings) as first:
        pass
    first.closed = True
    first.broken = True
    with database.database_connection(settings) as replacement:
        assert replacement is not first
        assert not replacement.autocommit
    assert len(connections) == 2


def test_commit_failure_discards_connection_without_retry(setup_pool):
    settings, connections, _ = setup_pool
    with pytest.raises(psycopg.OperationalError, match="commit failed"):
        with database.database_connection(settings) as first:
            first.pending.append("ambiguous write")
            first.fail_commit = True
    assert first.closed
    assert len(connections) == 1
    with database.database_connection(settings) as replacement:
        assert replacement is not first


def test_rollback_failure_preserves_original_error_and_discards_connection(setup_pool):
    settings, _, _ = setup_pool
    original = ValueError("original error")
    with pytest.raises(ValueError) as raised:
        with database.database_connection(settings) as first:
            first.fail_rollback = True
            raise original
    assert raised.value is original
    assert first.closed
    with database.database_connection(settings) as replacement:
        assert replacement is not first


def test_concurrent_borrowers_use_exclusive_connections(setup_pool):
    settings, connections, _ = setup_pool
    barrier = Barrier(2)

    def borrow():
        with database.database_connection(settings) as connection:
            barrier.wait(timeout=5)
            return connection

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(borrow) for _ in range(2)]
        first, second = [future.result(timeout=5) for future in futures]
    assert first is not second
    assert len(connections) == 2
    assert len(database._pools) == 1


def test_pool_limit_timeout_remains_psycopg_error(setup_pool):
    settings, connections, _ = setup_pool
    settings = settings.model_copy(update={"db_pool_size": 1, "db_max_overflow": 1})
    with database.database_connection(settings):
        with database.database_connection(settings):
            with pytest.raises(psycopg.OperationalError, match="连接池暂时繁忙") as raised:
                with database.database_connection(settings):
                    pytest.fail("exceeded pool limit")
            assert raised.value.__cause__ is None
    assert len(connections) == 2
    assert sum(not connection.closed for connection in connections) == 1


def test_dispose_and_changed_config_do_not_reuse_old_connections(setup_pool):
    settings, _, _ = setup_pool
    with database.database_connection(settings) as first:
        pass
    other_settings = Settings(
        **{**settings.model_dump(), "db_password": "different-secret"}, _env_file=None
    )
    with database.database_connection(other_settings) as other:
        assert other is not first
    assert "different-secret" not in repr(database._PoolConfig.from_settings(other_settings))
    database.close_database_pool(settings)
    assert first.closed
    assert not other.closed
    with database.database_connection(settings) as replacement:
        assert replacement is not first


def test_connect_failure_can_recover_on_next_borrow(setup_pool, monkeypatch):
    settings, _, _ = setup_pool
    original_connect = database.psycopg.connect

    def unavailable(**kwargs):
        raise psycopg.OperationalError("unavailable")

    monkeypatch.setattr(database.psycopg, "connect", unavailable)
    with pytest.raises(psycopg.OperationalError):
        with database.database_connection(settings):
            pytest.fail("database unavailable")
    monkeypatch.setattr(database.psycopg, "connect", original_connect)
    with database.database_connection(settings) as connection:
        assert not connection.closed


def test_application_start_is_lazy_and_shutdown_closes_idle_connections(setup_pool, monkeypatch):
    from zhiqing_rag import main

    settings, connections, _ = setup_pool
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    app = main.create_app()
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert connections == []
        with database.database_connection(settings) as connection:
            pass
        assert not connection.closed
    assert connection.closed
