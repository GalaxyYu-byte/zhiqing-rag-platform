from typing import Any

import pytest

from zq_rag_app.workers.index_worker import startup_worker


class _OldPool:
    def __init__(self) -> None:
        self.connection_kwargs = {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "encoding": "utf8",
        }
        self.max_connections = 8
        self.disconnected = False

    async def disconnect(self, *, inuse_connections: bool = True) -> None:
        assert inuse_connections is True
        self.disconnected = True


class _FakeArqRedis:
    def __init__(self) -> None:
        self.connection_pool: Any = _OldPool()


@pytest.mark.asyncio
async def test_worker_replaces_arq_pool_with_recoverable_connections() -> None:
    redis = _FakeArqRedis()
    old_pool = redis.connection_pool

    await startup_worker({"redis": redis})

    assert old_pool.disconnected is True
    assert redis.connection_pool is not old_pool
    assert redis.connection_pool.max_connections == 8
    assert redis.connection_pool.connection_kwargs["health_check_interval"] == 30
    assert TypeError in redis.connection_pool.connection_kwargs["retry_on_error"]
    await redis.connection_pool.disconnect(inuse_connections=True)
