"""按事务借用进程内连接池；启动时不连接数据库，不缓存权限。"""

import atexit
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

import psycopg
from psycopg.rows import dict_row
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.pool import QueuePool

from zhiqing_rag.core.config import Settings


@dataclass(frozen=True)
class _PoolConfig:
    host: str
    port: int
    database: str
    username: str
    password: str = field(repr=False)
    schema: str
    connect_timeout: int
    size: int
    overflow: int
    timeout: int

    @classmethod
    def from_settings(cls, settings: Settings):
        return cls(
            settings.db_host,
            settings.db_port,
            settings.db_name,
            settings.db_username,
            settings.db_password.get_secret_value(),
            settings.db_schema,
            max(1, int(settings.service_timeout_seconds)),
            settings.db_pool_size,
            settings.db_max_overflow,
            settings.db_pool_timeout,
        )

    def connect(self):
        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.database,
            user=self.username,
            password=self.password,
            connect_timeout=self.connect_timeout,
            options="-c statement_timeout=15000 -c lock_timeout=5000",
            row_factory=dict_row,
        )


_pools: dict[_PoolConfig, QueuePool] = {}
_pool_lock = Lock()


def _get_pool(settings: Settings) -> QueuePool:
    config = _PoolConfig.from_settings(settings)
    with _pool_lock:
        pool = _pools.get(config)
        if pool is None:
            pool = QueuePool(
                config.connect,
                pool_size=config.size,
                max_overflow=config.overflow,
                timeout=config.timeout,
                use_lifo=True,
                pre_ping=True,
                reset_on_return="rollback",
                dialect=PGDialect_psycopg(dbapi=psycopg),
            )
            _pools[config] = pool
        return pool


@contextmanager
def database_connection(settings: Settings):
    try:
        borrowed = _get_pool(settings).connect()
    except PoolTimeoutError:
        # 保持现有调用方的 psycopg.Error -> 脱敏 503 处理。
        raise psycopg.OperationalError("数据库连接池暂时繁忙") from None
    connection = borrowed.driver_connection
    try:
        yield connection
    except BaseException:
        try:
            connection.rollback()
        except BaseException as rollback_error:
            borrowed.invalidate(rollback_error)
        raise
    else:
        try:
            connection.commit()
        except BaseException as commit_error:
            # 提交结果可能不确定，废弃连接并上抛；不自动重试写操作。
            borrowed.invalidate(commit_error)
            raise
    finally:
        borrowed.close()


def close_database_pool(settings: Settings) -> None:
    """应用停止、请求已结束后，关闭此配置的空闲连接。"""
    with _pool_lock:
        pool = _pools.pop(_PoolConfig.from_settings(settings), None)
    if pool is not None:
        pool.dispose()


def close_database_pools() -> None:
    """脚本及 worker 进程正常退出时释放空闲连接。"""
    with _pool_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        pool.dispose()


def _after_fork() -> None:
    # 子进程不能复用父进程的连接或锁，也不能关闭父进程的网络连接。
    global _pool_lock, _pools
    _pools = {}
    _pool_lock = Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(close_database_pools)
