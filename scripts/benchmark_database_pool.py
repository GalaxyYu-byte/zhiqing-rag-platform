"""同一进程比较短连接和连接池，只查 SELECT 1 与事务设置，不读取业务数据。"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from statistics import median
from threading import Barrier

import psycopg
from psycopg.rows import dict_row

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.connection import (
    close_database_pool,
    database_connection,
)


@contextmanager
def short_connection(settings):
    with psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_username,
        password=settings.db_password.get_secret_value(),
        connect_timeout=max(1, int(settings.service_timeout_seconds)),
        options="-c statement_timeout=15000 -c lock_timeout=5000",
        row_factory=dict_row,
    ) as connection:
        yield connection


def measure(factory, settings):
    started = time.perf_counter()
    with factory(settings) as connection:
        acquired = time.perf_counter()
        assert connection.execute("SELECT 1 AS value").fetchone()["value"] == 1
        queried = time.perf_counter()
    finished = time.perf_counter()
    return {
        "acquire_ms": round((acquired - started) * 1000, 1),
        "query_ms": round((queried - acquired) * 1000, 1),
        "cleanup_ms": round((finished - queried) * 1000, 1),
        "total_ms": round((finished - started) * 1000, 1),
    }


def verify_transactions(settings):
    with database_connection(settings) as first:
        first.execute("SET TRANSACTION READ ONLY")
        state = first.execute(
            "SELECT current_setting('transaction_isolation') AS isolation, "
            "current_setting('lock_timeout') AS lock_timeout"
        ).fetchone()
        assert state["isolation"] == "read committed"
        first.execute("SET LOCAL lock_timeout = '1234ms'")
    with database_connection(settings) as reused:
        assert reused is first
        assert (
            reused.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == state["lock_timeout"]
        )
    try:
        with database_connection(settings) as failed:
            failed.execute("SET LOCAL lock_timeout = '2345ms'")
            raise ValueError("exercise rollback")
    except ValueError:
        pass
    with database_connection(settings) as reused:
        assert reused is first
        assert (
            reused.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == state["lock_timeout"]
        )
    # 仅关闭本脚本已归还的空闲连接，模拟服务端断连后的恢复。
    first.close()
    with database_connection(settings) as replacement:
        assert replacement is not first
        assert replacement.execute("SELECT 1 AS value").fetchone()["value"] == 1
    checks = {
        "reuse": True,
        "commit_reset": True,
        "rollback_reset": True,
        "disconnect_recovery": True,
    }
    if settings.db_pool_size + settings.db_max_overflow >= 2:
        barrier = Barrier(2)

        def borrow():
            with database_connection(settings) as connection:
                connection.execute("SELECT 1").fetchone()
                barrier.wait(timeout=15)
                return connection

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(borrow) for _ in range(2)]
            first, second = [future.result(timeout=20) for future in futures]
        assert first is not second
        checks["concurrent_exclusive"] = True
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, choices=range(1, 21), default=5)
    args = parser.parse_args()
    settings = get_settings()
    close_database_pool(settings)
    try:
        cold = measure(database_connection, settings)
        samples = {"short": [], "pooled_warm": []}
        for _ in range(args.samples):
            for name, factory in (
                ("short", short_connection),
                ("pooled_warm", database_connection),
            ):
                samples[name].append(measure(factory, settings))
        result = {
            "cold_pool": cold,
            "samples": samples,
            "median_ms": {
                name: {key: round(median(row[key] for row in rows), 1) for key in rows[0]}
                for name, rows in samples.items()
            },
            "checks": verify_transactions(settings),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}))
        return 1
    finally:
        close_database_pool(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
