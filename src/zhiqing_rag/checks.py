"""环境诊断：只读检查中间件连接，不建表、不写缓存、不调用模型生成。"""

import asyncio
import json
import sys
from urllib.parse import urlsplit

import psycopg
import urllib3
from minio import Minio
from neo4j import READ_ACCESS, AsyncGraphDatabase
from redis.asyncio import Redis

from zhiqing_rag.core.config import Settings, get_settings


def _check_postgres(settings: Settings) -> dict:
    # 在线程中执行短只读查询，兼容 Windows 默认事件循环。
    with psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_username,
        password=settings.db_password.get_secret_value(),
        connect_timeout=max(1, int(settings.service_timeout_seconds)),
        options="-c default_transaction_read_only=on -c statement_timeout=5000",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            vector = cursor.fetchone()
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = %s)",
                (settings.db_schema,),
            )
            schema_exists = cursor.fetchone()[0]
    return {
        "status": "ok" if vector else "error",
        "pgvector": vector[0] if vector else "missing",
        "project_schema": "present" if schema_exists else "pending_database_design",
    }


async def check_postgres(settings: Settings) -> dict:
    return await asyncio.to_thread(_check_postgres, settings)


async def check_redis(settings: Settings) -> dict:
    client = Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password.get_secret_value() or None,
        socket_connect_timeout=settings.service_timeout_seconds,
        socket_timeout=settings.service_timeout_seconds,
    )
    try:
        return {"status": "ok" if await client.ping() else "error"}
    finally:
        await client.aclose()


def _check_minio(settings: Settings) -> dict:
    endpoint = urlsplit(settings.minio_endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
        raise ValueError("MINIO_ENDPOINT must include http:// or https://")
    pool = urllib3.PoolManager(
        timeout=urllib3.Timeout(total=settings.service_timeout_seconds), retries=False
    )
    try:
        client = Minio(
            endpoint.netloc,
            access_key=settings.minio_access_key.get_secret_value(),
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=endpoint.scheme == "https",
            http_client=pool,
        )
        return {"status": "ok" if client.bucket_exists(settings.minio_bucket) else "error"}
    finally:
        pool.clear()


async def check_minio(settings: Settings) -> dict:
    return await asyncio.to_thread(_check_minio, settings)


async def check_neo4j(settings: Settings) -> dict:
    if not settings.neo4j_enabled:
        return {"status": "disabled"}
    async with AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        connection_timeout=settings.service_timeout_seconds,
        connection_acquisition_timeout=settings.service_timeout_seconds,
        max_transaction_retry_time=0,
    ) as driver:
        async with driver.session(
            database=settings.neo4j_database, default_access_mode=READ_ACCESS
        ) as session:
            result = await session.run("RETURN 1 AS ready")
            record = await result.single()
            return {"status": "ok" if record and record["ready"] == 1 else "error"}


async def check_services(settings: Settings) -> dict:
    async def checked(function):
        try:
            async with asyncio.timeout(settings.service_timeout_seconds + 2):
                return await function(settings)
        except Exception as exc:
            # 异常内容可能包含连接串和凭据，仅返回类型。
            return {"status": "error", "error_type": type(exc).__name__}

    names = ["postgresql", "redis", "minio", "neo4j"]
    functions = [check_postgres, check_redis, check_minio, check_neo4j]
    results = await asyncio.gather(*(checked(function) for function in functions))
    checks = dict(zip(names, results, strict=True))
    return {
        "status": "ok" if all(r["status"] in {"ok", "disabled"} for r in results) else "error",
        "checks": checks,
    }


def main() -> None:
    settings = get_settings()
    report = asyncio.run(check_services(settings))
    report["python"] = sys.version.split()[0]
    report["models"] = {
        "dashscope": "configured" if settings.dashscope_api_key.get_secret_value() else "missing",
        "deepseek": "configured"
        if settings.deepseek_api_key.get_secret_value()
        else "not_configured",
        "api_calls_verified": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
