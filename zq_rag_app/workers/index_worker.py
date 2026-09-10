"""ARQ 文档索引 Worker 配置。

启动命令：uv run python -m arq zq_rag_app.workers.index_worker.WorkerSettings
"""

import asyncio
import sys
from typing import Any

from arq import Retry as JobRetry
from arq import func
from arq.connections import RedisSettings
from redis.asyncio.connection import ConnectionPool
from redis.asyncio.retry import Retry as RedisRetry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

# ARQ 通过当前事件循环运行任务；Windows 默认 ProactorEventLoop 与 psycopg
# 异步驱动不兼容，因此必须在 Worker 创建事件循环前切换策略。
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from ..core.config import settings
from ..core.database import close_database
from ..core.redis import close_redis
from ..services.document_service import (
    RetryableIndexTaskError,
    is_retryable_index_exception,
    run_document_index_task,
)


async def execute_document_index(
    context: dict[str, Any],
    task_id: int,
) -> bool:
    try:
        return await run_document_index_task(task_id)
    except RetryableIndexTaskError as exc:
        raise JobRetry(defer=exc.delay_seconds) from exc
    except Exception as exc:
        job_try = int(context.get("job_try", 1))
        if (
            is_retryable_index_exception(exc)
            and job_try <= settings.index_task_max_retry
        ):
            delay = min(
                60.0,
                settings.index_task_retry_base_seconds
                * (2 ** max(0, job_try - 1)),
            )
            raise JobRetry(defer=delay) from exc
        raise


async def startup_worker(context: dict[str, Any]) -> None:
    """为 ARQ 队列连接启用健康检查和断线自动重连。

    ARQ 的 ``RedisSettings`` 只暴露初次连接重试，无法配置命令执行期间的
    retry。Worker 创建连接后在这里替换连接池，后续任务完成状态写回也会使用
    带重试的新连接。
    """

    redis = context["redis"]
    old_pool = redis.connection_pool
    connection_kwargs = dict(old_pool.connection_kwargs)
    retry_errors = (
        RedisConnectionError,
        RedisTimeoutError,
        TypeError,
    )
    connection_kwargs.update(
        socket_connect_timeout=5,
        socket_timeout=10,
        socket_keepalive=True,
        health_check_interval=30,
        retry=RedisRetry(
            ExponentialBackoff(cap=1.0, base=0.05),
            retries=3,
            supported_errors=retry_errors,
        ),
        retry_on_error=list(retry_errors),
    )
    redis.connection_pool = ConnectionPool(
        max_connections=old_pool.max_connections,
        **connection_kwargs,
    )
    await old_pool.disconnect(inuse_connections=True)


async def shutdown_worker(_: dict[str, Any]) -> None:
    await close_redis()
    await close_database()


class WorkerSettings:
    functions = [
        func(
            execute_document_index,
            name="execute_document_index",
            max_tries=settings.index_task_max_retry + 1,
            keep_result=60,
        )
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = settings.task_max_workers
    job_timeout = 60 * 60
    on_startup = startup_worker
    on_shutdown = shutdown_worker
