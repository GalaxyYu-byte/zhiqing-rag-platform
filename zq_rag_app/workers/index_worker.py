"""ARQ 文档索引 Worker 配置。

启动命令：uv run arq zq_rag_app.workers.index_worker.WorkerSettings
"""

from typing import Any

from arq import Retry, func
from arq.connections import RedisSettings

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
        raise Retry(defer=exc.delay_seconds) from exc
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
            raise Retry(defer=delay) from exc
        raise


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
    on_shutdown = shutdown_worker
