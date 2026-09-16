"""独立 Graph RAG ARQ Worker。

启动命令：uv run python -m arq zq_rag_app.workers.graph_worker.WorkerSettings
"""

from typing import Any

from arq import Retry as JobRetry
from arq import func
from arq.connections import RedisSettings

from ..core.config import settings
from ..services.graph_task_service import (
    RetryableGraphTaskError,
    is_retryable_graph_exception,
    run_graph_task,
)
from .index_worker import shutdown_worker, startup_worker


async def execute_graph_index(
    context: dict[str, Any],
    task_id: int,
) -> bool:
    try:
        return await run_graph_task(task_id)
    except RetryableGraphTaskError as exc:
        raise JobRetry(defer=exc.delay_seconds) from exc
    except Exception as exc:
        job_try = int(context.get("job_try", 1))
        if (
            is_retryable_graph_exception(exc)
            and job_try <= settings.graph_task_max_retry
        ):
            delay = min(
                120.0,
                settings.graph_task_retry_base_seconds
                * (2 ** max(0, job_try - 1)),
            )
            raise JobRetry(defer=delay) from exc
        raise


class WorkerSettings:
    functions = [
        func(
            execute_graph_index,
            name="execute_graph_index",
            max_tries=settings.graph_task_max_retry + 1,
            keep_result=60,
        )
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    queue_name = settings.graph_task_queue_name
    max_jobs = settings.graph_worker_max_jobs
    job_timeout = 60 * 60
    on_startup = startup_worker
    on_shutdown = shutdown_worker
