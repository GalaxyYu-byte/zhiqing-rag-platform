"""ARQ 分布式索引任务队列客户端。"""

import asyncio

from arq.connections import ArqRedis, RedisSettings, create_pool

from .config import settings


_pool: ArqRedis | None = None
_pool_lock = asyncio.Lock()


async def get_task_queue() -> ArqRedis:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is None:
            _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return _pool


async def enqueue_document_index(task_id: int) -> bool:
    """按 task_id 去重投递；返回 False 表示相同任务已在队列中。"""

    queue = await get_task_queue()
    job = await queue.enqueue_job(
        "execute_document_index",
        task_id,
        _job_id=f"document-index:{task_id}",
    )
    return job is not None


async def close_task_queue() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
