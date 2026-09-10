"""ARQ 分布式索引任务队列客户端。"""

import asyncio
import logging
from typing import Any

from arq.connections import ArqRedis, RedisSettings, create_pool
from redis.exceptions import RedisError

from .config import settings


logger = logging.getLogger(__name__)

# Python 3.12 的 SelectorSocketTransport 在 Redis 连接刚断开时可能直接抛出
# TypeError，而不是 redis-py 的 RedisError。这里只在 ARQ 网络调用边界捕获。
_QUEUE_IO_ERRORS = (
    RedisError,
    OSError,
    TimeoutError,
    ConnectionError,
    TypeError,
)

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
    """按 task_id 幂等投递，连接失效时重建客户端并重试一次。

    首次正常返回 ``None`` 仍表示同 Job ID 已存在，因此返回 False。首次网络
    调用抛错后，第二次只要未抛错就视为投递成功：此时返回 ``None`` 可能代表
    第一次请求其实已由 Redis 写入，只是响应阶段连接中断。
    """

    queue: ArqRedis | None = None
    for attempt in range(2):
        try:
            queue = await get_task_queue()
            job = await queue.enqueue_job(
                "execute_document_index",
                task_id,
                _job_id=f"document-index:{task_id}",
            )
            return job is not None or attempt > 0
        except _QUEUE_IO_ERRORS:
            if attempt > 0:
                logger.exception(
                    "ARQ 重连后索引任务仍入队失败: task_id=%s",
                    task_id,
                )
                await _discard_task_queue(queue)
                raise
            logger.warning(
                "ARQ 索引任务入队连接异常，重建连接后重试: task_id=%s",
                task_id,
                exc_info=True,
            )
            await _discard_task_queue(queue)
    return False


async def _discard_task_queue(queue: Any | None) -> None:
    """仅清理当前共享的失效客户端，避免并发请求关闭新连接。"""

    global _pool
    if queue is None:
        return
    should_close = False
    async with _pool_lock:
        if _pool is queue:
            _pool = None
            should_close = True
    if should_close:
        try:
            await queue.aclose()
        except Exception:
            logger.debug("关闭失效 ARQ 客户端失败", exc_info=True)


async def close_task_queue() -> None:
    global _pool
    async with _pool_lock:
        queue = _pool
        _pool = None
    if queue is not None:
        await queue.aclose()
