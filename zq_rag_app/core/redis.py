"""Redis 异步客户端。"""

from collections.abc import AsyncIterator

from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from .config import settings


_REDIS_RETRY_ERRORS = (
    RedisConnectionError,
    RedisTimeoutError,
    # Python 3.12 SelectorSocketTransport 在连接已断开时可能抛出此异常，
    # redis-py 没有把它包装成 RedisError。
    TypeError,
)


def _create_redis_client(*, decode_responses: bool) -> Redis:
    """创建带断线检测和有限重连的异步 Redis 客户端。"""

    return Redis.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=decode_responses,
        socket_connect_timeout=5,
        socket_timeout=10,
        socket_keepalive=True,
        health_check_interval=30,
        retry=Retry(
            ExponentialBackoff(cap=1.0, base=0.05),
            retries=3,
            supported_errors=_REDIS_RETRY_ERRORS,
        ),
        retry_on_error=list(_REDIS_RETRY_ERRORS),
    )


# Redis.from_url() 只创建客户端，首次命令执行时才建立网络连接。
redis_client: Redis = _create_redis_client(decode_responses=True)

# Embedding 向量使用 float32 二进制存储，必须保留 bytes，不能自动解码为 str。
redis_binary_client: Redis = _create_redis_client(decode_responses=False)


async def get_redis() -> AsyncIterator[Redis]:
    """FastAPI 依赖：获取共享 Redis 客户端。"""
# yield 语句会暂停函数执行，返回 redis_client 给调用者。  
# 当调用者完成对 Redis 的操作后，函数会继续执行，允许在此处添加清理逻辑（如关闭连接）。 
    yield redis_client


async def ping_redis() -> bool:
    """检查 Redis 是否可用。"""

    return bool(await redis_client.ping())


async def close_redis() -> None:
    """应用关闭时释放 Redis 连接池。"""

    await redis_client.aclose()
    await redis_binary_client.aclose()
