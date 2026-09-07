"""Redis 异步客户端。"""

from collections.abc import AsyncIterator

from redis.asyncio import Redis

from .config import settings


# Redis.from_url() 只创建客户端，不会在导入模块时立即建立网络连接。
# 真正执行 get/set/ping 时才会从连接池获取连接。
redis_client: Redis = Redis.from_url(
    # Redis 连接 URL，格式为 redis://[:password]@host:port/db
    settings.redis_url,
    # 设置最大连接数，避免过多连接导致 Redis 服务器拒绝服务。
    max_connections=settings.redis_max_connections,
    # 解码响应为字符串
    decode_responses=True,
)


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
