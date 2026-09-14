"""Neo4j 异步驱动生命周期管理。

驱动采用懒加载：应用启动时只读取配置，不主动连接 Neo4j；首次图谱操作
或显式健康检查时才建立连接。这样在图谱功能尚未启用时不会影响向量 RAG。
"""

from __future__ import annotations

from neo4j import AsyncDriver, AsyncGraphDatabase

from .config import settings


_driver: AsyncDriver | None = None


def get_neo4j_driver() -> AsyncDriver:
    """返回进程级 Neo4j 异步驱动；首次调用时创建。"""

    global _driver
    if _driver is not None:
        return _driver
    if not settings.neo4j_password:
        raise RuntimeError(
            "Neo4j 未配置密码，请设置 NEO4J_PASSWORD 后再启用图谱功能"
        )

    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
        max_connection_pool_size=settings.neo4j_max_connection_pool_size,
        connection_timeout=settings.neo4j_connection_timeout_seconds,
    )
    return _driver


def get_neo4j_session():
    """按配置打开一个 Neo4j 异步 Session。"""

    return get_neo4j_driver().session(database=settings.neo4j_database)


async def ping_neo4j() -> bool:
    """验证 Neo4j 网络连接和认证是否可用。"""

    driver = get_neo4j_driver()
    await driver.verify_connectivity()
    return True


async def close_neo4j() -> None:
    """关闭进程级 Neo4j 驱动。"""

    global _driver
    driver = _driver
    _driver = None
    if driver is not None:
        await driver.close()
