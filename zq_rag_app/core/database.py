"""PostgreSQL 数据库连接与 Session。"""

from collections.abc import AsyncIterator
# 导入 SQLAlchemy 的异步相关模块，包括异步会话、异步会话工厂和异步数据库引擎创建函数。
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from .config import settings

# 创建异步数据库引擎，使用 SQLAlchemy 的 create_async_engine 方法。
engine = create_async_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True,
)
# 创建异步 Session 工厂，使用 async_sessionmaker 方法，并绑定到数据库引擎。
SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 定义 ORM 模型的基类，所有的 ORM 模型都应该继承自这个基类。
class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""

# 定义一个异步生成器函数，用于在 FastAPI 中作为依赖注入，提供数据库会话。
async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 数据库依赖。"""

    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

# 定义一个异步函数，用于在应用关闭时释放数据库连接池，确保资源的正确释放。
async def close_database() -> None:
    """应用关闭时释放数据库连接池。"""

    await engine.dispose()


# 导入所有 ORM 实体，确保 Base.metadata 包含全部业务表。
# 放在 Base 定义之后，避免模型模块导入 Base 时产生循环依赖问题。
from .. import models  # noqa: E402, F401
