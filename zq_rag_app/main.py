"""FastAPI 应用入口模块。"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from .api.auth import router as auth_router
from .api.document import router as document_router
from .api.knowledge_base import router as knowledge_base_router
from .api.retrieval import router as retrieval_router
from .core.config import settings
from .core.database import close_database
from .core.executor import shutdown_index_executor
from .core.minio import ensure_bucket, ping_minio
from .core.redis import close_redis, ping_redis
from .core.security import DEFAULT_ADMIN_USER, clear_user_context, set_user_context
from .core.task_queue import close_task_queue


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """管理应用级资源的生命周期。"""

    try:
        bucket = await asyncio.to_thread(ensure_bucket)
        logger.info("MinIO Bucket 已就绪: %s", bucket)
    except Exception:
        # 不因为 MinIO 暂时不可用阻止应用启动；/health 会返回 503。
        logger.exception("MinIO Bucket 初始化失败")

    yield
    await asyncio.to_thread(shutdown_index_executor)
    await close_task_queue()
    await close_redis()
    await close_database()


# 创建 FastAPI 应用实例。
# title 会从配置文件中读取，作为接口文档中的应用名称。
app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
)
app.include_router(auth_router)
app.include_router(document_router)
app.include_router(knowledge_base_router)
app.include_router(retrieval_router)


@app.middleware("http")
async def bind_current_user(request: Request, call_next):
    """认证模块接入前，为每个请求绑定固定的管理员登录态。"""

    token = set_user_context(DEFAULT_ADMIN_USER)
    request.state.current_user = DEFAULT_ADMIN_USER
    try:
        return await call_next(request)
    finally:
        clear_user_context(token)


@app.get("/health")
async def health(response: Response):
    """检查应用、Redis 和 MinIO 的可用性。"""

    checks: dict[str, dict[str, str]] = {}

    try:
        redis_ok = await ping_redis()
        checks["redis"] = {"status": "up" if redis_ok else "down"}
    except Exception:
        logger.exception("Redis 健康检查失败")
        checks["redis"] = {"status": "down"}

    try:
        minio_ok = await asyncio.to_thread(ping_minio)
        checks["minio"] = {"status": "up" if minio_ok else "down"}
    except Exception:
        logger.exception("MinIO 健康检查失败")
        checks["minio"] = {"status": "down"}

    healthy = all(item["status"] == "up" for item in checks.values())
    response.status_code = 200 if healthy else 503

    return {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
    }


# 前端工作台使用原生静态资源，挂载在所有 API 路由之后，避免覆盖接口。
_static_dir = Path(__file__).with_name("static")
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")


if __name__ == "__main__":
    # 直接运行本文件时，使用 Uvicorn 启动 FastAPI 服务。
    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
    )
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        # Uvicorn 在 Windows 单进程模式会显式创建 ProactorEventLoop，
        # psycopg 异步驱动必须在 SelectorEventLoop 中运行。
        asyncio.run(server.serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        server.run()
