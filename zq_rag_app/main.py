"""FastAPI 应用入口模块。"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from .api.document import router as document_router
from .core.config import settings
from .core.database import close_database
from .core.executor import shutdown_index_executor
from .core.minio import ensure_bucket, ping_minio
from .core.redis import close_redis, ping_redis
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
app.include_router(document_router)


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


if __name__ == "__main__":
    # 直接运行本文件时，使用 Uvicorn 启动 FastAPI 服务。
    import uvicorn

    uvicorn.run(
        # 模块路径:应用对象，便于 reload=True 重新加载代码。
        "zq_rag_app.main:app",
        # 监听所有网卡，使局域网内其他设备也可以访问。
        host="0.0.0.0",
        # 服务端口，可通过 http://localhost:8000 访问。
        port=8000,
        # 开发模式下代码修改后自动重启服务；生产环境不要开启。
        reload=True,
    )
