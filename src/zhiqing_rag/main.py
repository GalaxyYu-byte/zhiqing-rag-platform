"""FastAPI 开发入口；业务模块将在环境就绪后逐步实现。"""

import uvicorn
from fastapi import FastAPI, Response

from zhiqing_rag.checks import check_services
from zhiqing_rag.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")

    @app.get("/health/live", tags=["health"])
    async def liveness():
        return {"status": "ok", "application": settings.app_name}

    @app.get("/health/ready", tags=["health"])
    async def readiness(response: Response):
        report = await check_services(settings)
        if report["status"] != "ok":
            response.status_code = 503
        return report

    return app


app = create_app()


if __name__ == "__main__":
    config = get_settings()
    uvicorn.run(
        "zhiqing_rag.main:app",
        host=config.app_host,
        port=config.app_port,
        reload=config.app_env == "dev",
        log_level=config.log_level.lower(),
    )
