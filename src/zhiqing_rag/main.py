"""FastAPI 开发入口；业务模块将在环境就绪后逐步实现。"""

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from zhiqing_rag.checks import check_services
from zhiqing_rag.core.config import get_settings
from zhiqing_rag.core.upload_limits import UploadBodyLimitMiddleware
from zhiqing_rag.modules.documents.router import router as document_router
from zhiqing_rag.modules.documents.schemas import UploadError


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0")
    app.add_middleware(
        UploadBodyLimitMiddleware, max_bytes=settings.upload_max_file_size + 1024 * 1024
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.include_router(document_router)

    @app.exception_handler(UploadError)
    async def upload_error_handler(request: Request, error: UploadError):
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": {"code": error.code, "message": str(error)}},
        )

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
