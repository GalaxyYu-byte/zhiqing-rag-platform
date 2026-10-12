"""FastAPI 入口，注册认证、文档接口与健康检查。"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from zhiqing_rag.checks import check_services
from zhiqing_rag.core.config import get_settings
from zhiqing_rag.core.upload_limits import UploadBodyLimitMiddleware
from zhiqing_rag.infrastructure.database.connection import close_database_pool
from zhiqing_rag.modules.chat.router import router as chat_router
from zhiqing_rag.modules.documents.router import router as document_router
from zhiqing_rag.modules.documents.schemas import UploadError
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import router as auth_router
from zhiqing_rag.modules.knowledge.router import router as permission_router
from zhiqing_rag.query_analysis.router import router as query_analysis_router
from zhiqing_rag.query_answering.router import router as query_answer_router
from zhiqing_rag.query_retrieval.router import router as query_retrieval_router
from zhiqing_rag.query_routing.router import router as query_routing_router


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            await run_in_threadpool(close_database_pool, settings)

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        UploadBodyLimitMiddleware, max_bytes=settings.upload_max_file_size + 1024 * 1024
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    )
    app.include_router(auth_router)
    app.include_router(document_router)
    app.include_router(permission_router)
    app.include_router(query_analysis_router)
    app.include_router(query_routing_router)
    app.include_router(query_retrieval_router)
    app.include_router(query_answer_router)
    app.include_router(chat_router)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, error: RequestValidationError):
        # FastAPI 默认返回无效字段的原值；认证字段可能包含密码或令牌。
        if request.url.path.startswith("/api/auth/"):
            return JSONResponse(
                status_code=422,
                content={"detail": {"code": "INVALID_AUTH_REQUEST", "message": "请检查登录参数"}},
                headers={"Cache-Control": "no-store"},
            )
        return await request_validation_exception_handler(request, error)

    @app.exception_handler(AuthError)
    async def auth_error_handler(request: Request, error: AuthError):
        headers = {"Cache-Control": "no-store"}
        if error.status_code == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": {"code": error.code, "message": str(error)}},
            headers=headers,
        )

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
