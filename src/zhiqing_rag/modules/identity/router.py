"""认证 HTTP 入口；刷新令牌只通过 HttpOnly Cookie 传输。"""

from collections.abc import Iterator
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Request, Response

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.infrastructure.database.connection import database_connection

from .auth_repository import AuthRepository
from .auth_schemas import (
    AuthError,
    AuthIdentity,
    CurrentIdentityResponse,
    LoginRequest,
    LoginResponse,
    SelectTenantRequest,
    TokenResponse,
)
from .auth_service import AuthService, identity_from_access
from .auth_store import AuthStore, create_auth_redis
from .auth_tokens import decode_access_token, signing_key

router = APIRouter(prefix="/api/auth", tags=["登录认证"])
SettingsDependency = Annotated[Settings, Depends(get_settings)]
COOKIE_PATH = "/api/auth"


def bearer_token(request: Request, *, required: bool = True) -> str | None:
    authorization = request.headers.get("authorization", "")
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    if required or authorization:
        raise AuthError(401, "AUTH_REQUIRED", "请先登录")
    return None


def require_identity(request: Request, settings: SettingsDependency) -> AuthIdentity:
    # 缺少 Bearer 的请求在任何基础设施连接前立即拒绝。
    token = bearer_token(request)
    claims = decode_access_token(settings, token)
    try:
        with database_connection(settings) as connection:
            return identity_from_access(AuthRepository(connection, settings.db_schema), claims)
    except psycopg.Error as error:
        raise AuthError(503, "DATABASE_UNAVAILABLE", "数据库暂不可用，请稍后重试") from error


def require_cookie_request(request: Request, settings: SettingsDependency) -> None:
    origin = request.headers.get("origin")
    if origin is not None and origin not in settings.cors_origins:
        raise AuthError(403, "ORIGIN_FORBIDDEN", "不允许此页面来源访问登录接口")
    if request.headers.get("x-requested-with") != "XMLHttpRequest":
        raise AuthError(403, "CSRF_VALIDATION_FAILED", "请求校验失败，请刷新页面后重试")


def get_auth_service(
    request: Request, settings: SettingsDependency,
    csrf: Annotated[None, Depends(require_cookie_request)],
) -> Iterator[AuthService]:
    signing_key(settings)
    try:
        with database_connection(settings) as connection, create_auth_redis(settings) as client:
            yield AuthService(
                AuthRepository(connection, settings.db_schema),
                AuthStore(client, settings.redis_key_prefix), settings,
            )
    except psycopg.Error as error:
        raise AuthError(503, "DATABASE_UNAVAILABLE", "数据库暂不可用，请稍后重试") from error


ServiceDependency = Annotated[AuthService, Depends(get_auth_service)]
IdentityDependency = Annotated[AuthIdentity, Depends(require_identity)]


def no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"


def set_refresh_cookie(response: Response, settings: Settings, token: str, ttl: int) -> None:
    no_store(response)
    response.set_cookie(
        key=settings.refresh_cookie_name, value=token, max_age=ttl,
        path=COOKIE_PATH, secure=settings.refresh_cookie_secure, httponly=True, samesite="lax",
    )


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request, response: Response, service: ServiceDependency):
    no_store(response)
    return service.login(body.email, body.password, request.client.host if request.client else "")


@router.post("/select-tenant", response_model=TokenResponse)
def select_tenant(
    body: SelectTenantRequest, request: Request, response: Response, service: ServiceDependency,
):
    result, refresh_token, ttl = service.select_tenant(
        body.selection_token, body.tenant_id, request.headers.get("user-agent"),
        request.client.host if request.client else None,
    )
    set_refresh_cookie(response, service.settings, refresh_token, ttl)
    return result


@router.post("/refresh", response_model=TokenResponse)
def refresh(request: Request, response: Response, service: ServiceDependency):
    token = request.cookies.get(service.settings.refresh_cookie_name)
    if not token:
        raise AuthError(401, "REFRESH_TOKEN_INVALID", "登录已失效，请重新登录")
    result, refresh_token, ttl = service.refresh(token)
    set_refresh_cookie(response, service.settings, refresh_token, ttl)
    return result


@router.get("/me", response_model=CurrentIdentityResponse)
def me(response: Response, identity: IdentityDependency):
    no_store(response)
    return CurrentIdentityResponse(user=identity.user, tenant=identity.tenant)


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response, service: ServiceDependency):
    service.logout(bearer_token(request, required=False),
                   request.cookies.get(service.settings.refresh_cookie_name))
    no_store(response)
    response.delete_cookie(
        service.settings.refresh_cookie_name, path=COOKIE_PATH,
        secure=service.settings.refresh_cookie_secure, httponly=True, samesite="lax",
    )
