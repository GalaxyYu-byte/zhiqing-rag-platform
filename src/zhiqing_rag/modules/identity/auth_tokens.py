"""短期 Access JWT 的严格签发和解析。"""

import time
from uuid import UUID

import jwt

from zhiqing_rag.core.config import Settings

from .auth_schemas import AuthError

ACCESS_TTL = 15 * 60
REFRESH_TTL = 7 * 24 * 60 * 60
SELECTION_TTL = 5 * 60


def signing_key(settings: Settings) -> str:
    secret = settings.jwt_secret_key.get_secret_value()
    if len(secret.encode()) < 32:
        raise AuthError(503, "AUTH_CONFIG_INVALID", "登录服务尚未正确配置")
    return secret


def issue_access_token(settings: Settings, row: dict, session_key: UUID, now: int) -> str:
    return jwt.encode(
        {
            "iss": settings.jwt_issuer, "aud": settings.jwt_audience,
            "sub": str(row["user_id"]), "sid": str(session_key), "type": "access",
            "tid": str(row["tenant_id"]), "mid": str(row["member_id"]),
            "av": row["auth_version"], "tav": row["tenant_auth_version"],
            "iat": now, "exp": now + ACCESS_TTL,
        }, signing_key(settings), algorithm="HS256",
    )


def decode_access_token(settings: Settings, token: str) -> dict:
    try:
        claims = jwt.decode(
            token, signing_key(settings), algorithms=["HS256"],
            issuer=settings.jwt_issuer, audience=settings.jwt_audience,
            options={
                "verify_exp": False,
                "require": ["iss", "aud", "sub", "sid", "type", "tid", "mid", "av",
                            "tav", "iat", "exp"],
            },
        )
        if claims["type"] != "access":
            raise ValueError("Invalid token type")
        for name in ("av", "tav", "iat", "exp"):
            if type(claims[name]) is not int or claims[name] <= 0:
                raise ValueError("Invalid numeric claim")
        if claims["exp"] <= claims["iat"] or claims["exp"] - claims["iat"] > ACCESS_TTL:
            raise ValueError("Invalid token lifetime")
        for name in ("sub", "tid", "mid"):
            if not isinstance(claims[name], str) or not claims[name].isdigit():
                raise ValueError("Invalid identifier")
            claims[name] = int(claims[name])
            if claims[name] <= 0:
                raise ValueError("Invalid identifier")
        claims["sid"] = UUID(claims["sid"])
        return claims
    except (jwt.InvalidTokenError, ValueError, TypeError, AttributeError) as error:
        raise AuthError(401, "ACCESS_TOKEN_INVALID", "登录凭证无效，请重新登录") from error


def require_unexpired(claims: dict, now: int | None = None) -> None:
    if claims["exp"] <= (int(time.time()) if now is None else now):
        raise AuthError(401, "ACCESS_TOKEN_EXPIRED", "登录凭证已过期")
