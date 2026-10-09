"""Redis 登录票据、刷新摘要及原子轮换；Redis 故障时拒绝签发。"""

import hashlib
import json

from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

from zhiqing_rag.core.config import Settings

from .auth_schemas import AuthError
from .auth_tokens import SELECTION_TTL

CONSUME_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
redis.call('DEL', KEYS[1])
return 1
"""
ROTATE_SCRIPT = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
if redis.call('EXISTS', KEYS[2]) == 1 then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
redis.call('DEL', KEYS[1])
return 1
"""
RATE_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return count
"""


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_auth_redis(settings: Settings) -> Redis:
    return Redis(
        host=settings.redis_host, port=settings.redis_port, db=settings.redis_db,
        password=settings.redis_password.get_secret_value() or None,
        max_connections=settings.redis_max_connections,
        socket_connect_timeout=settings.service_timeout_seconds,
        socket_timeout=settings.service_timeout_seconds,
        retry=Retry(NoBackoff(), 0), decode_responses=True,
    )


def unavailable(error: Exception) -> AuthError:
    return AuthError(503, "AUTH_STORE_UNAVAILABLE", "登录服务暂不可用，请稍后重试")


class AuthStore:
    def __init__(self, client: Redis, prefix: str):
        self.client = client
        self.prefix = prefix + "auth:"

    def key(self, kind: str, digest: str) -> str:
        return self.prefix + kind + ":" + digest

    @staticmethod
    def serialize(value: dict) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    def limit_login(self, email: str, ip_address: str) -> None:
        identifiers = (
            ("ip", ip_address, 30), ("email", email, 20),
            ("pair", ip_address + "\0" + email, 10),
        )
        try:
            for name, value, limit in identifiers:
                count = self.client.eval(
                    RATE_SCRIPT, 1, self.key("rate:" + name, token_hash(value)), 300
                )
                if count > limit:
                    raise AuthError(429, "LOGIN_RATE_LIMITED", "登录尝试过于频繁，请稍后重试")
        except RedisError as error:
            raise unavailable(error) from error

    def put_selection(self, token: str, value: dict) -> None:
        self._put("selection", token_hash(token), value, SELECTION_TTL)

    def put_refresh(self, digest: str, value: dict, ttl: int) -> None:
        self._put("refresh", digest, value, ttl)

    def _put(self, kind: str, digest: str, value: dict, ttl: int) -> None:
        try:
            if not self.client.set(self.key(kind, digest), self.serialize(value), ex=ttl, nx=True):
                raise AuthError(503, "AUTH_STORE_UNAVAILABLE", "登录服务暂不可用，请稍后重试")
        except RedisError as error:
            raise unavailable(error) from error

    def get(self, kind: str, token: str) -> tuple[dict, str]:
        if not token or len(token) > 256:
            self.invalid(kind)
        try:
            raw = self.client.get(self.key(kind, token_hash(token)))
        except RedisError as error:
            raise unavailable(error) from error
        if raw is None:
            self.invalid(kind)
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                self.invalid(kind)
            return value, raw
        except (ValueError, TypeError) as error:
            raise AuthError(401, "REFRESH_TOKEN_INVALID" if kind == "refresh" else
                            "SELECTION_TOKEN_INVALID", "登录凭证已失效，请重新登录") from error

    @staticmethod
    def invalid(kind: str) -> None:
        raise AuthError(401, "REFRESH_TOKEN_INVALID" if kind == "refresh" else
                        "SELECTION_TOKEN_INVALID", "登录凭证已失效，请重新登录")

    def consume_selection(self, token: str, raw: str) -> None:
        try:
            consumed = self.client.eval(
                CONSUME_SCRIPT, 1, self.key("selection", token_hash(token)), raw
            )
        except RedisError as error:
            raise unavailable(error) from error
        if not consumed:
            self.invalid("selection")

    def rotate(self, old_token: str, old_raw: str, new_digest: str, value: dict, ttl: int) -> None:
        try:
            rotated = self.client.eval(
                ROTATE_SCRIPT, 2, self.key("refresh", token_hash(old_token)),
                self.key("refresh", new_digest), old_raw, self.serialize(value), ttl,
            )
        except RedisError as error:
            raise unavailable(error) from error
        if not rotated:
            self.invalid("refresh")

    def delete_refreshes(self, digests: list[str]) -> None:
        if not digests:
            return
        try:
            self.client.delete(*(self.key("refresh", digest) for digest in digests))
        except RedisError as error:
            raise unavailable(error) from error
