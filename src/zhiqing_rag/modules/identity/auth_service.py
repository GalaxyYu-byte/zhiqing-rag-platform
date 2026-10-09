"""邮箱登录、租户选择、固定期限的刷新轮换和全部设备退出。"""

import ipaddress
import secrets
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from zhiqing_rag.core.config import Settings

from .auth_repository import AuthRepository
from .auth_schemas import (
    AuthError,
    AuthIdentity,
    AuthTenant,
    AuthUser,
    LoginResponse,
    TokenResponse,
)
from .auth_store import AuthStore, token_hash
from .auth_tokens import (
    REFRESH_TTL,
    decode_access_token,
    issue_access_token,
    require_unexpired,
    signing_key,
)
from .passwords import DUMMY_PASSWORD_HASH, verify_password


def current_time() -> datetime:
    return datetime.fromtimestamp(int(time.time()), UTC)


def public_identity(row: dict) -> tuple[AuthUser, AuthTenant]:
    return (
        AuthUser(id=str(row["user_id"]), email=row["email"], display_name=row["display_name"]),
        AuthTenant(id=str(row["tenant_id"]), code=row["tenant_code"],
                   name=row["tenant_name"], role=row["role"]),
    )


def validate_session(session: dict | None, now: datetime) -> dict:
    if session is None or session["revoked_at"] is not None or session["expires_at"] <= now:
        raise AuthError(401, "SESSION_REVOKED", "登录已失效，请重新登录")
    return session


def validate_authorization(row: dict | None, av: int, tav: int) -> dict:
    if row is None or row["auth_version"] != av or row["tenant_auth_version"] != tav:
        raise AuthError(401, "AUTHORIZATION_CHANGED", "账号或租户授权已变更，请重新登录")
    return row


def authorization_from_session(
    repository: AuthRepository, session_key: UUID, expected: tuple[int, int, int],
    auth_version: int, tenant_auth_version: int,
) -> dict:
    session, row = repository.session_identity(session_key)
    session = validate_session(session, current_time())
    if (session["user_id"], session["tenant_id"], session["tenant_member_id"]) != expected:
        raise AuthError(401, "ACCESS_TOKEN_INVALID", "登录凭证无效，请重新登录")
    row = validate_authorization(row, auth_version, tenant_auth_version)
    if (row["user_id"], row["tenant_id"], row["member_id"]) != expected:
        raise AuthError(401, "ACCESS_TOKEN_INVALID", "登录凭证无效，请重新登录")
    return row


def identity_from_access(repository: AuthRepository, claims: dict) -> AuthIdentity:
    row = authorization_from_session(
        repository, claims["sid"], (claims["sub"], claims["tid"], claims["mid"]),
        claims["av"], claims["tav"],
    )
    # 被撤销/禁用的过期令牌不会引发无意义的自动刷新。
    require_unexpired(claims)
    user, tenant = public_identity(row)
    return AuthIdentity(
        user_id=row["user_id"], tenant_id=row["tenant_id"], member_id=row["member_id"],
        department_id=row["department_id"], role=row["role"], clearance=row["clearance"],
        auth_version=row["auth_version"], tenant_auth_version=row["tenant_auth_version"],
        session_key=claims["sid"], user=user, tenant=tenant,
    )


class AuthService:
    def __init__(self, repository: AuthRepository, store: AuthStore, settings: Settings):
        self.repository = repository
        self.store = store
        self.settings = settings

    def login(self, email: str, password: str, ip_address: str) -> LoginResponse:
        signing_key(self.settings)
        self.store.limit_login(email, ip_address)
        user = self.repository.user_by_email(email)
        valid = verify_password(password, user["password_hash"] if user else DUMMY_PASSWORD_HASH)
        if not valid or user is None or not user["is_active"]:
            raise AuthError(401, "LOGIN_FAILED", "邮箱或密码错误")
        with self.repository.connection.transaction():
            locked = self.repository.lock_user(user["id"])
            if (locked is None or not locked["is_active"] or
                    locked["password_hash"] != user["password_hash"]):
                raise AuthError(401, "LOGIN_FAILED", "邮箱或密码错误")
            tenants = self.repository.tenants(user["id"])
            if not tenants:
                raise AuthError(403, "NO_AVAILABLE_TENANT", "账号没有可用的租户，请联系管理员")
            token = secrets.token_urlsafe(48)
            self.store.put_selection(token, {
                "user_id": user["id"], "password_stamp": token_hash(user["password_hash"]),
                "revocation_stamp": self.repository.revocation_stamp(user["id"]),
            })
        self.repository.connection.commit()
        return LoginResponse(
            selection_token=token,
            user=AuthUser(id=str(user["id"]), email=user["email"],
                          display_name=user["display_name"]),
            tenants=[AuthTenant(id=str(row["id"]), code=row["code"], name=row["name"],
                                role=row["role"]) for row in tenants],
        )

    def select_tenant(
        self, selection_token: str, tenant_id: int, user_agent: str | None,
        ip_address: str | None,
    ) -> tuple[TokenResponse, str, int]:
        signing_key(self.settings)
        ticket, raw = self.store.get("selection", selection_token)
        try:
            user_id = ticket["user_id"]
            if type(user_id) is not int or user_id <= 0:
                self.store.invalid("selection")
            password_stamp = ticket["password_stamp"]
            revocation_stamp = ticket["revocation_stamp"]
        except (KeyError, TypeError):
            self.store.invalid("selection")
        refresh = secrets.token_urlsafe(48)
        digest = token_hash(refresh)
        now = current_time()
        expires_at = datetime.fromtimestamp(int(now.timestamp()) + REFRESH_TTL, UTC)
        session_key = uuid4()
        try:
            with self.repository.connection.transaction():
                user = self.repository.lock_user(user_id)
                if (user is None or not user["is_active"] or
                        token_hash(user["password_hash"]) != password_stamp or
                        self.repository.revocation_stamp(user_id) != revocation_stamp):
                    self.store.invalid("selection")
                row = self.repository.identity(user_id, tenant_id)
                if row is None:
                    raise AuthError(403, "TENANT_FORBIDDEN", "租户不可用或没有访问权限")
                response = self._token_response(row, session_key, now)
                self.store.consume_selection(selection_token, raw)
                self.repository.create_session(
                    row, session_key, digest, now, expires_at,
                    user_agent[:500] if user_agent else None, self._ip(ip_address),
                )
                self.store.put_refresh(
                    digest, self._refresh_record(row, session_key, 0, expires_at), REFRESH_TTL,
                )
            # 在发 Cookie 前确认 DB 提交，避免请求依赖清理阶段才提交。
            self.repository.connection.commit()
        except Exception:
            self._discard_refresh(digest)
            raise
        return response, refresh, REFRESH_TTL

    def refresh(self, refresh_token: str) -> tuple[TokenResponse, str, int]:
        signing_key(self.settings)
        record, raw = self.store.get("refresh", refresh_token)
        session_key, user_id = self._record_identity(record)
        new_refresh = secrets.token_urlsafe(48)
        digest = token_hash(new_refresh)
        now = current_time()
        try:
            with self.repository.connection.transaction():
                user = self.repository.lock_user(user_id)
                if user is None or not user["is_active"]:
                    raise AuthError(401, "AUTHORIZATION_CHANGED", "账号已停用，请重新登录")
                session = validate_session(self.repository.session(session_key, lock=True), now)
                self._match_record(session, record)
                row = validate_authorization(
                    self.repository.identity(user_id, session["tenant_id"],
                                             session["tenant_member_id"]),
                    record["auth_version"], record["tenant_auth_version"],
                )
                ttl = int(session["expires_at"].timestamp()) - int(now.timestamp())
                if ttl <= 0:
                    self.store.invalid("refresh")
                response = self._token_response(row, session_key, now)
                self.store.rotate(
                    refresh_token, raw, digest,
                    self._refresh_record(row, session_key, session["rotation"] + 1,
                                         session["expires_at"]), ttl,
                )
                self.repository.rotate(session_key, digest, now)
            self.repository.connection.commit()
        except Exception:
            self._discard_refresh(digest)
            raise
        return response, new_refresh, ttl

    def logout(self, access_token: str | None, refresh_token: str | None) -> None:
        # 正常 Access 或仍有效的 Refresh 都可退出；过期 Access 自身不能撤销后续会话。
        if access_token:
            try:
                claims = decode_access_token(self.settings, access_token)
                require_unexpired(claims)
                session_key, user_id = claims["sid"], claims["sub"]
                record = None
            except AuthError as error:
                if error.code not in {"ACCESS_TOKEN_EXPIRED", "ACCESS_TOKEN_INVALID"}:
                    raise
                if not refresh_token:
                    raise
                record, _ = self.store.get("refresh", refresh_token)
                session_key, user_id = self._record_identity(record)
        elif refresh_token:
            record, _ = self.store.get("refresh", refresh_token)
            session_key, user_id = self._record_identity(record)
        else:
            raise AuthError(401, "AUTH_REQUIRED", "请先登录")
        now = current_time()
        with self.repository.connection.transaction():
            if self.repository.lock_user(user_id) is None:
                raise AuthError(401, "SESSION_REVOKED", "登录已失效，请重新登录")
            session = validate_session(self.repository.session(session_key, lock=True), now)
            if session["user_id"] != user_id:
                raise AuthError(401, "ACCESS_TOKEN_INVALID", "登录凭证无效，请重新登录")
            if record is not None:
                self._match_record(session, record)
                # 再读当前摘要，防止等待锁期间该 Refresh 已被另一个请求轮换。
                current, _ = self.store.get("refresh", refresh_token)
                if current != record:
                    self.store.invalid("refresh")
            elif (session["tenant_id"], session["tenant_member_id"]) != (
                claims["tid"], claims["mid"],
            ):
                raise AuthError(401, "ACCESS_TOKEN_INVALID", "登录凭证无效，请重新登录")
            revoked = self.repository.revoke_user(user_id, now)
        self.repository.connection.commit()
        # DB 撤销已持久化；即便 Redis 此时故障，所有 Access/Refresh 也已被拒绝。
        try:
            self.store.delete_refreshes([row["refresh_token_hash"] for row in revoked])
        except AuthError:
            pass

    def _token_response(self, row: dict, session_key: UUID, now: datetime) -> TokenResponse:
        user, tenant = public_identity(row)
        return TokenResponse(
            access_token=issue_access_token(self.settings, row, session_key, int(now.timestamp())),
            user=user, tenant=tenant,
        )

    @staticmethod
    def _refresh_record(row: dict, sid: UUID, rotation: int, expires_at: datetime) -> dict:
        return {
            "sid": str(sid), "user_id": row["user_id"], "tenant_id": row["tenant_id"],
            "member_id": row["member_id"], "auth_version": row["auth_version"],
            "tenant_auth_version": row["tenant_auth_version"], "rotation": rotation,
            "expires_at": int(expires_at.timestamp()),
        }

    def _record_identity(self, record: dict) -> tuple[UUID, int]:
        try:
            for key in ("user_id", "tenant_id", "member_id", "auth_version",
                        "tenant_auth_version", "rotation", "expires_at"):
                if type(record[key]) is not int or record[key] < (0 if key == "rotation" else 1):
                    self.store.invalid("refresh")
            if record["expires_at"] <= int(current_time().timestamp()):
                self.store.invalid("refresh")
            return UUID(record["sid"]), record["user_id"]
        except (KeyError, ValueError, TypeError, AttributeError):
            self.store.invalid("refresh")

    def _match_record(self, session: dict, record: dict) -> None:
        if (session["user_id"], session["tenant_id"], session["tenant_member_id"],
            session["rotation"], int(session["expires_at"].timestamp())) != (
            record["user_id"], record["tenant_id"], record["member_id"],
            record["rotation"], record["expires_at"],
        ):
            self.store.invalid("refresh")

    def _discard_refresh(self, digest: str) -> None:
        try:
            self.store.delete_refreshes([digest])
        except AuthError:
            pass

    @staticmethod
    def _ip(value: str | None) -> str | None:
        try:
            return str(ipaddress.ip_address(value)) if value else None
        except ValueError:
            return None
