"""当前请求的用户上下文与 FastAPI 登录态依赖。"""

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status


@dataclass(frozen=True, slots=True)
class UserContext:
    """业务接口需要使用的当前用户信息。"""

    user_id: int
    username: str
    department_id: str
    role: str
    clearance: str = "内部公开"


# 认证系统接入前统一使用这个管理员账号。后续只需替换请求中间件中的用户解析，
# 下游接口和权限服务无需改动。
DEFAULT_ADMIN_USER = UserContext(
    user_id=1,
    username="admin",
    department_id="ADMIN",
    role="ADMIN",
    clearance="机密",
)

_current_user: ContextVar[UserContext | None] = ContextVar(
    "current_user",
    default=None,
)


def set_user_context(
    user_id: int | UserContext,
    department_id: str | None = None,
    role: str | None = None,
    username: str | None = None,
    clearance: str = "内部公开",
) -> Token[UserContext | None]:
    """把用户绑定到当前请求上下文。

    同时兼容原有的 ``set_user_context(user_id, department_id, role)`` 调用方式。
    """

    if isinstance(user_id, UserContext):
        user = user_id
    else:
        if department_id is None or role is None:
            raise ValueError("department_id 和 role 不能为空")
        user = UserContext(
            user_id=user_id,
            username=username or str(user_id),
            department_id=department_id,
            role=role,
            clearance=clearance,
        )
    return _current_user.set(user)


def get_user_context() -> UserContext | None:
    """获取当前请求用户；请求尚未绑定登录态时返回 ``None``。"""

    return _current_user.get()


def require_current_user() -> UserContext:
    """FastAPI 依赖：返回当前用户，缺少登录态时返回 401。"""

    user = get_user_context()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户未登录",
        )
    return user


CurrentUser = Annotated[UserContext, Depends(require_current_user)]


def clear_user_context(token: Token[UserContext | None] | None = None) -> None:
    """清理当前用户；传入 token 时恢复绑定前的上下文。"""

    if token is None:
        _current_user.set(None)
    else:
        _current_user.reset(token)


def get_user_id() -> int | None:
    user = get_user_context()
    return user.user_id if user else None


def get_department_id() -> str | None:
    user = get_user_context()
    return user.department_id if user else None


def get_role() -> str | None:
    user = get_user_context()
    return user.role if user else None


def is_admin(user: UserContext | None = None) -> bool:
    """判断指定用户或当前请求用户是否为管理员。"""

    target = user or get_user_context()
    return target is not None and target.role.upper() == "ADMIN"
