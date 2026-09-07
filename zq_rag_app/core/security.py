"""当前请求用户上下文。"""

from contextvars import ContextVar, Token
from dataclasses import dataclass


# ContextVar 用于存储当前请求的用户上下文信息，支持异步环境下的上下文隔离。
@dataclass(frozen=True, slots=True)
class UserContext:
    """当前用户的最小权限信息。"""

    user_id: int
    department_id: str
    role: str

# ContextVar 用于存储当前请求的用户上下文信息，支持异步环境下的上下文隔离。
_current_user: ContextVar[UserContext | None] = ContextVar(
    "current_user",
    default=None,
)

# ContextVar 是 Python 3.7 引入的一个类，用于在异步任务中存储和管理上下文变量。它允许在不同的协程或线程中保持独立的上下文状态，从而避免数据冲突和共享问题。在 FastAPI 等异步框架中，ContextVar 常用于存储请求相关的信息，如当前用户、请求 ID 等，以便在整个请求处理过程中访问这些信息。
def set_user_context(
    user_id: int,
    department_id: str,
    role: str,
) -> Token[UserContext | None]:
    """设置当前用户，并返回用于恢复上下文的 token。"""

    return _current_user.set(
        UserContext(
            user_id=user_id,
            department_id=department_id,
            role=role,
        )
    )

# ContextVar.set() 方法用于设置当前上下文变量的值，并返回一个 Token 对象。这个 Token 对象可以在之后用于恢复上下文变量的先前状态。通过使用 Token，可以在异步任务中安全地修改和恢复上下文变量，确保不同协程或线程之间的上下文隔离，从而避免数据冲突和共享问题。在 FastAPI 等异步框架中，这种机制常用于管理请求相关的信息，如当前用户、请求 ID 等。
def get_user_context() -> UserContext | None:
    """获取当前用户；未认证时返回 None。"""

    return _current_user.get()


def clear_user_context(token: Token[UserContext | None] | None = None) -> None:
    """清理当前用户上下文。"""

    if token is None:
        _current_user.set(None)
    else:
        _current_user.reset(token)


def get_user_id() -> int | None:
    """获取当前用户 ID。"""

    user = get_user_context()
    return user.user_id if user else None


def get_department_id() -> str | None:
    """获取当前用户部门 ID。"""

    user = get_user_context()
    return user.department_id if user else None


def get_role() -> str | None:
    """获取当前用户角色。"""

    user = get_user_context()
    return user.role if user else None


def is_admin() -> bool:
    """判断当前用户是否为管理员。"""

    role = get_role()
    return role is not None and role.upper() == "ADMIN"
