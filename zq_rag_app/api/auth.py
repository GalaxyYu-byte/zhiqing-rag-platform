"""当前用户及知识库权限查询接口。"""

from typing import Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.security import CurrentUser, UserContext
from ..services.permission_service import (
    PermissionLevel,
    has_knowledge_base_permission,
)


router = APIRouter(prefix="/auth", tags=["auth"])


class CurrentUserResponse(BaseModel):
    user_id: int
    username: str
    department_id: str
    role: str
    clearance: str

    @classmethod
    def from_context(cls, user: UserContext) -> "CurrentUserResponse":
        return cls(
            user_id=user.user_id,
            username=user.username,
            department_id=user.department_id,
            role=user.role,
            clearance=user.clearance,
        )


class PermissionCheckResponse(BaseModel):
    kb_id: int
    permission: Literal["READ", "WRITE"]
    allowed: bool


@router.get("/me", response_model=CurrentUserResponse)
async def get_current_user(current_user: CurrentUser) -> CurrentUserResponse:
    """返回绑定在当前请求上下文中的登录用户。"""

    return CurrentUserResponse.from_context(current_user)


@router.get(
    "/knowledge-bases/{kb_id}/permission",
    response_model=PermissionCheckResponse,
)
async def check_knowledge_base_permission(
    kb_id: int,
    current_user: CurrentUser,
    permission: Literal["READ", "WRITE"] = Query(),
    session: AsyncSession = Depends(get_db),
) -> PermissionCheckResponse:
    """判断当前用户对指定知识库是否拥有读或写权限。"""

    allowed = await has_knowledge_base_permission(
        session,
        user=current_user,
        kb_id=kb_id,
        required=PermissionLevel(permission),
    )
    return PermissionCheckResponse(
        kb_id=kb_id,
        permission=permission,
        allowed=allowed,
    )
