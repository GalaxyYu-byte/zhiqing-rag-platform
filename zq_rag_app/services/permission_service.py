"""知识库访问权限判断。"""

from enum import StrEnum

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.security import UserContext, is_admin
from ..models.knowledge_base import KnowledgeBase, Permission


class PermissionLevel(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    ADMIN = "ADMIN"


_SATISFYING_LEVELS: dict[PermissionLevel, tuple[str, ...]] = {
    PermissionLevel.READ: ("READ", "WRITE", "ADMIN"),
    PermissionLevel.WRITE: ("WRITE", "ADMIN"),
    PermissionLevel.ADMIN: ("ADMIN",),
}


async def has_knowledge_base_permission(
    session: AsyncSession,
    *,
    user: UserContext,
    kb_id: int,
    required: PermissionLevel,
) -> bool:
    """判断用户是否具备指定知识库权限。

    系统管理员直接放行。普通用户可通过知识库创建者、公开只读范围、用户授权
    或部门授权获得权限；高等级权限自动包含低等级权限。
    """

    if is_admin(user):
        return True

    permission_subject = or_(
        and_(
            Permission.subject_type == "USER",
            Permission.subject_id == str(user.user_id),
        ),
        and_(
            Permission.subject_type == "DEPARTMENT",
            Permission.subject_id == user.department_id,
        ),
    )
    explicit_permission = (
        select(Permission.id)
        .where(
            Permission.kb_id == KnowledgeBase.id,
            permission_subject,
            Permission.permission.in_(_SATISFYING_LEVELS[required]),
        )
        .exists()
    )

    access_rules = [
        KnowledgeBase.created_by == user.user_id,
        explicit_permission,
    ]
    if required is PermissionLevel.READ:
        access_rules.append(KnowledgeBase.is_public.is_(True))

    statement = select(KnowledgeBase.id).where(
        KnowledgeBase.id == kb_id,
        KnowledgeBase.is_deleted.is_(False),
        or_(*access_rules),
    )
    return await session.scalar(statement) is not None
