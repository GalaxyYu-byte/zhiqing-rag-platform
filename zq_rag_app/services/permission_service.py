"""知识库访问权限判断。"""

from enum import StrEnum

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.security import UserContext, is_admin
from ..models.knowledge_base import KnowledgeBase, Permission
from ..models.document import Document


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


_CLEARANCE_RANK = {"内部公开": 0, "部门内部": 1, "机密": 2}


async def list_accessible_document_ids(
    session: AsyncSession,
    *,
    user: UserContext,
    kb_ids: list[int],
    requested_doc_ids: list[int] | None = None,
) -> list[int]:
    """返回用户在知识库范围内可检索的活动文档 ID。"""

    normalized_kb_ids = sorted(set(kb_ids))
    if not normalized_kb_ids:
        return []
    filters = [
        Document.kb_id.in_(normalized_kb_ids),
        Document.is_deleted.is_(False),
        Document.status == "DONE",
    ]
    if requested_doc_ids is not None:
        filters.append(Document.id.in_(sorted(set(requested_doc_ids))))
    if not is_admin(user):
        clearance = _CLEARANCE_RANK.get(user.clearance, -1)
        access_rules = [Document.confidentiality == "内部公开"]
        if clearance >= _CLEARANCE_RANK["部门内部"]:
            access_rules.append(
                and_(
                    Document.department_id == user.department_id,
                    Document.confidentiality == "部门内部",
                )
            )
        if clearance >= _CLEARANCE_RANK["机密"]:
            access_rules.append(
                and_(
                    Document.department_id == user.department_id,
                    Document.confidentiality == "机密",
                )
            )
        filters.append(or_(*access_rules))
    rows = await session.execute(select(Document.id).where(*filters))
    return sorted({int(row[0]) for row in rows.all()})
