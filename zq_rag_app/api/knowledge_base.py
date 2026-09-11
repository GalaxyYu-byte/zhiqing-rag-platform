"""知识库创建与列表接口。"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.security import CurrentUser, is_admin
from ..models.document import Document
from ..models.knowledge_base import KnowledgeBase


router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


class KnowledgeBaseListItemResponse(BaseModel):
    id: int
    name: str
    description: str | None
    department_id: str
    is_public: bool
    created_by: int
    created_at: datetime
    document_count: int

    @classmethod
    def from_model(
        cls,
        knowledge_base: KnowledgeBase,
        *,
        document_count: int = 0,
    ) -> "KnowledgeBaseListItemResponse":
        return cls(
            id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description,
            department_id=knowledge_base.department_id,
            is_public=knowledge_base.is_public,
            created_by=knowledge_base.created_by,
            created_at=knowledge_base.created_at,
            document_count=document_count,
        )


class KnowledgeBaseListResponse(BaseModel):
    total: int
    items: list[KnowledgeBaseListItemResponse]


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=2_000)
    department_id: str | None = Field(default=None, min_length=1, max_length=50)
    is_public: bool = False

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("知识库名称不能为空")
        return value

    @field_validator("description", "department_id")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None


@router.post(
    "",
    response_model=KnowledgeBaseListItemResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBaseListItemResponse:
    """使用当前登录用户创建知识库。"""

    department_id = request.department_id or current_user.department_id
    if department_id != current_user.department_id and not is_admin(current_user):
        raise HTTPException(status_code=403, detail="不能为其他部门创建知识库")

    duplicate_id = await session.scalar(
        select(KnowledgeBase.id).where(
            KnowledgeBase.department_id == department_id,
            func.lower(KnowledgeBase.name) == request.name.lower(),
            KnowledgeBase.is_deleted.is_(False),
        )
    )
    if duplicate_id is not None:
        raise HTTPException(status_code=409, detail="该部门已存在同名知识库")

    knowledge_base = KnowledgeBase(
        name=request.name,
        description=request.description,
        department_id=department_id,
        is_public=request.is_public,
        created_by=current_user.user_id,
        is_deleted=False,
    )
    session.add(knowledge_base)
    await session.commit()
    await session.refresh(knowledge_base)
    return KnowledgeBaseListItemResponse.from_model(knowledge_base)


@router.get("", response_model=KnowledgeBaseListResponse)
async def get_knowledge_bases(
    keyword: Annotated[str | None, Query(max_length=100)] = None,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBaseListResponse:
    """返回数据库中所有未删除的知识库。

    当前默认登录态是管理员，因此返回全量数据；切换为真实用户登录态时，列表
    还需要按公开范围、创建者和 ``kb_permission`` 过滤。
    """

    document_count = (
        select(func.count(Document.id))
        .where(
            Document.kb_id == KnowledgeBase.id,
            Document.is_deleted.is_(False),
        )
        .correlate(KnowledgeBase)
        .scalar_subquery()
        .label("document_count")
    )
    filters = [KnowledgeBase.is_deleted.is_(False)]
    if keyword and keyword.strip():
        escaped = keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(KnowledgeBase.name.ilike(f"%{escaped}%", escape="\\"))

    rows = (
        await session.execute(
            select(KnowledgeBase, document_count)
            .where(*filters)
            .order_by(KnowledgeBase.updated_at.desc(), KnowledgeBase.id.desc())
        )
    ).all()
    items = [
        KnowledgeBaseListItemResponse.from_model(
            knowledge_base,
            document_count=int(count or 0),
        )
        for knowledge_base, count in rows
    ]
    return KnowledgeBaseListResponse(total=len(items), items=items)
