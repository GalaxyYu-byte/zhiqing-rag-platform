"""知识库列表接口。"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
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


class KnowledgeBaseListResponse(BaseModel):
    total: int
    items: list[KnowledgeBaseListItemResponse]


@router.get("", response_model=KnowledgeBaseListResponse)
async def get_knowledge_bases(
    keyword: Annotated[str | None, Query(max_length=100)] = None,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBaseListResponse:
    """返回数据库中所有未删除的知识库。

    备注：认证与权限模块目前为空。接入登录态后，应在这里增加公开知识库、
    创建者和 ``kb_permission`` 的可见范围过滤，不能继续返回全量数据。
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
        KnowledgeBaseListItemResponse(
            id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description,
            department_id=knowledge_base.department_id,
            is_public=knowledge_base.is_public,
            created_by=knowledge_base.created_by,
            created_at=knowledge_base.created_at,
            document_count=int(count or 0),
        )
        for knowledge_base, count in rows
    ]
    return KnowledgeBaseListResponse(total=len(items), items=items)
