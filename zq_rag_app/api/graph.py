"""独立图谱构建任务的创建和状态查询接口。"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.database import get_db
from ..core.security import CurrentUser, UserContext
from ..core.task_queue import enqueue_graph_index
from ..models.document import Document
from ..services.graph_task_service import (
    GraphTaskConflictError,
    GraphTaskNotFoundError,
    GraphTaskSnapshot,
    create_graph_task,
    get_graph_task_snapshot,
    get_latest_document_graph_snapshot,
)
from ..services.permission_service import PermissionLevel, has_knowledge_base_permission


logger = logging.getLogger(__name__)
router = APIRouter(tags=["graph"])


class GraphTaskStatusResponse(BaseModel):
    task_id: int
    doc_id: int
    kb_id: int
    doc_version: int
    extractor_version: str
    status: str
    stage: str
    progress_percent: int
    total_chunks: int
    processed_chunks: int
    extracted_entities: int
    extracted_claims: int
    retry_count: int
    max_retry: int
    error_msg: str | None
    worker_id: str | None
    heartbeat_at: datetime | None
    lease_expires_at: datetime | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: GraphTaskSnapshot,
    ) -> "GraphTaskStatusResponse":
        values = {
            field: getattr(snapshot, field)
            for field in snapshot.__dataclass_fields__
        }
        values["task_id"] = values.pop("id")
        return cls(**values)


class CreateGraphTaskResponse(GraphTaskStatusResponse):
    created: bool
    queued: bool


async def _require_permission(
    session: AsyncSession,
    *,
    doc_id: int,
    current_user: UserContext,
    required: PermissionLevel,
) -> Document:
    document = await session.scalar(
        select(Document).where(
            Document.id == doc_id,
            Document.is_deleted.is_(False),
        )
    )
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    allowed = await has_knowledge_base_permission(
        session,
        user=current_user,
        kb_id=document.kb_id,
        required=required,
    )
    if not allowed:
        raise HTTPException(status_code=403, detail="没有该知识库的操作权限")
    return document


@router.post(
    "/documents/{doc_id}/graph-index",
    response_model=CreateGraphTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_graph_index(
    doc_id: int,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> CreateGraphTaskResponse:
    if not settings.graph_extraction_enabled:
        raise HTTPException(status_code=409, detail="图谱抽取功能尚未启用")
    await _require_permission(
        session,
        doc_id=doc_id,
        current_user=current_user,
        required=PermissionLevel.WRITE,
    )
    try:
        task, created = await create_graph_task(session, doc_id=doc_id)
    except GraphTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GraphTaskConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        queued = await enqueue_graph_index(task.id)
    except Exception:
        logger.exception("图任务入队失败: task_id=%s", task.id)
        queued = False
    return CreateGraphTaskResponse(
        **GraphTaskStatusResponse.from_snapshot(
            GraphTaskSnapshot.from_model(task)
        ).model_dump(),
        created=created,
        queued=queued,
    )


@router.get(
    "/documents/{doc_id}/graph-status",
    response_model=GraphTaskStatusResponse,
)
async def get_document_graph_status(
    doc_id: int,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> GraphTaskStatusResponse:
    await _require_permission(
        session,
        doc_id=doc_id,
        current_user=current_user,
        required=PermissionLevel.READ,
    )
    snapshot = await get_latest_document_graph_snapshot(session, doc_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="文档尚无图谱任务")
    return GraphTaskStatusResponse.from_snapshot(snapshot)


@router.get(
    "/graph-tasks/{task_id}",
    response_model=GraphTaskStatusResponse,
)
async def get_graph_task_status(
    task_id: int,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> GraphTaskStatusResponse:
    snapshot = await get_graph_task_snapshot(session, task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="图谱任务不存在")
    await _require_permission(
        session,
        doc_id=snapshot.doc_id,
        current_user=current_user,
        required=PermissionLevel.READ,
    )
    return GraphTaskStatusResponse.from_snapshot(snapshot)
