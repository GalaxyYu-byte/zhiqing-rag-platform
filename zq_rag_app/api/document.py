"""文档索引任务创建和状态查询接口。"""

import logging
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.task_queue import enqueue_document_index
from ..services.document_service import (
    IndexTaskNotFoundError,
    IndexTaskSnapshot,
    create_index_task,
    get_index_task_snapshot,
    get_latest_document_index_snapshot,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


class CreateIndexRequest(BaseModel):
    task_type: Literal["INDEX", "REINDEX"] = "INDEX"


class IndexTaskStatusResponse(BaseModel):
    task_id: int
    doc_id: int
    task_type: str
    status: str
    stage: str
    progress_percent: int
    total_chunks: int
    embedded_chunks: int
    persisted_chunks: int
    cache_hit_chunks: int
    total_tokens: int
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
        cls, snapshot: IndexTaskSnapshot
    ) -> "IndexTaskStatusResponse":
        values = {
            field: getattr(snapshot, field)
            for field in snapshot.__dataclass_fields__
        }
        values["task_id"] = values.pop("id")
        return cls(**values)


class CreateIndexResponse(IndexTaskStatusResponse):
    created: bool
    queued: bool


@router.post(
    "/{doc_id}/index",
    response_model=CreateIndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_document_index(
    doc_id: int,
    request: CreateIndexRequest,
    session: AsyncSession = Depends(get_db),
) -> CreateIndexResponse:
    try:
        task, created = await create_index_task(
            session,
            doc_id=doc_id,
            task_type=request.task_type,
        )
    except IndexTaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        queued = await enqueue_document_index(task.id)
    except Exception:
        # 任务已经可靠地保存在数据库中，队列恢复后再次调用接口即可重新投递。
        logger.exception("索引任务入队失败: task_id=%s", task.id)
        queued = False

    snapshot = IndexTaskSnapshot.from_model(task)
    return CreateIndexResponse(
        **IndexTaskStatusResponse.from_snapshot(snapshot).model_dump(),
        created=created,
        queued=queued,
    )


@router.get(
    "/index-tasks/{task_id}",
    response_model=IndexTaskStatusResponse,
)
async def get_document_index_task(
    task_id: int,
    session: AsyncSession = Depends(get_db),
) -> IndexTaskStatusResponse:
    snapshot = await get_index_task_snapshot(session, task_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="索引任务不存在")
    return IndexTaskStatusResponse.from_snapshot(snapshot)


@router.get(
    "/{doc_id}/index-status",
    response_model=IndexTaskStatusResponse,
)
async def get_document_index_status(
    doc_id: int,
    session: AsyncSession = Depends(get_db),
) -> IndexTaskStatusResponse:
    snapshot = await get_latest_document_index_snapshot(session, doc_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="该文档没有索引任务")
    return IndexTaskStatusResponse.from_snapshot(snapshot)
