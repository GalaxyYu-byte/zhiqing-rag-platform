"""文档索引任务创建和状态查询接口。"""

import logging
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.task_queue import enqueue_document_index
from ..models.document import Document, IndexTask
from ..services.document_catalog_service import (
    DocumentStorageError,
    DocumentUploadValidationError,
    create_uploaded_documents,
    list_documents,
)
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


class DocumentTaskSummaryResponse(BaseModel):
    task_id: int
    status: str
    stage: str
    progress_percent: int
    total_chunks: int
    embedded_chunks: int
    persisted_chunks: int
    error_msg: str | None

    @classmethod
    def from_model(
        cls, task: IndexTask | None
    ) -> "DocumentTaskSummaryResponse | None":
        if task is None:
            return None
        return cls(
            task_id=task.id,
            status=task.status,
            stage=task.stage,
            progress_percent=task.progress_percent,
            total_chunks=task.total_chunks,
            embedded_chunks=task.embedded_chunks,
            persisted_chunks=task.persisted_chunks,
            error_msg=task.error_msg,
        )


class DocumentResponse(BaseModel):
    id: int
    kb_id: int
    file_name: str
    file_type: str
    file_size: int
    minio_path: str
    status: str
    error_msg: str | None
    chunk_count: int
    token_count: int
    version: int
    uploaded_by: int
    uploaded_at: datetime
    indexed_at: datetime | None
    latest_task: DocumentTaskSummaryResponse | None = None

    @classmethod
    def from_model(
        cls,
        document: Document,
        latest_task: IndexTask | None = None,
    ) -> "DocumentResponse":
        return cls(
            id=document.id,
            kb_id=document.kb_id,
            file_name=document.file_name,
            file_type=document.file_type,
            file_size=document.file_size,
            minio_path=document.minio_path,
            status=document.status,
            error_msg=document.error_msg,
            chunk_count=document.chunk_count,
            token_count=document.token_count,
            version=document.version,
            uploaded_by=document.uploaded_by,
            uploaded_at=document.uploaded_at,
            indexed_at=document.indexed_at,
            latest_task=DocumentTaskSummaryResponse.from_model(latest_task),
        )


class UploadedDocumentResponse(BaseModel):
    document: DocumentResponse
    index_task: CreateIndexResponse


class UploadDocumentsResponse(BaseModel):
    items: list[UploadedDocumentResponse]


class DocumentListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[DocumentResponse]


@router.post(
    "/upload",
    response_model=UploadDocumentsResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="待上传文档")],
    kb_id: Annotated[int, Form(gt=0)],
    # 认证模块目前尚未实现，测试阶段暂由表单传入；接入 JWT 后应从登录态读取。
    uploaded_by: Annotated[int, Form(gt=0)] = 1,
    session: AsyncSession = Depends(get_db),
) -> UploadDocumentsResponse:
    try:
        documents = await create_uploaded_documents(
            session,
            files=files,
            kb_id=kb_id,
            uploaded_by=uploaded_by,
        )
    except DocumentUploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentStorageError as exc:
        logger.exception("文档上传失败")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    items: list[UploadedDocumentResponse] = []
    for document in documents:
        task, created = await create_index_task(
            session,
            doc_id=document.id,
            task_type="INDEX",
        )
        try:
            queued = await enqueue_document_index(task.id)
        except Exception:
            # 文档和任务均已落库，队列恢复后可通过 /{doc_id}/index 重新投递。
            logger.exception("上传后的索引任务入队失败: task_id=%s", task.id)
            queued = False
        snapshot = IndexTaskSnapshot.from_model(task)
        items.append(
            UploadedDocumentResponse(
                document=DocumentResponse.from_model(document, task),
                index_task=CreateIndexResponse(
                    **IndexTaskStatusResponse.from_snapshot(snapshot).model_dump(),
                    created=created,
                    queued=queued,
                ),
            )
        )
    return UploadDocumentsResponse(items=items)


@router.get("", response_model=DocumentListResponse)
async def get_documents(
    kb_id: Annotated[int | None, Query(gt=0)] = None,
    document_status: Annotated[
        Literal["PENDING", "PROCESSING", "DONE", "FAILED", "CANCELED"] | None,
        Query(alias="status"),
    ] = None,
    keyword: Annotated[str | None, Query(max_length=255)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    session: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    total, items = await list_documents(
        session,
        kb_id=kb_id,
        status=document_status,
        keyword=keyword,
        offset=offset,
        limit=limit,
    )
    return DocumentListResponse(
        total=total,
        offset=offset,
        limit=limit,
        items=[
            DocumentResponse.from_model(item.document, item.latest_task)
            for item in items
        ],
    )


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
