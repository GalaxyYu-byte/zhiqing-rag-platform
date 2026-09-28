"""文档上传和可写知识库列表 HTTP 入口。"""

from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated
from urllib.parse import quote

import psycopg
from fastapi import APIRouter, Depends, File, Form, Path, Query, Request, Response, UploadFile

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.infrastructure.object_storage import document_object_storage

from .preview import preview_content
from .repository import UploadRepository
from .schemas import (
    DocumentTaskPage,
    DocumentTaskStatus,
    KnowledgeBaseOption,
    UploadError,
    UploadIdentity,
    UploadParameters,
    UploadResult,
)
from .service import DocumentUploadService

router = APIRouter(prefix="/api", tags=["文档上传"])


def require_local_development(request: Request, settings: Settings) -> None:
    origin = request.headers.get("origin")
    if origin and origin not in settings.cors_origins:
        raise UploadError(403, "ORIGIN_FORBIDDEN", "不允许此页面来源访问上传接口")
    if (
        settings.app_env != "dev"
        or not settings.dev_upload_identity_enabled
        or settings.app_host not in {"127.0.0.1", "localhost", "::1"}
        or not request.client
        or request.client.host not in {"127.0.0.1", "::1"}
    ):
        raise UploadError(401, "AUTH_REQUIRED", "请接入正式登录鉴权；开发身份仅支持本机开发环境")


def get_upload_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Iterator[DocumentUploadService]:
    require_local_development(request, settings)
    try:
        with (
            database_connection(settings) as connection,
            document_object_storage(settings) as storage,
        ):
            yield DocumentUploadService(
                UploadRepository(connection, settings.db_schema), storage, settings
            )
    except psycopg.Error as error:
        raise UploadError(503, "DATABASE_UNAVAILABLE", "数据库暂不可用，请稍后重试") from error


ServiceDependency = Annotated[DocumentUploadService, Depends(get_upload_service)]


def get_upload_identity(service: ServiceDependency) -> UploadIdentity:
    return service.repository.identity(
        service.settings.dev_upload_user_email,
        service.settings.dev_upload_tenant_code,
    )


IdentityDependency = Annotated[UploadIdentity, Depends(get_upload_identity)]


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseOption])
def list_knowledge_bases(service: ServiceDependency, identity: IdentityDependency):
    """返回当前服务端确认身份有 WRITE/ADMIN 授权且处于 ACTIVE 的知识库。"""
    return service.repository.writable_bases(identity)


@router.post("/documents/upload", response_model=UploadResult, status_code=202)
def upload_document(
    service: ServiceDependency,
    identity: IdentityDependency,
    file: Annotated[UploadFile, File(description="原始文档，默认上限 50 MiB")],
    knowledge_base_id: Annotated[int, Form(gt=0)],
):
    """保存原文件并在同一事务创建持久化异步索引任务，返回后由独立 Worker 处理。"""
    try:
        parameters = UploadParameters(
            knowledge_base_id=knowledge_base_id,
            chunk_size=service.settings.upload_chunk_size,
            chunk_overlap=service.settings.upload_chunk_overlap,
        )
        return service.upload(
            file.file, file.filename or "", file.content_type, identity, parameters
        )
    finally:
        file.file.close()


@router.get("/document-tasks", response_model=list[DocumentTaskStatus])
def list_document_tasks(service: ServiceDependency, identity: IdentityDependency):
    return service.repository.list_tasks(identity)


@router.get("/document-upload-settings")
def document_upload_settings(service: ServiceDependency, identity: IdentityDependency):
    return {
        "chunk_size": service.settings.upload_chunk_size,
        "chunk_overlap": service.settings.upload_chunk_overlap,
    }


@router.get("/document-tasks/page", response_model=DocumentTaskPage)
def document_task_page(
    service: ServiceDependency,
    identity: IdentityDependency,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    knowledge_base_id: Annotated[int | None, Query(gt=0)] = None,
    search: Annotated[str, Query(max_length=500)] = "",
    status: Annotated[
        str, Query(pattern="^(all|pending|done|processing|error|cancelled)$")
    ] = "all",
    date_from: date | None = None,
    date_to: date | None = None,
):
    if date_from and date_to and date_from > date_to:
        raise UploadError(422, "INVALID_DATE_RANGE", "开始日期不能晚于结束日期")
    local_timezone = timezone(timedelta(hours=8))
    start = datetime.combine(date_from, time.min, local_timezone) if date_from else None
    end = (
        datetime.combine(date_to, time.min, local_timezone) + timedelta(days=1) if date_to else None
    )
    return service.repository.paginate_tasks(
        identity,
        offset=offset,
        limit=limit,
        knowledge_base_id=knowledge_base_id,
        search=search.strip(),
        status=status,
        start=start,
        end=end,
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: Annotated[int, Path(gt=0)],
    service: ServiceDependency,
    identity: IdentityDependency,
):
    """停止后台任务，永久删除原文件、所有修订、分块、向量及处理历史。"""
    service.delete_document(identity, document_id)
    return Response(status_code=204)


@router.get("/documents/{document_id}/content")
def document_content(
    document_id: Annotated[int, Path(gt=0)],
    revision_id: Annotated[int, Query(gt=0)],
    service: ServiceDependency,
    identity: IdentityDependency,
):
    row, content = service.original_document(identity, document_id, revision_id)
    encoded_filename = quote(row["original_filename"])
    return Response(
        content,
        media_type=row["mime_type"],
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/documents/{document_id}/preview")
def document_preview(
    document_id: Annotated[int, Path(gt=0)],
    revision_id: Annotated[int, Query(gt=0)],
    service: ServiceDependency,
    identity: IdentityDependency,
    response: Response,
):
    row, content = service.original_document(identity, document_id, revision_id)
    response.headers["Cache-Control"] = "no-store"
    return preview_content(content, row["original_filename"], service.settings.upload_max_file_size)


@router.post("/documents/preview")
def local_document_preview(
    service: ServiceDependency,
    identity: IdentityDependency,
    file: Annotated[UploadFile, File()],
    response: Response,
):
    try:
        content = file.file.read(service.settings.upload_max_file_size + 1)
        if len(content) > service.settings.upload_max_file_size:
            raise UploadError(413, "FILE_TOO_LARGE", "文件超过预览大小限制")
        response.headers["Cache-Control"] = "no-store"
        return preview_content(content, file.filename or "", service.settings.upload_max_file_size)
    finally:
        file.file.close()


@router.get("/document-tasks/{task_id}", response_model=DocumentTaskStatus)
def document_task_status(task_id: int, service: ServiceDependency, identity: IdentityDependency):
    return service.repository.task_status(identity, task_id)


@router.post("/document-tasks/{task_id}/retry", response_model=DocumentTaskStatus)
def retry_document_task(task_id: int, service: ServiceDependency, identity: IdentityDependency):
    return service.repository.retry_task(identity, task_id)
