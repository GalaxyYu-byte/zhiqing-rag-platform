"""文档上传、MinIO 持久化和文档列表查询服务。"""

from __future__ import annotations

import asyncio
import contextlib
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.minio import minio_client
from ..models.document import Document, IndexTask
from ..models.knowledge_base import KnowledgeBase


ALLOWED_DOCUMENT_TYPES = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".txt": "TXT",
    ".md": "MD",
    ".xlsx": "XLSX",
    ".xls": "XLS",
}


class DocumentUploadValidationError(ValueError):
    """上传参数或文件不符合约束。"""


class DocumentStorageError(RuntimeError):
    """文件无法可靠写入 MinIO。"""


@dataclass(slots=True, frozen=True)
class DocumentCatalogItem:
    document: Document
    latest_task: IndexTask | None


def _safe_file_name(value: str | None) -> str:
    """去除客户端路径，只保留可作为对象元数据展示的文件名。"""

    normalized = (value or "").strip().replace("\\", "/")
    file_name = Path(normalized).name
    if not file_name:
        raise DocumentUploadValidationError("文件名不能为空")
    if len(file_name) > 255:
        raise DocumentUploadValidationError(f"文件名超过 255 个字符: {file_name}")
    return file_name


async def _file_size(upload: UploadFile) -> int:
    """优先使用 Starlette 已统计的真实大小，缺失时再检查临时文件。"""

    if upload.size is not None:
        return int(upload.size)
    current = await asyncio.to_thread(upload.file.tell)
    await asyncio.to_thread(upload.file.seek, 0, 2)
    size = await asyncio.to_thread(upload.file.tell)
    await asyncio.to_thread(upload.file.seek, current)
    return int(size)


async def create_uploaded_documents(
    session: AsyncSession,
    *,
    files: list[UploadFile],
    kb_id: int,
    uploaded_by: int,
    object_store: Any = minio_client,
) -> list[Document]:
    """校验文件、写入 MinIO，并在同一数据库事务中创建文档记录。

    MinIO 对象使用 UUID 路径，避免同名文件互相覆盖。若任一步骤失败，会尽力
    删除本次已经写入的对象；索引任务由 API 层在文档提交后创建并投递。
    """

    if not files:
        raise DocumentUploadValidationError("至少选择一个文件")
    if kb_id <= 0 or uploaded_by <= 0:
        raise DocumentUploadValidationError("kb_id 和 uploaded_by 必须大于 0")

    knowledge_base = await session.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == kb_id,
            KnowledgeBase.is_deleted.is_(False),
        )
    )
    if knowledge_base is None:
        raise DocumentUploadValidationError(f"知识库不存在: {kb_id}")

    max_file_size = settings.max_file_size_mb * 1024 * 1024
    max_request_size = settings.max_request_size_mb * 1024 * 1024
    prepared: list[tuple[UploadFile, str, str, int, str]] = []
    total_size = 0

    # 所有文件先完成校验，避免批量请求中途因格式问题产生部分上传。
    for upload in files:
        file_name = _safe_file_name(upload.filename)
        suffix = Path(file_name).suffix.lower()
        file_type = ALLOWED_DOCUMENT_TYPES.get(suffix)
        if file_type is None:
            allowed = ", ".join(sorted(ALLOWED_DOCUMENT_TYPES))
            raise DocumentUploadValidationError(
                f"不支持的文件类型: {file_name}，允许类型: {allowed}"
            )
        size = await _file_size(upload)
        if size <= 0:
            raise DocumentUploadValidationError(f"文件内容为空: {file_name}")
        if size > max_file_size:
            raise DocumentUploadValidationError(
                f"文件超过 {settings.max_file_size_mb} MB: {file_name}"
            )
        total_size += size
        object_name = f"kb/{kb_id}/{uuid.uuid4().hex}/{file_name}"
        prepared.append((upload, file_name, file_type, size, object_name))

    if total_size > max_request_size:
        raise DocumentUploadValidationError(
            f"本次上传总大小超过 {settings.max_request_size_mb} MB"
        )

    stored_objects: list[str] = []
    documents: list[Document] = []
    try:
        for upload, file_name, file_type, size, object_name in prepared:
            await upload.seek(0)
            content_type = (
                upload.content_type
                or mimetypes.guess_type(file_name)[0]
                or "application/octet-stream"
            )
            await asyncio.to_thread(
                object_store.put_object,
                settings.minio_bucket,
                object_name,
                upload.file,
                size,
                content_type=content_type,
            )
            stored_objects.append(object_name)
            document = Document(
                kb_id=kb_id,
                file_name=file_name,
                file_type=file_type,
                file_size=size,
                minio_path=f"{settings.minio_bucket}/{object_name}",
                status="PENDING",
                chunk_count=0,
                token_count=0,
                version=1,
                uploaded_by=uploaded_by,
                is_deleted=False,
            )
            documents.append(document)

        session.add_all(documents)
        await session.commit()
        for document in documents:
            await session.refresh(document)
        return documents
    except DocumentUploadValidationError:
        raise
    except Exception as exc:
        await session.rollback()
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    object_store.remove_object,
                    settings.minio_bucket,
                    object_name,
                )
                for object_name in stored_objects
            ),
            return_exceptions=True,
        )
        raise DocumentStorageError("文档上传或记录创建失败") from exc
    finally:
        # FastAPI 最终也会关闭文件，这里仅确保服务层独立调用时及时释放句柄。
        for upload in files:
            with contextlib.suppress(Exception):
                await upload.close()


async def list_documents(
    session: AsyncSession,
    *,
    kb_id: int | None = None,
    status: str | None = None,
    keyword: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> tuple[int, list[DocumentCatalogItem]]:
    """分页查询未删除文档，并附带每个文档最新的索引任务。"""

    filters = [Document.is_deleted.is_(False)]
    if kb_id is not None:
        filters.append(Document.kb_id == kb_id)
    if status:
        filters.append(Document.status == status.upper())
    if keyword and keyword.strip():
        escaped = keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        filters.append(Document.file_name.ilike(f"%{escaped}%", escape="\\"))

    total = int(
        await session.scalar(
            select(func.count(Document.id)).where(*filters)
        )
        or 0
    )

    latest_task_id = (
        select(IndexTask.id)
        .where(IndexTask.doc_id == Document.id)
        .order_by(IndexTask.created_at.desc(), IndexTask.id.desc())
        .limit(1)
        .correlate(Document)
        .scalar_subquery()
    )
    rows = (
        await session.execute(
            select(Document, IndexTask)
            .outerjoin(IndexTask, IndexTask.id == latest_task_id)
            .where(*filters)
            .order_by(Document.uploaded_at.desc(), Document.id.desc())
            .offset(offset)
            .limit(limit)
        )
    ).all()
    return total, [
        DocumentCatalogItem(document=document, latest_task=task)
        for document, task in rows
    ]
