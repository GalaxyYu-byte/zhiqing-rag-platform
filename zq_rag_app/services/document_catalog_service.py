"""文档上传、MinIO 持久化和文档列表查询服务。"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
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
from ..models.document import Document, DocumentVersion, IndexTask
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


class DocumentUpdateConflictError(RuntimeError):
    """文档版本冲突、内容未变化或已有更新任务。"""


class DocumentVersionNotFoundError(LookupError):
    """指定的文档历史版本不存在。"""


class DocumentRestoreError(RuntimeError):
    """历史版本恢复任务无法可靠创建。"""


@dataclass(slots=True, frozen=True)
class DocumentCatalogItem:
    document: Document
    latest_task: IndexTask | None


@dataclass(slots=True, frozen=True)
class DocumentUpdateResult:
    document: Document
    version: DocumentVersion
    task: IndexTask


async def list_document_versions(
    session: AsyncSession,
    *,
    doc_id: int,
) -> list[DocumentVersion]:
    """按版本号倒序返回一个逻辑文档的完整版本历史。"""

    return list(
        (
            await session.scalars(
                select(DocumentVersion)
                .where(DocumentVersion.doc_id == doc_id)
                .order_by(DocumentVersion.version.desc())
            )
        ).all()
    )


async def create_document_restore(
    session: AsyncSession,
    *,
    doc_id: int,
    source_version: int,
    expected_current_version: int,
    restored_by: int,
) -> DocumentUpdateResult:
    """基于 READY 历史文件创建 RESTORE 候选版本，正式版本保持不变。"""

    try:
        document = await session.scalar(
            select(Document)
            .where(Document.id == doc_id, Document.is_deleted.is_(False))
            .with_for_update()
        )
        if document is None:
            raise DocumentVersionNotFoundError(f"文档不存在: {doc_id}")
        if document.version != expected_current_version:
            raise DocumentUpdateConflictError(
                f"文档版本已变化，当前版本为 V{document.version}"
            )
        if source_version == document.version:
            raise DocumentUpdateConflictError(
                f"V{source_version} 已经是当前正式版本"
            )

        active_task = await session.scalar(
            select(IndexTask.id).where(
                IndexTask.doc_id == doc_id,
                IndexTask.status.in_(["PENDING", "PROCESSING"]),
            )
        )
        if active_task is not None:
            raise DocumentUpdateConflictError("该文档已有正在执行的索引任务")

        source = await session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.doc_id == doc_id,
                DocumentVersion.version == source_version,
            )
        )
        if source is None:
            raise DocumentVersionNotFoundError(
                f"文档 V{source_version} 不存在"
            )
        if source.status != "READY":
            raise DocumentUpdateConflictError(
                f"只有 READY 历史版本可以恢复，V{source_version} 当前为 {source.status}"
            )

        latest_version = int(
            await session.scalar(
                select(func.max(DocumentVersion.version)).where(
                    DocumentVersion.doc_id == doc_id
                )
            )
            or document.version
        )
        target_version = latest_version + 1
        candidate = DocumentVersion(
            doc_id=document.id,
            version=target_version,
            file_name=source.file_name,
            file_type=source.file_type,
            file_size=source.file_size,
            file_hash=source.file_hash,
            minio_path=source.minio_path,
            operation_type="RESTORE",
            source_version=source.version,
            status="PENDING",
            uploaded_by=restored_by,
        )
        task = IndexTask(
            doc_id=document.id,
            doc_version=target_version,
            task_type="RESTORE",
            status="PENDING",
            stage="PENDING",
            progress_percent=0,
            total_chunks=0,
            embedded_chunks=0,
            persisted_chunks=0,
            cache_hit_chunks=0,
            total_tokens=0,
            retry_count=0,
            max_retry=settings.index_task_max_retry,
        )
        session.add_all([candidate, task])
        await session.commit()
        await session.refresh(candidate)
        await session.refresh(task)
        return DocumentUpdateResult(
            document=document,
            version=candidate,
            task=task,
        )
    except (DocumentVersionNotFoundError, DocumentUpdateConflictError):
        raise
    except Exception as exc:
        await session.rollback()
        raise DocumentRestoreError("历史版本恢复任务创建失败") from exc


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


async def _file_sha256(upload: UploadFile) -> str:
    digest = hashlib.sha256()
    await upload.seek(0)
    while chunk := await upload.read(1024 * 1024):
        digest.update(chunk)
    await upload.seek(0)
    return digest.hexdigest()


async def create_document_update(
    session: AsyncSession,
    *,
    doc_id: int,
    file: UploadFile,
    expected_version: int,
    uploaded_by: int,
    object_store: Any = minio_client,
) -> DocumentUpdateResult:
    """保存候选文件版本并创建 UPDATE 任务，正式版本保持不变。"""

    object_name: str | None = None
    try:
        document = await session.scalar(
            select(Document)
            .where(Document.id == doc_id, Document.is_deleted.is_(False))
            .with_for_update()
        )
        if document is None:
            raise DocumentUploadValidationError(f"文档不存在: {doc_id}")
        if document.version != expected_version:
            raise DocumentUpdateConflictError(
                f"文档版本已变化，当前版本为 V{document.version}"
            )

        active_task = await session.scalar(
            select(IndexTask.id).where(
                IndexTask.doc_id == doc_id,
                IndexTask.status.in_(["PENDING", "PROCESSING"]),
            )
        )
        if active_task is not None:
            raise DocumentUpdateConflictError("该文档已有正在执行的索引任务")

        file_name = _safe_file_name(file.filename)
        file_type = ALLOWED_DOCUMENT_TYPES.get(Path(file_name).suffix.lower())
        if file_type is None:
            allowed = ", ".join(sorted(ALLOWED_DOCUMENT_TYPES))
            raise DocumentUploadValidationError(
                f"不支持的文件类型: {file_name}，允许类型: {allowed}"
            )
        file_size = await _file_size(file)
        if file_size <= 0:
            raise DocumentUploadValidationError(f"文件内容为空: {file_name}")
        if file_size > settings.max_file_size_mb * 1024 * 1024:
            raise DocumentUploadValidationError(
                f"文件超过 {settings.max_file_size_mb} MB: {file_name}"
            )
        file_hash = await _file_sha256(file)

        current_version = await session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.doc_id == doc_id,
                DocumentVersion.version == document.version,
            )
        )
        if current_version is not None and current_version.file_hash == file_hash:
            raise DocumentUpdateConflictError("上传文件内容与当前版本完全一致")

        latest_version = int(
            await session.scalar(
                select(func.max(DocumentVersion.version)).where(
                    DocumentVersion.doc_id == doc_id
                )
            )
            or document.version
        )
        target_version = latest_version + 1
        object_name = (
            f"kb/{document.kb_id}/{document.id}/v{target_version}/"
            f"{uuid.uuid4().hex}/{file_name}"
        )
        content_type = (
            file.content_type
            or mimetypes.guess_type(file_name)[0]
            or "application/octet-stream"
        )
        await file.seek(0)
        await asyncio.to_thread(
            object_store.put_object,
            settings.minio_bucket,
            object_name,
            file.file,
            file_size,
            content_type=content_type,
        )

        version = DocumentVersion(
            doc_id=document.id,
            version=target_version,
            file_name=file_name,
            file_type=file_type,
            file_size=file_size,
            file_hash=file_hash,
            minio_path=f"{settings.minio_bucket}/{object_name}",
            operation_type="UPDATE",
            status="PENDING",
            uploaded_by=uploaded_by,
        )
        task = IndexTask(
            doc_id=document.id,
            doc_version=target_version,
            task_type="UPDATE",
            status="PENDING",
            stage="PENDING",
            progress_percent=0,
            total_chunks=0,
            embedded_chunks=0,
            persisted_chunks=0,
            cache_hit_chunks=0,
            total_tokens=0,
            retry_count=0,
            max_retry=settings.index_task_max_retry,
        )
        session.add_all([version, task])
        await session.commit()
        await session.refresh(version)
        await session.refresh(task)
        return DocumentUpdateResult(document=document, version=version, task=task)
    except (DocumentUploadValidationError, DocumentUpdateConflictError):
        raise
    except Exception as exc:
        await session.rollback()
        if object_name is not None:
            await asyncio.gather(
                asyncio.to_thread(
                    object_store.remove_object,
                    settings.minio_bucket,
                    object_name,
                ),
                return_exceptions=True,
            )
        raise DocumentStorageError("新版文件上传或任务创建失败") from exc
    finally:
        with contextlib.suppress(Exception):
            await file.close()


async def create_uploaded_documents(
    session: AsyncSession,
    *,
    files: list[UploadFile],
    kb_id: int,
    uploaded_by: int,
    department_id: str | None = None,
    confidentiality: str = "机密",
    business_status: str = "生效",
    document_code: str | None = None,
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
    prepared: list[tuple[UploadFile, str, str, int, str, str]] = []
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
        file_hash = await _file_sha256(upload)
        prepared.append((upload, file_name, file_type, size, object_name, file_hash))

    if total_size > max_request_size:
        raise DocumentUploadValidationError(
            f"本次上传总大小超过 {settings.max_request_size_mb} MB"
        )

    stored_objects: list[str] = []
    documents: list[Document] = []
    document_hashes: list[str] = []
    try:
        for upload, file_name, file_type, size, object_name, file_hash in prepared:
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
                document_code=document_code,
                department_id=department_id,
                confidentiality=confidentiality,
                business_status=business_status,
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
            document_hashes.append(file_hash)

        session.add_all(documents)
        await session.flush()
        session.add_all(
            [
                DocumentVersion(
                    doc_id=document.id,
                    version=1,
                    file_name=document.file_name,
                    file_type=document.file_type,
                    file_size=document.file_size,
                    file_hash=file_hash,
                    minio_path=document.minio_path,
                    operation_type="UPLOAD",
                    status="PENDING",
                    uploaded_by=uploaded_by,
                )
                for document, file_hash in zip(
                    documents, document_hashes, strict=True
                )
            ]
        )
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
