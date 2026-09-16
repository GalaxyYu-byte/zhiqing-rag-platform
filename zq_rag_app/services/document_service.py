"""可追踪、可重试且幂等的文档索引任务。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from minio.error import S3Error
from redis.exceptions import RedisError
from sqlalchemy import and_, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..core.config import settings
from ..core.database import SessionLocal
from ..core.executor import submit_index_task
from ..core.minio import minio_client
from ..core.task_queue import enqueue_graph_index
from ..document_processing.chunking import ChunkingConfig, TextChunk
from ..document_processing.chunking import chunk_cleaned_document
from ..document_processing.cleaning import clean_parsed_blocks
from ..models.document import DocChunk, Document, DocumentVersion, IndexTask
from ..models.graph import GraphTask
from ..utils.document_parser import parse_document
from .embedding_service import (
    EmbeddingBatchResult,
    EmbeddingProgress,
    EmbeddingService,
    get_embedding_service,
    is_retryable_embedding_exception,
)


logger = logging.getLogger(__name__)


class IndexStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class IndexStage(StrEnum):
    PENDING = "PENDING"
    RETRY_WAIT = "RETRY_WAIT"
    PARSING = "PARSING"
    CLEANING = "CLEANING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    PERSISTING = "PERSISTING"
    DONE = "DONE"
    FAILED = "FAILED"


class IndexTaskNotFoundError(LookupError):
    """索引任务或其文档不存在。"""


class IndexTaskLeaseLostError(RuntimeError):
    """任务已经被另一节点接管，当前 Worker 必须停止写入。"""


class RetryableIndexTaskError(RuntimeError):
    """任务已回到 PENDING，队列应在 delay_seconds 后再次投递。"""

    def __init__(self, message: str, *, delay_seconds: float) -> None:
        super().__init__(message)
        self.delay_seconds = delay_seconds


@dataclass(slots=True, frozen=True)
class IndexTaskSnapshot:
    id: int
    doc_id: int
    doc_version: int
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
    def from_model(cls, task: IndexTask) -> "IndexTaskSnapshot":
        return cls(
            id=task.id,
            doc_id=task.doc_id,
            doc_version=task.doc_version,
            task_type=task.task_type,
            status=task.status,
            stage=task.stage,
            progress_percent=task.progress_percent,
            total_chunks=task.total_chunks,
            embedded_chunks=task.embedded_chunks,
            persisted_chunks=task.persisted_chunks,
            cache_hit_chunks=task.cache_hit_chunks,
            total_tokens=task.total_tokens,
            retry_count=task.retry_count,
            max_retry=task.max_retry,
            error_msg=task.error_msg,
            worker_id=task.worker_id,
            heartbeat_at=task.heartbeat_at,
            lease_expires_at=task.lease_expires_at,
            created_at=task.created_at,
            started_at=task.started_at,
            finished_at=task.finished_at,
        )


@dataclass(slots=True, frozen=True)
class _DocumentSnapshot:
    id: int
    kb_id: int
    file_name: str
    minio_path: str
    version: int


def _utcnow_naive() -> datetime:
    """数据库当前使用无时区 TIMESTAMP，统一写入 UTC 的 naive 时间。"""

    return datetime.now(UTC).replace(tzinfo=None)


def is_retryable_index_exception(exc: BaseException) -> bool:
    """判断任务级异常是否可能通过稍后重试恢复。"""

    if isinstance(exc, FileNotFoundError):
        return False
    if is_retryable_embedding_exception(exc):
        return True
    if isinstance(exc, (OperationalError, InterfaceError, RedisError)):
        return True
    if isinstance(exc, S3Error):
        return exc.code in {
            "InternalError",
            "RequestTimeout",
            "ServiceUnavailable",
            "SlowDown",
        }
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


async def create_index_task(
    session: AsyncSession,
    *,
    doc_id: int,
    task_type: str = "INDEX",
) -> tuple[IndexTask, bool]:
    """创建索引任务；已有活跃任务时直接复用，保证接口调用幂等。"""

    normalized_type = task_type.upper()
    if normalized_type not in {"INDEX", "REINDEX"}:
        raise ValueError("task_type 只能是 INDEX 或 REINDEX")

    document = await session.scalar(
        select(Document)
        .where(Document.id == doc_id, Document.is_deleted.is_(False))
        .with_for_update()
    )
    if document is None:
        raise IndexTaskNotFoundError(f"文档不存在: {doc_id}")

    active = await session.scalar(
        select(IndexTask)
        .where(
            IndexTask.doc_id == doc_id,
            IndexTask.status.in_(
                [IndexStatus.PENDING.value, IndexStatus.PROCESSING.value]
            ),
        )
        .order_by(IndexTask.created_at.desc())
        .limit(1)
    )
    if active is not None:
        return active, False

    current_version = await session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.doc_id == document.id,
            DocumentVersion.version == document.version,
        )
    )
    if current_version is None:
        current_version = DocumentVersion(
            doc_id=document.id,
            version=document.version,
            file_name=document.file_name,
            file_type=document.file_type,
            file_size=document.file_size,
            minio_path=document.minio_path,
            operation_type="UPLOAD",
            status=(
                "READY"
                if document.status == IndexStatus.DONE.value
                else IndexStatus.PENDING.value
            ),
            uploaded_by=document.uploaded_by,
        )
        session.add(current_version)

    target_version = document.version
    if normalized_type == "REINDEX":
        latest_version = int(
            await session.scalar(
                select(func.max(DocumentVersion.version)).where(
                    DocumentVersion.doc_id == document.id
                )
            )
            or document.version
        )
        target_version = latest_version + 1
        session.add(
            DocumentVersion(
                doc_id=document.id,
                version=target_version,
                file_name=document.file_name,
                file_type=document.file_type,
                file_size=document.file_size,
                file_hash=current_version.file_hash,
                minio_path=document.minio_path,
                operation_type="REINDEX",
                status=IndexStatus.PENDING.value,
                uploaded_by=document.uploaded_by,
            )
        )
    elif document.status != IndexStatus.DONE.value:
        document.status = IndexStatus.PENDING.value
        document.error_msg = None

    task = IndexTask(
        doc_id=doc_id,
        doc_version=target_version,
        task_type=normalized_type,
        status=IndexStatus.PENDING.value,
        stage=IndexStage.PENDING.value,
        max_retry=settings.index_task_max_retry,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task, True


async def get_index_task_snapshot(
    session: AsyncSession,
    task_id: int,
) -> IndexTaskSnapshot | None:
    task = await session.get(IndexTask, task_id)
    return IndexTaskSnapshot.from_model(task) if task is not None else None


async def get_latest_document_index_snapshot(
    session: AsyncSession,
    doc_id: int,
) -> IndexTaskSnapshot | None:
    task = await session.scalar(
        select(IndexTask)
        .where(IndexTask.doc_id == doc_id)
        .order_by(IndexTask.created_at.desc())
        .limit(1)
    )
    return IndexTaskSnapshot.from_model(task) if task is not None else None


class DocumentIndexService:
    """编排解析、清洗、分块、向量化和幂等持久化。"""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        embedding_service: EmbeddingService | None = None,
        object_store: Any = minio_client,
        worker_id: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.embedding_service = embedding_service or get_embedding_service()
        self.object_store = object_store
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"

    async def run(
        self,
        task_id: int,
        *,
        source_path: str | Path | None = None,
    ) -> bool:
        """执行一次任务；成功返回 True，已完成或被其他节点占用返回 False。"""

        document = await self._claim(task_id)
        if document is None:
            return False

        heartbeat = asyncio.create_task(self._heartbeat_loop(task_id))
        try:
            graph_task_id = await self._run_pipeline(
                task_id, document, source_path
            )
            if graph_task_id is not None:
                try:
                    await enqueue_graph_index(graph_task_id)
                except Exception:
                    # 图任务已经持久化，独立队列恢复后可再次幂等投递。
                    logger.exception(
                        "图任务入队失败: graph_task_id=%s",
                        graph_task_id,
                    )
            return True
        except IndexTaskLeaseLostError:
            logger.warning("索引任务租约已转移，停止当前执行: task_id=%s", task_id)
            return False
        except Exception as exc:
            should_retry, delay = await self._record_failure(task_id, exc)
            if should_retry:
                raise RetryableIndexTaskError(
                    str(exc), delay_seconds=delay
                ) from exc
            raise
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _run_pipeline(
        self,
        task_id: int,
        document: _DocumentSnapshot,
        source_path: str | Path | None,
    ) -> int | None:
        async with self._materialize_document(document, source_path) as local_path:
            await self._set_stage(task_id, IndexStage.PARSING, 5)
            parsed = await asyncio.to_thread(parse_document, local_path)

            await self._set_stage(task_id, IndexStage.CLEANING, 15)
            cleaned = await asyncio.to_thread(clean_parsed_blocks, parsed)

            await self._set_stage(task_id, IndexStage.CHUNKING, 25)
            chunked = await asyncio.to_thread(
                chunk_cleaned_document,
                cleaned,
                ChunkingConfig(
                    chunk_size=settings.rag_chunk_size,
                    chunk_overlap=settings.rag_chunk_overlap,
                ),
            )
            chunks = chunked.chunks
            await self._update_progress(
                task_id,
                stage=IndexStage.EMBEDDING,
                progress_percent=30,
                total_chunks=len(chunks),
                embedded_chunks=0,
                persisted_chunks=0,
                cache_hit_chunks=0,
                total_tokens=chunked.report.total_tokens,
            )

            async def on_embedding_progress(progress: EmbeddingProgress) -> None:
                percent = 85 if progress.total == 0 else (
                    30 + int(55 * progress.completed / progress.total)
                )
                await self._update_progress(
                    task_id,
                    stage=IndexStage.EMBEDDING,
                    progress_percent=percent,
                    embedded_chunks=progress.completed,
                    cache_hit_chunks=progress.cache_hits,
                )

            embedding_result = await self.embedding_service.embed_many(
                [chunk.embedding_content for chunk in chunks],
                progress_callback=on_embedding_progress,
            )

            await self._update_progress(
                task_id,
                stage=IndexStage.PERSISTING,
                progress_percent=88,
                embedded_chunks=len(chunks),
                cache_hit_chunks=embedding_result.cache_hits,
            )
            await self._persist_chunks(
                task_id=task_id,
                document=document,
                chunks=chunks,
                embeddings=embedding_result,
            )
            return await self._complete_task(
                task_id=task_id,
                document=document,
                chunk_count=len(chunks),
                token_count=chunked.report.total_tokens,
            )

    async def _claim(self, task_id: int) -> _DocumentSnapshot | None:
        now = _utcnow_naive()
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    select(IndexTask, Document, DocumentVersion)
                    .join(Document, Document.id == IndexTask.doc_id)
                    .join(
                        DocumentVersion,
                        and_(
                            DocumentVersion.doc_id == IndexTask.doc_id,
                            DocumentVersion.version == IndexTask.doc_version,
                        ),
                    )
                    .where(IndexTask.id == task_id)
                    .with_for_update()
                )
            ).first()
            if row is None:
                raise IndexTaskNotFoundError(f"索引任务不存在: {task_id}")
            task, document, version = row
            if task.status in {
                IndexStatus.DONE.value,
                IndexStatus.FAILED.value,
                IndexStatus.CANCELED.value,
            }:
                return None
            if (
                task.status == IndexStatus.PROCESSING.value
                and task.lease_expires_at is not None
                and task.lease_expires_at > now
            ):
                return None
            if document.is_deleted:
                raise IndexTaskNotFoundError(f"文档已删除: {document.id}")

            task.status = IndexStatus.PROCESSING.value
            task.stage = IndexStage.PARSING.value
            task.progress_percent = 1
            task.total_chunks = 0
            task.embedded_chunks = 0
            task.persisted_chunks = 0
            task.cache_hit_chunks = 0
            task.total_tokens = 0
            task.error_msg = None
            task.worker_id = self.worker_id
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(
                seconds=settings.index_lease_seconds
            )
            task.started_at = task.started_at or now
            task.finished_at = None
            task.updated_at = now
            version.status = IndexStatus.PROCESSING.value
            version.error_msg = None
            if task.doc_version == document.version and document.status != IndexStatus.DONE.value:
                document.status = IndexStatus.PROCESSING.value
                document.error_msg = None
            await session.commit()
            return _DocumentSnapshot(
                id=document.id,
                kb_id=document.kb_id,
                file_name=version.file_name,
                minio_path=version.minio_path,
                version=version.version,
            )

    async def _set_stage(
        self,
        task_id: int,
        stage: IndexStage,
        progress_percent: int,
    ) -> None:
        await self._update_progress(
            task_id,
            stage=stage,
            progress_percent=progress_percent,
        )

    async def _update_progress(
        self,
        task_id: int,
        *,
        stage: IndexStage,
        progress_percent: int,
        total_chunks: int | None = None,
        embedded_chunks: int | None = None,
        persisted_chunks: int | None = None,
        cache_hit_chunks: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        now = _utcnow_naive()
        values: dict[str, Any] = {
            "stage": stage.value,
            "progress_percent": max(0, min(100, progress_percent)),
            "heartbeat_at": now,
            "lease_expires_at": now
            + timedelta(seconds=settings.index_lease_seconds),
            "updated_at": now,
        }
        for name, value in {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "persisted_chunks": persisted_chunks,
            "cache_hit_chunks": cache_hit_chunks,
            "total_tokens": total_tokens,
        }.items():
            if value is not None:
                values[name] = value

        async with self.session_factory() as session:
            result = await session.execute(
                update(IndexTask)
                .where(
                    IndexTask.id == task_id,
                    IndexTask.status == IndexStatus.PROCESSING.value,
                    IndexTask.worker_id == self.worker_id,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                await session.rollback()
                raise IndexTaskLeaseLostError(
                    f"索引任务不再归当前 Worker 所有: {task_id}"
                )
            await session.commit()

    async def _heartbeat_loop(self, task_id: int) -> None:
        while True:
            await asyncio.sleep(settings.index_heartbeat_interval_seconds)
            try:
                now = _utcnow_naive()
                async with self.session_factory() as session:
                    result = await session.execute(
                        update(IndexTask)
                        .where(
                            IndexTask.id == task_id,
                            IndexTask.status == IndexStatus.PROCESSING.value,
                            IndexTask.worker_id == self.worker_id,
                        )
                        .values(
                            heartbeat_at=now,
                            lease_expires_at=now
                            + timedelta(seconds=settings.index_lease_seconds),
                            updated_at=now,
                        )
                    )
                    await session.commit()
                    if result.rowcount != 1:
                        return
            except Exception:
                logger.exception("更新索引任务心跳失败: task_id=%s", task_id)

    async def _persist_chunks(
        self,
        *,
        task_id: int,
        document: _DocumentSnapshot,
        chunks: list[TextChunk],
        embeddings: EmbeddingBatchResult,
    ) -> None:
        batch_size = settings.index_upsert_batch_size
        for start in range(0, len(chunks), batch_size):
            chunk_batch = chunks[start : start + batch_size]
            vector_batch = embeddings.vectors[start : start + batch_size]
            rows = [
                {
                    "doc_id": document.id,
                    "kb_id": document.kb_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "embedding": vector,
                    "page_num": chunk.page_num,
                    "section_title": chunk.section_title,
                    "token_count": chunk.token_count,
                    "doc_version": document.version,
                }
                for chunk, vector in zip(
                    chunk_batch, vector_batch, strict=True
                )
            ]
            persisted = start + len(rows)
            percent = 88 + int(10 * persisted / max(1, len(chunks)))
            await self._upsert_batch(
                task_id=task_id,
                rows=rows,
                persisted_chunks=persisted,
                progress_percent=percent,
            )

    async def _upsert_batch(
        self,
        *,
        task_id: int,
        rows: list[dict[str, Any]],
        persisted_chunks: int,
        progress_percent: int,
    ) -> None:
        """新事务重试数据库 IO；提交结果未知时重复执行仍由 Upsert 保证幂等。"""

        attempts = max(1, settings.index_db_retry_attempts)
        for attempt in range(1, attempts + 1):
            try:
                async with self.session_factory() as session:
                    if rows:
                        statement = pg_insert(DocChunk).values(rows)
                        excluded = statement.excluded
                        statement = statement.on_conflict_do_update(
                            index_elements=[
                                DocChunk.doc_id,
                                DocChunk.doc_version,
                                DocChunk.chunk_index,
                            ],
                            set_={
                                "kb_id": excluded.kb_id,
                                "content": excluded.content,
                                "embedding": excluded.embedding,
                                "page_num": excluded.page_num,
                                "section_title": excluded.section_title,
                                "token_count": excluded.token_count,
                            },
                        )
                        await session.execute(statement)

                    now = _utcnow_naive()
                    progress_result = await session.execute(
                        update(IndexTask)
                        .where(
                            IndexTask.id == task_id,
                            IndexTask.status == IndexStatus.PROCESSING.value,
                            IndexTask.worker_id == self.worker_id,
                        )
                        .values(
                            stage=IndexStage.PERSISTING.value,
                            persisted_chunks=persisted_chunks,
                            progress_percent=min(98, progress_percent),
                            heartbeat_at=now,
                            lease_expires_at=now
                            + timedelta(seconds=settings.index_lease_seconds),
                            updated_at=now,
                        )
                    )
                    if progress_result.rowcount != 1:
                        raise IndexTaskLeaseLostError(
                            f"持久化时任务租约已丢失: {task_id}"
                        )
                    await session.commit()
                return
            except (OperationalError, InterfaceError):
                if attempt >= attempts:
                    raise
                delay = min(
                    settings.embedding_retry_max_wait_seconds,
                    settings.embedding_retry_min_wait_seconds
                    * (2 ** (attempt - 1)),
                )
                await asyncio.sleep(delay)

    async def _complete_task(
        self,
        *,
        task_id: int,
        document: _DocumentSnapshot,
        chunk_count: int,
        token_count: int,
    ) -> int | None:
        now = _utcnow_naive()
        async with self.session_factory() as session:
            task = await session.get(IndexTask, task_id, with_for_update=True)
            db_document = await session.get(
                Document, document.id, with_for_update=True
            )
            db_version = await session.scalar(
                select(DocumentVersion)
                .where(
                    DocumentVersion.doc_id == document.id,
                    DocumentVersion.version == document.version,
                )
                .with_for_update()
            )
            if (
                task is None
                or task.status != IndexStatus.PROCESSING.value
                or task.worker_id != self.worker_id
            ):
                raise IndexTaskLeaseLostError(
                    f"完成任务时租约已丢失: {task_id}"
                )
            if (
                db_document is None
                or db_version is None
                or task.doc_version != document.version
                or db_document.version > document.version
            ):
                raise IndexTaskLeaseLostError(
                    f"文档版本已经变化: {document.id}"
                )

            # 新版本完整可用后再清理旧版本，重试过程中旧索引始终可用。
            await session.execute(
                delete(DocChunk).where(
                    DocChunk.doc_id == document.id,
                    DocChunk.doc_version != document.version,
                )
            )
            task.status = IndexStatus.DONE.value
            task.stage = IndexStage.DONE.value
            task.progress_percent = 100
            task.embedded_chunks = chunk_count
            task.persisted_chunks = chunk_count
            task.finished_at = now
            task.heartbeat_at = now
            task.lease_expires_at = None
            task.updated_at = now
            db_version.status = "READY"
            db_version.indexed_at = now
            db_version.error_msg = None
            db_document.version = db_version.version
            db_document.file_name = db_version.file_name
            db_document.file_type = db_version.file_type
            db_document.file_size = db_version.file_size
            db_document.minio_path = db_version.minio_path
            db_document.uploaded_by = db_version.uploaded_by
            db_document.status = IndexStatus.DONE.value
            db_document.error_msg = None
            db_document.chunk_count = chunk_count
            db_document.token_count = token_count
            db_document.indexed_at = now

            graph_task_id: int | None = None
            if settings.graph_extraction_enabled:
                graph_task = await session.scalar(
                    select(GraphTask).where(
                        GraphTask.doc_id == document.id,
                        GraphTask.doc_version == document.version,
                    )
                )
                if graph_task is None:
                    await session.execute(
                        update(GraphTask)
                        .where(
                            GraphTask.doc_id == document.id,
                            GraphTask.status.in_(["PENDING", "PROCESSING"]),
                        )
                        .values(
                            status="CANCELED",
                            stage="CANCELED",
                            error_msg="文档已有更新版本",
                            finished_at=now,
                            lease_expires_at=None,
                            updated_at=now,
                        )
                    )
                    graph_task = GraphTask(
                        doc_id=document.id,
                        kb_id=document.kb_id,
                        doc_version=document.version,
                        extractor_version=settings.graph_extractor_version,
                        status="PENDING",
                        stage="PENDING",
                        max_retry=settings.graph_task_max_retry,
                    )
                    session.add(graph_task)
                    await session.flush()
                if graph_task.status == "PENDING":
                    graph_task_id = graph_task.id
            await session.commit()
            return graph_task_id

    async def _record_failure(
        self,
        task_id: int,
        exc: Exception,
    ) -> tuple[bool, float]:
        retryable = is_retryable_index_exception(exc)
        now = _utcnow_naive()
        error = f"{type(exc).__name__}: {exc}"
        async with self.session_factory() as session:
            task = await session.get(IndexTask, task_id, with_for_update=True)
            if task is None or task.worker_id != self.worker_id:
                return False, 0.0
            document = await session.get(Document, task.doc_id)
            version = await session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.doc_id == task.doc_id,
                    DocumentVersion.version == task.doc_version,
                )
            )
            should_retry = retryable and task.retry_count < task.max_retry
            if should_retry:
                delay = min(
                    60.0,
                    settings.index_task_retry_base_seconds
                    * (2**task.retry_count),
                )
                task.retry_count += 1
                task.status = IndexStatus.PENDING.value
                task.stage = IndexStage.RETRY_WAIT.value
                task.error_msg = error
                task.worker_id = None
                task.lease_expires_at = None
                task.updated_at = now
                if document is not None:
                    if task.doc_version == document.version and document.status != IndexStatus.DONE.value:
                        document.status = IndexStatus.PENDING.value
                        document.error_msg = error
                if version is not None:
                    version.status = IndexStatus.PENDING.value
                    version.error_msg = error
            else:
                delay = 0.0
                task.status = IndexStatus.FAILED.value
                task.stage = IndexStage.FAILED.value
                task.error_msg = error
                task.finished_at = now
                task.heartbeat_at = now
                task.lease_expires_at = None
                task.updated_at = now
                if document is not None:
                    if task.doc_version == document.version and document.status != IndexStatus.DONE.value:
                        document.status = IndexStatus.FAILED.value
                        document.error_msg = error
                if version is not None:
                    version.status = IndexStatus.FAILED.value
                    version.error_msg = error
                await session.execute(
                    delete(DocChunk).where(
                        DocChunk.doc_id == task.doc_id,
                        DocChunk.doc_version == task.doc_version,
                    )
                )
            await session.commit()
        return should_retry, delay

    @asynccontextmanager
    async def _materialize_document(
        self,
        document: _DocumentSnapshot,
        source_path: str | Path | None,
    ) -> AsyncIterator[Path]:
        if source_path is not None:
            path = Path(source_path)
            if not path.is_file():
                raise FileNotFoundError(f"待索引文档不存在: {path}")
            yield path
            return

        suffix = Path(document.file_name).suffix
        with tempfile.TemporaryDirectory(prefix="zq-rag-index-") as temp_dir:
            local_path = Path(temp_dir) / f"document{suffix}"
            bucket, object_name = self._resolve_minio_location(
                document.minio_path
            )
            await asyncio.to_thread(
                self.object_store.fget_object,
                bucket,
                object_name,
                str(local_path),
            )
            yield local_path

    @staticmethod
    def _resolve_minio_location(path: str) -> tuple[str, str]:
        normalized = path.strip().lstrip("/")
        if normalized.startswith("s3://"):
            normalized = normalized[5:]
            bucket, separator, object_name = normalized.partition("/")
            if not separator or not object_name:
                raise ValueError(f"无效的 MinIO 路径: {path}")
            return bucket, object_name
        prefix = f"{settings.minio_bucket}/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
        if not normalized:
            raise ValueError("MinIO 对象路径不能为空")
        return settings.minio_bucket, normalized


async def run_document_index_task(
    task_id: int,
    *,
    source_path: str | Path | None = None,
) -> bool:
    """供 ARQ Worker 或测试直接调用的统一任务入口。"""

    return await DocumentIndexService().run(task_id, source_path=source_path)


def submit_document_index(
    index_callable: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
):
    """保留原有单节点线程池入口，新的多节点部署应优先使用 ARQ。"""

    return submit_index_task(index_callable, *args, **kwargs)
