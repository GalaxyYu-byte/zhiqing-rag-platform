"""真实 Chunk 的独立图抽取任务、租约、重试和版本切换。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)
from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..core.config import settings
from ..core.database import SessionLocal
from ..graph.models import GraphChunkContext
from ..graph.service import GraphIndexingService
from ..models.document import DocChunk, Document
from ..models.graph import GraphTask


logger = logging.getLogger(__name__)


class GraphTaskStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class GraphTaskStage(StrEnum):
    PENDING = "PENDING"
    RETRY_WAIT = "RETRY_WAIT"
    LOADING_CHUNKS = "LOADING_CHUNKS"
    EXTRACTING = "EXTRACTING"
    SWITCHING = "SWITCHING"
    CLEANING = "CLEANING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class GraphTaskNotFoundError(LookupError):
    pass


class GraphTaskConflictError(RuntimeError):
    pass


class GraphTaskLeaseLostError(RuntimeError):
    pass


class RetryableGraphTaskError(RuntimeError):
    def __init__(self, message: str, *, delay_seconds: float) -> None:
        super().__init__(message)
        self.delay_seconds = delay_seconds


@dataclass(slots=True, frozen=True)
class GraphTaskSnapshot:
    id: int
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
    def from_model(cls, task: GraphTask) -> "GraphTaskSnapshot":
        return cls(
            **{
                field: getattr(task, field)
                for field in cls.__dataclass_fields__
            }
        )


@dataclass(slots=True, frozen=True)
class _GraphDocumentSnapshot:
    doc_id: int
    kb_id: int
    doc_version: int
    file_name: str


@dataclass(slots=True, frozen=True)
class _ChunkSnapshot:
    chunk_index: int
    content: str
    page_num: int | None
    section_title: str | None


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


async def create_graph_task(
    session: AsyncSession,
    *,
    doc_id: int,
) -> tuple[GraphTask, bool]:
    """为当前已完成的真实文档版本创建唯一图任务。"""

    document = await session.get(Document, doc_id, with_for_update=True)
    if document is None or document.is_deleted:
        raise GraphTaskNotFoundError(f"文档不存在: {doc_id}")
    if document.status != "DONE":
        raise GraphTaskConflictError("文档向量索引尚未完成")

    existing = await session.scalar(
        select(GraphTask).where(
            GraphTask.doc_id == doc_id,
            GraphTask.doc_version == document.version,
        )
    )
    if existing is not None:
        if existing.status in {
            GraphTaskStatus.FAILED.value,
            GraphTaskStatus.CANCELED.value,
        }:
            existing.status = GraphTaskStatus.PENDING.value
            existing.stage = GraphTaskStage.PENDING.value
            existing.progress_percent = 0
            existing.retry_count = 0
            existing.error_msg = None
            existing.worker_id = None
            existing.heartbeat_at = None
            existing.lease_expires_at = None
            existing.started_at = None
            existing.finished_at = None
            existing.updated_at = _utcnow_naive()
            await session.commit()
            await session.refresh(existing)
            return existing, True
        return existing, False

    now = _utcnow_naive()
    await session.execute(
        update(GraphTask)
        .where(
            GraphTask.doc_id == doc_id,
            GraphTask.status.in_(["PENDING", "PROCESSING"]),
        )
        .values(
            status=GraphTaskStatus.CANCELED.value,
            stage=GraphTaskStage.CANCELED.value,
            error_msg="文档已有更新版本",
            finished_at=now,
            lease_expires_at=None,
            updated_at=now,
        )
    )
    task = GraphTask(
        doc_id=document.id,
        kb_id=document.kb_id,
        doc_version=document.version,
        extractor_version=settings.graph_extractor_version,
        status=GraphTaskStatus.PENDING.value,
        stage=GraphTaskStage.PENDING.value,
        max_retry=settings.graph_task_max_retry,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task, True


async def get_graph_task_snapshot(
    session: AsyncSession,
    task_id: int,
) -> GraphTaskSnapshot | None:
    task = await session.get(GraphTask, task_id)
    return GraphTaskSnapshot.from_model(task) if task is not None else None


async def get_latest_document_graph_snapshot(
    session: AsyncSession,
    doc_id: int,
) -> GraphTaskSnapshot | None:
    task = await session.scalar(
        select(GraphTask)
        .where(GraphTask.doc_id == doc_id)
        .order_by(GraphTask.created_at.desc())
        .limit(1)
    )
    return GraphTaskSnapshot.from_model(task) if task is not None else None


class GraphTaskService:
    """按真实 PostgreSQL Chunk 构建候选图，并在完成后切换版本。"""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        indexing_service: GraphIndexingService | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.indexing_service = indexing_service or GraphIndexingService()
        self.repository = self.indexing_service.repository
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"

    async def run(self, task_id: int) -> bool:
        document = await self._claim(task_id)
        if document is None:
            return False

        heartbeat = asyncio.create_task(self._heartbeat_loop(task_id))
        try:
            await self._run_pipeline(task_id, document)
            return True
        except GraphTaskLeaseLostError:
            logger.warning("图任务租约已转移，停止当前执行: task_id=%s", task_id)
            return False
        except Exception as exc:
            should_retry, delay = await self._record_failure(task_id, exc)
            if should_retry:
                raise RetryableGraphTaskError(
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
        document: _GraphDocumentSnapshot,
    ) -> None:
        await self.repository.ensure_schema()
        await self._set_stage(task_id, GraphTaskStage.LOADING_CHUNKS, 3)
        chunks = await self._load_chunks(document)
        await self.repository.begin_document_version(
            kb_id=document.kb_id,
            doc_id=document.doc_id,
            doc_version=document.doc_version,
            file_name=document.file_name,
        )
        await self._update_progress(
            task_id,
            stage=GraphTaskStage.EXTRACTING,
            progress_percent=5,
            total_chunks=len(chunks),
            processed_chunks=0,
            extracted_entities=0,
            extracted_claims=0,
        )

        entity_count = 0
        claim_count = 0
        for position, chunk in enumerate(chunks, start=1):
            result = await self.indexing_service.index_chunk(
                GraphChunkContext(
                    kb_id=document.kb_id,
                    doc_id=document.doc_id,
                    doc_version=document.doc_version,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    file_name=document.file_name,
                    page_num=chunk.page_num,
                    section_title=chunk.section_title,
                )
            )
            entity_count += result.entity_count
            claim_count += result.claim_count
            await self._update_progress(
                task_id,
                stage=GraphTaskStage.EXTRACTING,
                progress_percent=5 + int(85 * position / max(1, len(chunks))),
                processed_chunks=position,
                extracted_entities=entity_count,
                extracted_claims=claim_count,
            )

        await self._set_stage(task_id, GraphTaskStage.SWITCHING, 93)
        await self.repository.activate_document_version(
            doc_id=document.doc_id,
            doc_version=document.doc_version,
            expected_chunk_count=len(chunks),
        )
        await self._set_stage(task_id, GraphTaskStage.CLEANING, 97)
        await self.repository.prune_inactive_document_versions(
            doc_id=document.doc_id,
            keep_version=document.doc_version,
        )
        await self._complete_task(task_id, document)

    async def _claim(self, task_id: int) -> _GraphDocumentSnapshot | None:
        now = _utcnow_naive()
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    select(GraphTask, Document)
                    .join(Document, Document.id == GraphTask.doc_id)
                    .where(GraphTask.id == task_id)
                    .with_for_update()
                )
            ).first()
            if row is None:
                raise GraphTaskNotFoundError(f"图任务不存在: {task_id}")
            task, document = row
            if task.status in {
                GraphTaskStatus.DONE.value,
                GraphTaskStatus.FAILED.value,
                GraphTaskStatus.CANCELED.value,
            }:
                return None
            if (
                task.status == GraphTaskStatus.PROCESSING.value
                and task.lease_expires_at is not None
                and task.lease_expires_at > now
            ):
                return None
            if document.is_deleted or document.version != task.doc_version:
                task.status = GraphTaskStatus.CANCELED.value
                task.stage = GraphTaskStage.CANCELED.value
                task.error_msg = "文档已删除或已有更新版本"
                task.finished_at = now
                task.updated_at = now
                await session.commit()
                return None

            task.status = GraphTaskStatus.PROCESSING.value
            task.stage = GraphTaskStage.LOADING_CHUNKS.value
            task.progress_percent = 1
            task.total_chunks = 0
            task.processed_chunks = 0
            task.extracted_entities = 0
            task.extracted_claims = 0
            task.error_msg = None
            task.worker_id = self.worker_id
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(
                seconds=settings.graph_lease_seconds
            )
            task.started_at = task.started_at or now
            task.finished_at = None
            task.updated_at = now
            await session.commit()
            return _GraphDocumentSnapshot(
                doc_id=document.id,
                kb_id=document.kb_id,
                doc_version=task.doc_version,
                file_name=document.file_name,
            )

    async def _load_chunks(
        self,
        document: _GraphDocumentSnapshot,
    ) -> list[_ChunkSnapshot]:
        async with self.session_factory() as session:
            chunks = (
                await session.scalars(
                    select(DocChunk)
                    .where(
                        DocChunk.doc_id == document.doc_id,
                        DocChunk.doc_version == document.doc_version,
                    )
                    .order_by(DocChunk.chunk_index)
                )
            ).all()
        return [
            _ChunkSnapshot(
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_num=chunk.page_num,
                section_title=chunk.section_title,
            )
            for chunk in chunks
        ]

    async def _set_stage(
        self,
        task_id: int,
        stage: GraphTaskStage,
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
        stage: GraphTaskStage,
        progress_percent: int,
        total_chunks: int | None = None,
        processed_chunks: int | None = None,
        extracted_entities: int | None = None,
        extracted_claims: int | None = None,
    ) -> None:
        now = _utcnow_naive()
        values = {
            "stage": stage.value,
            "progress_percent": max(0, min(100, progress_percent)),
            "heartbeat_at": now,
            "lease_expires_at": now
            + timedelta(seconds=settings.graph_lease_seconds),
            "updated_at": now,
        }
        for name, value in {
            "total_chunks": total_chunks,
            "processed_chunks": processed_chunks,
            "extracted_entities": extracted_entities,
            "extracted_claims": extracted_claims,
        }.items():
            if value is not None:
                values[name] = value

        async with self.session_factory() as session:
            result = await session.execute(
                update(GraphTask)
                .where(
                    GraphTask.id == task_id,
                    GraphTask.status == GraphTaskStatus.PROCESSING.value,
                    GraphTask.worker_id == self.worker_id,
                )
                .values(**values)
            )
            if result.rowcount != 1:
                await session.rollback()
                raise GraphTaskLeaseLostError(f"图任务租约已丢失: {task_id}")
            await session.commit()

    async def _heartbeat_loop(self, task_id: int) -> None:
        while True:
            await asyncio.sleep(settings.graph_heartbeat_interval_seconds)
            try:
                now = _utcnow_naive()
                async with self.session_factory() as session:
                    result = await session.execute(
                        update(GraphTask)
                        .where(
                            GraphTask.id == task_id,
                            GraphTask.status
                            == GraphTaskStatus.PROCESSING.value,
                            GraphTask.worker_id == self.worker_id,
                        )
                        .values(
                            heartbeat_at=now,
                            lease_expires_at=now
                            + timedelta(seconds=settings.graph_lease_seconds),
                            updated_at=now,
                        )
                    )
                    await session.commit()
                    if result.rowcount != 1:
                        return
            except Exception:
                logger.exception("更新图任务心跳失败: task_id=%s", task_id)

    async def _complete_task(
        self,
        task_id: int,
        document: _GraphDocumentSnapshot,
    ) -> None:
        now = _utcnow_naive()
        async with self.session_factory() as session:
            task = await session.get(GraphTask, task_id, with_for_update=True)
            db_document = await session.get(
                Document, document.doc_id, with_for_update=True
            )
            if (
                task is None
                or task.status != GraphTaskStatus.PROCESSING.value
                or task.worker_id != self.worker_id
                or db_document is None
                or db_document.version != document.doc_version
            ):
                raise GraphTaskLeaseLostError(f"完成图任务时版本或租约变化: {task_id}")
            task.status = GraphTaskStatus.DONE.value
            task.stage = GraphTaskStage.DONE.value
            task.progress_percent = 100
            task.finished_at = now
            task.heartbeat_at = now
            task.lease_expires_at = None
            task.updated_at = now
            await session.commit()

    async def _record_failure(
        self,
        task_id: int,
        exc: Exception,
    ) -> tuple[bool, float]:
        retryable = is_retryable_graph_exception(exc)
        now = _utcnow_naive()
        error = f"{type(exc).__name__}: {exc}"
        async with self.session_factory() as session:
            task = await session.get(GraphTask, task_id, with_for_update=True)
            if task is None or task.worker_id != self.worker_id:
                return False, 0.0
            should_retry = retryable and task.retry_count < task.max_retry
            if should_retry:
                delay = min(
                    120.0,
                    settings.graph_task_retry_base_seconds
                    * (2**task.retry_count),
                )
                task.retry_count += 1
                task.status = GraphTaskStatus.PENDING.value
                task.stage = GraphTaskStage.RETRY_WAIT.value
                task.error_msg = error
                task.worker_id = None
                task.lease_expires_at = None
                task.updated_at = now
            else:
                delay = 0.0
                task.status = GraphTaskStatus.FAILED.value
                task.stage = GraphTaskStage.FAILED.value
                task.error_msg = error
                task.finished_at = now
                task.heartbeat_at = now
                task.lease_expires_at = None
                task.updated_at = now
            await session.commit()
        return should_retry, delay


def is_retryable_graph_exception(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            ServiceUnavailable,
            SessionExpired,
            TransientError,
            OperationalError,
            InterfaceError,
            ValidationError,
            ValueError,
            RuntimeError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


async def run_graph_task(task_id: int) -> bool:
    return await GraphTaskService().run(task_id)
