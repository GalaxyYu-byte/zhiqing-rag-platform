"""独立 GRAPH 任务队列；只更新图任务与投影，不触碰文档索引状态。"""

import random
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.documents import BackgroundTask, Document, DocumentChunk, TaskAttempt
from zhiqing_rag.modules.documents.task_queue import LeaseLostError
from zhiqing_rag.modules.retrieval import GraphProjection
from zhiqing_rag.workers.graph_extraction import (
    EXTRACTOR_VERSION,
    Extraction,
    validate_evidence,
)
from zhiqing_rag.workers.publication import lock_work_knowledge_base, verify_access

REBUILD_PIPELINE = "document-graph-rebuild-v1"


@dataclass(frozen=True)
class GraphLease:
    task_id: int
    tenant_id: int
    document_id: int
    revision_id: int
    generation_id: int
    owner: str
    epoch: int
    projection_version: int = 1


class GraphQueue:
    def __init__(self, engine, settings: Settings):
        self.engine = engine
        self.settings = settings

    def claim(self, owner: str, task_id: int | None = None) -> GraphLease | None:
        with Session(self.engine) as session, session.begin():
            task = session.scalar(
                select(BackgroundTask)
                .where(
                    BackgroundTask.task_type == "GRAPH",
                    BackgroundTask.payload["pipeline"]
                    .as_string()
                    .in_(
                        ["document-graph-v1", REBUILD_PIPELINE]
                        if task_id is not None
                        else ["document-graph-v1"]
                    ),
                    or_(BackgroundTask.id == task_id, task_id is None),
                    or_(
                        (BackgroundTask.status.in_(["PENDING", "RETRY_WAIT"]))
                        & (BackgroundTask.next_run_at <= func.clock_timestamp()),
                        (BackgroundTask.status == "RUNNING")
                        & (BackgroundTask.lease_expires_at <= func.clock_timestamp()),
                    ),
                )
                .order_by(BackgroundTask.next_run_at, BackgroundTask.id)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if task is None:
                return None
            for attempt in session.scalars(
                select(TaskAttempt).where(
                    TaskAttempt.tenant_id == task.tenant_id,
                    TaskAttempt.task_id == task.id,
                    TaskAttempt.status == "RUNNING",
                )
            ):
                attempt.status = "LOST"
                attempt.error_code = "LEASE_EXPIRED"
                attempt.finished_at = session.scalar(select(func.clock_timestamp()))
            if task.attempt_count >= task.max_attempts:
                self._terminal_failure(session, task, "ATTEMPTS_EXHAUSTED")
                return None
            now = session.scalar(select(func.clock_timestamp()))
            task.status = "RUNNING"
            task.stage = "EXTRACTING"
            task.attempt_count += 1
            task.lease_epoch += 1
            task.lease_owner = owner
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=self.settings.graph_lease_seconds)
            task.error_code = None
            session.add(
                TaskAttempt(
                    tenant_id=task.tenant_id,
                    task_id=task.id,
                    attempt_no=task.attempt_count,
                    lease_epoch=task.lease_epoch,
                    worker_id=owner,
                    stage="EXTRACTING",
                )
            )
            return GraphLease(
                task.id,
                task.tenant_id,
                task.document_id,
                task.revision_id,
                task.generation_id,
                owner,
                task.lease_epoch,
                int(task.payload.get("projection_version", 1)),
            )

    def _locked(self, session: Session, lease: GraphLease) -> BackgroundTask:
        task = session.scalar(
            select(BackgroundTask)
            .where(
                BackgroundTask.id == lease.task_id,
                BackgroundTask.tenant_id == lease.tenant_id,
                BackgroundTask.task_type == "GRAPH",
                BackgroundTask.status == "RUNNING",
                BackgroundTask.lease_owner == lease.owner,
                BackgroundTask.lease_epoch == lease.epoch,
                BackgroundTask.lease_expires_at > func.clock_timestamp(),
            )
            .with_for_update()
        )
        if (
            task is None
            or int(task.payload.get("projection_version", 1)) != lease.projection_version
        ):
            raise LeaseLostError("图任务租约已失效")
        return task

    def heartbeat(self, lease: GraphLease, progress: int | None = None):
        with Session(self.engine) as session, session.begin():
            task = self._locked(session, lease)
            now = session.scalar(select(func.clock_timestamp()))
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=self.settings.graph_lease_seconds)
            if progress is not None:
                task.progress = max(task.progress, progress)

    def checkpoint_state(self, lease: GraphLease) -> tuple[int, set[str], set[str]]:
        with Session(self.engine) as session:
            task = self._locked(session, lease)
            payload = task.payload
            return (
                int(payload.get("next_chunk_index", 0)),
                set(payload.get("entity_keys", [])),
                set(payload.get("relation_keys", [])),
            )

    def checkpoint(
        self,
        lease: GraphLease,
        next_chunk_index: int,
        total_chunks: int,
        entity_keys: set[str],
        relation_keys: set[str],
    ) -> None:
        """Neo4j 单块提交后再推进游标；崩溃时最多重放当前块。"""
        with Session(self.engine) as session, session.begin():
            task = self._locked(session, lease)
            if int(task.payload.get("next_chunk_index", 0)) != next_chunk_index - 1:
                raise LeaseLostError("图谱分块检查点已变更")
            task.payload = {
                **task.payload,
                "next_chunk_index": next_chunk_index,
                "entity_keys": sorted(entity_keys),
                "relation_keys": sorted(relation_keys),
            }
            now = session.scalar(select(func.clock_timestamp()))
            task.heartbeat_at = now
            task.lease_expires_at = now + timedelta(seconds=self.settings.graph_lease_seconds)
            task.progress = min(99, int(next_chunk_index * 100 / total_chunks))

    def scope(self, lease: GraphLease) -> tuple[dict, int]:
        with Session(self.engine) as session:
            lock_work_knowledge_base(session, lease.tenant_id, lease.document_id)
            task = self._locked(session, lease)
            document = session.scalar(
                select(Document).where(
                    Document.id == lease.document_id,
                    Document.tenant_id == lease.tenant_id,
                    Document.desired_revision_id == lease.revision_id,
                    Document.status.in_(["DRAFT", "PUBLISHED"]),
                    Document.deleted_at.is_(None),
                )
            )
            if document is None:
                raise ValueError("REVISION_UNAVAILABLE")
            if task.payload.get("pipeline") == REBUILD_PIPELINE and (
                document.status != "PUBLISHED"
                or document.active_generation_id != lease.generation_id
            ):
                raise ValueError("REVISION_UNAVAILABLE")
            verify_access(session, document, lease.revision_id)
            projection = session.scalar(
                select(GraphProjection).where(
                    GraphProjection.tenant_id == lease.tenant_id,
                    GraphProjection.generation_id == lease.generation_id,
                    GraphProjection.projection_version == lease.projection_version,
                )
            )
            if projection is None or (
                projection.model_name != self.settings.graph_extraction_model
                or projection.graph_namespace != self.settings.graph_namespace
                or projection.extractor_version != EXTRACTOR_VERSION
            ):
                raise ValueError("MODEL_CONFIG_CHANGED")
            count = session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(
                    DocumentChunk.tenant_id == lease.tenant_id,
                    DocumentChunk.generation_id == lease.generation_id,
                )
            )
            return {
                "tenant_id": lease.tenant_id,
                "kb_id": document.kb_id,
                "document_id": lease.document_id,
                "revision_id": lease.revision_id,
                "generation_id": lease.generation_id,
            }, count

    def chunk(self, lease: GraphLease, index: int) -> dict:
        with Session(self.engine) as session:
            chunk = session.scalar(
                select(DocumentChunk).where(
                    DocumentChunk.tenant_id == lease.tenant_id,
                    DocumentChunk.generation_id == lease.generation_id,
                    DocumentChunk.chunk_index == index,
                )
            )
            if chunk is None:
                raise ValueError("CHUNK_MISSING")
            return {
                field: getattr(chunk, field)
                for field in (
                    "id",
                    "chunk_index",
                    "content",
                    "content_sha256",
                    "page_number",
                    "section_title",
                    "start_offset",
                    "end_offset",
                    "metadata_",
                )
            }

    def cached_extraction(self, lease: GraphLease, chunk: dict) -> Extraction | None:
        """仅复用同一文档中正文完全相同、模型和抽取规则相同的结果。"""
        with Session(self.engine) as session:
            task = self._locked(session, lease)
            if lease.projection_version > 1:
                cache = task.payload.get("extraction_cache", {}).get(str(chunk["id"]), {})
                try:
                    if (
                        cache["content_sha256"] != chunk["content_sha256"]
                        or cache["model"] != self.settings.graph_extraction_model
                        or cache["extractor_version"] != EXTRACTOR_VERSION
                    ):
                        return None
                    result = Extraction.model_validate(cache["extraction"])
                    validate_evidence(result, chunk["content"])
                    return result
                except (KeyError, TypeError, ValueError):
                    return None
            candidates = session.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.tenant_id == lease.tenant_id,
                    DocumentChunk.document_id == lease.document_id,
                    DocumentChunk.content_sha256 == chunk["content_sha256"],
                    self._cache_expression(lease)["model"].as_string()
                    == self.settings.graph_extraction_model,
                    self._cache_expression(lease)["extractor_version"].as_string()
                    == EXTRACTOR_VERSION,
                )
                .order_by(DocumentChunk.id.desc())
            )
            for candidate in candidates:
                # 哈希只负责筛选；正文二次比较可防止误匹配。
                if candidate.content != chunk["content"]:
                    continue
                try:
                    result = Extraction.model_validate(
                        self._cache_record(lease, candidate.metadata_)["extraction"]
                    )
                    validate_evidence(result, chunk["content"])
                except (KeyError, TypeError, ValueError):
                    continue
                return result
        return None

    @staticmethod
    def _cache_expression(lease: GraphLease):
        return DocumentChunk.metadata_["_graph_extraction_cache"]

    @staticmethod
    def _cache_record(lease: GraphLease, metadata: dict) -> dict:
        return metadata["_graph_extraction_cache"]

    def save_extraction(self, lease: GraphLease, chunk: dict, extraction: Extraction) -> None:
        """LLM 成功后先存入 PostgreSQL；图写入或检查点失败时无需再次请求模型。"""
        validate_evidence(extraction, chunk["content"])
        with Session(self.engine) as session, session.begin():
            task = self._locked(session, lease)
            row = session.scalar(
                select(DocumentChunk)
                .where(
                    DocumentChunk.tenant_id == lease.tenant_id,
                    DocumentChunk.generation_id == lease.generation_id,
                    DocumentChunk.id == chunk["id"],
                )
                .with_for_update()
            )
            if row is None or row.content != chunk["content"]:
                raise ValueError("CHUNK_MISSING")
            cache = {
                "model": self.settings.graph_extraction_model,
                "extractor_version": EXTRACTOR_VERSION,
                "extraction": extraction.model_dump(mode="json"),
            }
            if lease.projection_version > 1:
                # 已发布 Chunk 由数据库触发器冻结，候选缓存属于独立任务。
                task.payload = {
                    **task.payload,
                    "extraction_cache": {
                        **task.payload.get("extraction_cache", {}),
                        str(row.id): {**cache, "content_sha256": row.content_sha256},
                    },
                }
            else:
                row.metadata_ = {**row.metadata_, "_graph_extraction_cache": cache}

    def finish(self, lease: GraphLease, entity_count: int, relation_count: int):
        with Session(self.engine) as session, session.begin():
            lock_work_knowledge_base(session, lease.tenant_id, lease.document_id)
            task = self._locked(session, lease)
            expires_at = task.lease_expires_at
            document = session.scalar(
                select(Document)
                .where(
                    Document.id == lease.document_id,
                    Document.tenant_id == lease.tenant_id,
                    Document.desired_revision_id == lease.revision_id,
                    Document.status.in_(["DRAFT", "PUBLISHED"]),
                    Document.deleted_at.is_(None),
                )
                .with_for_update()
            )
            if document is None:
                raise ValueError("REVISION_UNAVAILABLE")
            staged = task.payload.get("pipeline") == REBUILD_PIPELINE
            if staged and (
                document.status != "PUBLISHED"
                or document.active_generation_id != lease.generation_id
            ):
                raise ValueError("REVISION_UNAVAILABLE")
            verify_access(session, document, lease.revision_id)
            projection = session.scalar(
                select(GraphProjection)
                .where(
                    GraphProjection.tenant_id == lease.tenant_id,
                    GraphProjection.generation_id == lease.generation_id,
                    GraphProjection.projection_version == lease.projection_version,
                )
                .with_for_update()
            )
            now = session.scalar(select(func.clock_timestamp()))
            # 候选构建完成仍保持不可见；校验和显式切换后才成为 READY。
            projection.status = "BUILDING" if staged else "READY"
            projection.ready_at = now
            projection.entity_count = entity_count
            projection.relation_count = relation_count
            projection.error_code = None
            task.status = "SUCCEEDED"
            task.stage = "AWAITING_REVIEW" if staged else "COMPLETED"
            task.progress = 100
            task.finished_at = now
            task.lease_owner = task.lease_expires_at = None
            attempt = session.scalar(
                select(TaskAttempt).where(
                    TaskAttempt.tenant_id == lease.tenant_id,
                    TaskAttempt.task_id == lease.task_id,
                    TaskAttempt.lease_epoch == lease.epoch,
                    TaskAttempt.status == "RUNNING",
                )
            )
            attempt.status = "SUCCEEDED"
            attempt.stage = "COMPLETED"
            attempt.finished_at = now
            attempt.metrics = {"entity_count": entity_count, "relation_count": relation_count}
            session.flush()
            if session.scalar(select(func.clock_timestamp())) >= expires_at:
                raise LeaseLostError("提交前图任务租约已过期")

    def fail(self, lease: GraphLease, code: str, retryable: bool):
        with Session(self.engine) as session, session.begin():
            try:
                task = self._locked(session, lease)
            except LeaseLostError:
                return
            now = session.scalar(select(func.clock_timestamp()))
            retry = retryable and task.attempt_count < task.max_attempts
            task.status = "RETRY_WAIT" if retry else "FAILED"
            task.stage = task.status
            task.error_code = code
            task.lease_owner = task.lease_expires_at = None
            task.finished_at = None if retry else now
            if retry:
                task.next_run_at = now + timedelta(
                    seconds=min(300, 15 * 2 ** (task.attempt_count - 1)) + random.uniform(0, 5)
                )
            attempt = session.scalar(
                select(TaskAttempt).where(
                    TaskAttempt.tenant_id == lease.tenant_id,
                    TaskAttempt.task_id == lease.task_id,
                    TaskAttempt.lease_epoch == lease.epoch,
                    TaskAttempt.status == "RUNNING",
                )
            )
            attempt.status = "FAILED"
            attempt.error_code = code
            attempt.finished_at = now
            if not retry:
                projection = session.scalar(
                    select(GraphProjection)
                    .where(
                        GraphProjection.tenant_id == lease.tenant_id,
                        GraphProjection.generation_id == lease.generation_id,
                        GraphProjection.projection_version == lease.projection_version,
                    )
                    .with_for_update()
                )
                if projection:
                    projection.status = "FAILED"
                    projection.error_code = code

    @staticmethod
    def _terminal_failure(session: Session, task: BackgroundTask, code: str):
        now = session.scalar(select(func.clock_timestamp()))
        task.status = "FAILED"
        task.stage = "FAILED"
        task.error_code = code
        task.finished_at = now
        task.lease_owner = task.lease_expires_at = None
        projection = session.scalar(
            select(GraphProjection)
            .where(
                GraphProjection.tenant_id == task.tenant_id,
                GraphProjection.generation_id == task.generation_id,
                GraphProjection.projection_version
                == int(task.payload.get("projection_version", 1)),
            )
            .with_for_update()
        )
        if projection:
            projection.status = "FAILED"
            projection.error_code = code
