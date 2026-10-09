"""租约、授权、分块完整性与期望修订校验后，在同一事务发布索引。"""

from types import SimpleNamespace

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zhiqing_rag.modules import chat as _chat  # noqa: F401
from zhiqing_rag.modules.documents import (
    BackgroundTask,
    Document,
    DocumentRevision,
    IndexGeneration,
    OutboxEvent,
    TaskAttempt,
)
from zhiqing_rag.modules.documents.chunk_storage import (
    ChunkScope,
    complete_embedded_chunks,
    save_raw_chunks,
)
from zhiqing_rag.modules.documents.task_queue import LeaseLostError, ProcessingError, TaskLease
from zhiqing_rag.modules.identity import Department, Tenant, TenantMember, User
from zhiqing_rag.modules.knowledge import KnowledgeBase
from zhiqing_rag.modules.knowledge.permissions import document_access_condition
from zhiqing_rag.modules.retrieval import GraphProjection
from zhiqing_rag.workers.graph_extraction import EXTRACTOR_VERSION


def lock_work_knowledge_base(session: Session, tenant_id: int, document_id: int) -> int:
    """先锁知识库，再允许任务和文档行锁，避免与权限管理事务反向等待。"""
    kb_id = session.scalar(
        select(Document.kb_id).where(
            Document.tenant_id == tenant_id,
            Document.id == document_id,
        )
    )
    if kb_id is None:
        raise ProcessingError("REVISION_UNAVAILABLE")
    if session.scalar(
        select(KnowledgeBase.id)
        .where(
            KnowledgeBase.tenant_id == tenant_id,
            KnowledgeBase.id == kb_id,
            KnowledgeBase.status == "ACTIVE",
            KnowledgeBase.deleted_at.is_(None),
        )
        .with_for_update(read=True)
    ) is None:
        raise ProcessingError("ACCESS_REVOKED")
    return kb_id


def verify_access(session: Session, document: Document, revision_id: int | None = None) -> None:
    """调用者先持有知识库共享锁；按当前修订上传者的实时授权校验后台工作。"""
    uploaded_by = session.scalar(
        select(DocumentRevision.uploaded_by).where(
            DocumentRevision.tenant_id == document.tenant_id,
            DocumentRevision.document_id == document.id,
            DocumentRevision.id == (
                document.desired_revision_id if revision_id is None else revision_id
            ),
        )
    )
    if uploaded_by is None:
        raise ProcessingError("REVISION_UNAVAILABLE")
    member = session.scalar(
        select(TenantMember)
        .join(Tenant, Tenant.id == TenantMember.tenant_id)
        .join(User, User.id == TenantMember.user_id)
        .where(
            TenantMember.tenant_id == document.tenant_id,
            TenantMember.id == uploaded_by,
            TenantMember.status == "ACTIVE",
            Tenant.status == "ACTIVE",
            User.is_active,
            TenantMember.clearance >= document.confidentiality,
        )
        .with_for_update(read=True, of=[TenantMember, Tenant, User])
        .execution_options(populate_existing=True)
    )
    if member is None:
        raise ProcessingError("ACCESS_REVOKED")
    if (
        member.department_id is not None
        and session.scalar(
            select(Department.id)
            .where(
                Department.tenant_id == member.tenant_id,
                Department.id == member.department_id,
                Department.is_active,
            )
            .with_for_update(read=True)
        )
        is None
    ):
        raise ProcessingError("ACCESS_REVOKED")
    identity = SimpleNamespace(
        tenant_id=member.tenant_id,
        member_id=member.id,
        user_id=member.user_id,
        department_id=member.department_id,
        clearance=member.clearance,
    )
    if session.scalar(
        select(Document.id).where(
            Document.id == document.id,
            document_access_condition(identity, permission="WRITE", published_only=False),
        )
    ) is None:
        raise ProcessingError("ACCESS_REVOKED")


def verify_work_access(engine, lease: TaskLease) -> None:
    with Session(engine) as session, session.begin():
        kb_id = lock_work_knowledge_base(session, lease.tenant_id, lease.document_id)
        document = session.scalar(
            select(Document)
            .where(
                Document.tenant_id == lease.tenant_id,
                Document.id == lease.document_id,
                Document.desired_revision_id == lease.revision_id,
                Document.status.in_(["DRAFT", "PUBLISHED"]),
                Document.deleted_at.is_(None),
            )
            .with_for_update(read=True)
        )
        if document is None or document.kb_id != kb_id:
            raise ProcessingError("REVISION_UNAVAILABLE")
        verify_access(session, document, lease.revision_id)


def stage_chunks_for_graph(
    engine,
    lease: TaskLease,
    chunks,
    graph_model_name: str = "deepseek-flash",
    graph_namespace: str = "zhiqing-rag-platform",
    graph_max_attempts: int = 3,
) -> None:
    """解析完成即提交原始分块与 GRAPH 任务，不等待向量模型。"""
    with Session(engine) as session, session.begin():
        kb_id = lock_work_knowledge_base(session, lease.tenant_id, lease.document_id)
        task = session.scalar(
            select(BackgroundTask)
            .where(
                BackgroundTask.tenant_id == lease.tenant_id,
                BackgroundTask.id == lease.task_id,
                BackgroundTask.document_id == lease.document_id,
                BackgroundTask.revision_id == lease.revision_id,
                BackgroundTask.generation_id == lease.generation_id,
                BackgroundTask.status == "RUNNING",
                BackgroundTask.lease_owner == lease.worker_id,
                BackgroundTask.lease_epoch == lease.epoch,
                BackgroundTask.lease_expires_at > func.clock_timestamp(),
            )
            .with_for_update()
        )
        if task is None:
            raise LeaseLostError("租约已失效，不允许发布")
        document = session.scalar(
            select(Document)
            .where(
                Document.tenant_id == lease.tenant_id,
                Document.id == lease.document_id,
                Document.desired_revision_id == lease.revision_id,
                Document.status.in_(["DRAFT", "PUBLISHED"]),
                Document.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if document is None or document.kb_id != kb_id:
            raise ProcessingError("REVISION_UNAVAILABLE")
        verify_access(session, document, lease.revision_id)
        scope = ChunkScope(
            lease.tenant_id,
            document.kb_id,
            lease.document_id,
            lease.revision_id,
            lease.generation_id,
        )
        save_raw_chunks(session, scope, chunks)
        if (
            session.scalar(
                select(BackgroundTask.id).where(
                    BackgroundTask.tenant_id == lease.tenant_id,
                    BackgroundTask.idempotency_key == f"graph:{lease.generation_id}:v1",
                )
            )
            is None
        ):
            session.add(
                BackgroundTask(
                    tenant_id=lease.tenant_id,
                    task_type="GRAPH",
                    document_id=lease.document_id,
                    revision_id=lease.revision_id,
                    generation_id=lease.generation_id,
                    idempotency_key=f"graph:{lease.generation_id}:v1",
                    stage="QUEUED",
                    max_attempts=graph_max_attempts,
                    payload={"pipeline": "document-graph-v1"},
                )
            )
            session.add(
                GraphProjection(
                    tenant_id=lease.tenant_id,
                    document_id=lease.document_id,
                    revision_id=lease.revision_id,
                    generation_id=lease.generation_id,
                    projection_version=1,
                    graph_namespace=graph_namespace,
                    extractor_version=EXTRACTOR_VERSION,
                    model_name=graph_model_name,
                    status="BUILDING",
                )
            )
        expires_at = task.lease_expires_at
        session.flush()
        if session.scalar(select(func.clock_timestamp())) >= expires_at:
            raise LeaseLostError("提交前租约已过期，回滚分块暂存")


def publish_chunks(engine, lease: TaskLease, embedded, metrics: dict) -> None:
    with Session(engine) as session, session.begin():
        kb_id = lock_work_knowledge_base(session, lease.tenant_id, lease.document_id)
        task = session.scalar(
            select(BackgroundTask)
            .where(
                BackgroundTask.tenant_id == lease.tenant_id,
                BackgroundTask.id == lease.task_id,
                BackgroundTask.document_id == lease.document_id,
                BackgroundTask.revision_id == lease.revision_id,
                BackgroundTask.generation_id == lease.generation_id,
                BackgroundTask.status == "RUNNING",
                BackgroundTask.lease_owner == lease.worker_id,
                BackgroundTask.lease_epoch == lease.epoch,
                BackgroundTask.lease_expires_at > func.clock_timestamp(),
            )
            .with_for_update()
        )
        if task is None:
            raise LeaseLostError("租约已失效，不允许发布")
        document = session.scalar(
            select(Document)
            .where(
                Document.tenant_id == lease.tenant_id,
                Document.id == lease.document_id,
                Document.desired_revision_id == lease.revision_id,
                Document.status.in_(["DRAFT", "PUBLISHED"]),
                Document.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if document is None or document.kb_id != kb_id:
            raise ProcessingError("REVISION_UNAVAILABLE")
        verify_access(session, document, lease.revision_id)
        scope = ChunkScope(
            lease.tenant_id,
            document.kb_id,
            lease.document_id,
            lease.revision_id,
            lease.generation_id,
        )
        complete_embedded_chunks(session, scope, embedded)
        generation = session.get(IndexGeneration, lease.generation_id)
        now = session.scalar(select(func.clock_timestamp()))
        generation.first_published_at = now
        old_generation_id = document.active_generation_id
        if old_generation_id is not None and old_generation_id != lease.generation_id:
            old_projections = session.scalars(
                select(GraphProjection)
                .where(
                    GraphProjection.tenant_id == lease.tenant_id,
                    GraphProjection.generation_id == old_generation_id,
                )
                .with_for_update()
            )
            for old_projection in old_projections:
                old_projection.status = "STALE"
                dedup_key = f"graph:purge:{old_generation_id}:v{old_projection.projection_version}"
                existing_cleanup = session.scalar(
                    select(OutboxEvent.id).where(
                        OutboxEvent.tenant_id == lease.tenant_id,
                        OutboxEvent.deduplication_key == dedup_key,
                    )
                )
                if existing_cleanup is None:
                    session.add(
                        OutboxEvent(
                            tenant_id=lease.tenant_id,
                            event_type="GRAPH_PURGE_GENERATION",
                            aggregate_type="graph_cleanup",
                            aggregate_id=str(lease.document_id),
                            deduplication_key=dedup_key,
                            payload={
                                "document_id": lease.document_id,
                                "generation_id": old_generation_id,
                                "namespace": old_projection.graph_namespace,
                            },
                        )
                    )
        document.active_generation_id = lease.generation_id
        document.status = "PUBLISHED"
        task.status = "SUCCEEDED"
        task.stage = "COMPLETED"
        task.progress = 100
        task.finished_at = now
        expires_at = task.lease_expires_at
        task.lease_owner = task.lease_expires_at = None
        task.error_code = None
        attempt = session.scalar(
            select(TaskAttempt).where(
                TaskAttempt.tenant_id == lease.tenant_id,
                TaskAttempt.task_id == lease.task_id,
                TaskAttempt.lease_epoch == lease.epoch,
                TaskAttempt.status == "RUNNING",
            )
        )
        if attempt is None:
            raise LeaseLostError("执行尝试已失效")
        attempt.status = "SUCCEEDED"
        attempt.stage = "COMPLETED"
        attempt.finished_at = now
        attempt.metrics = metrics
        session.flush()
        if session.scalar(select(func.clock_timestamp())) >= expires_at:
            raise LeaseLostError("提交前租约已过期，回滚本次发布")
