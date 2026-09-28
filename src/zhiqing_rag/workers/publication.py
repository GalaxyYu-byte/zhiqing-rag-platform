"""租约、授权、分块完整性与期望修订校验后，在同一事务发布索引。"""

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from zhiqing_rag.modules import chat as _chat  # noqa: F401
from zhiqing_rag.modules.documents import BackgroundTask, Document, IndexGeneration, TaskAttempt
from zhiqing_rag.modules.documents.chunk_storage import ChunkScope, save_embedded_chunks
from zhiqing_rag.modules.documents.task_queue import LeaseLostError, ProcessingError, TaskLease
from zhiqing_rag.modules.identity import Department, Tenant, TenantMember, User
from zhiqing_rag.modules.knowledge import KnowledgeBase, KnowledgeBaseGrant


def verify_access(session: Session, document: Document) -> None:
    member = session.scalar(
        select(TenantMember)
        .join(Tenant, Tenant.id == TenantMember.tenant_id)
        .join(User, User.id == TenantMember.user_id)
        .where(
            TenantMember.tenant_id == document.tenant_id,
            TenantMember.id == document.created_by,
            TenantMember.status == "ACTIVE",
            Tenant.status == "ACTIVE",
            User.is_active,
            TenantMember.clearance >= document.confidentiality,
        )
        .with_for_update(read=True, of=[TenantMember, Tenant, User])
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
    kb = session.scalar(
        select(KnowledgeBase)
        .where(
            KnowledgeBase.tenant_id == document.tenant_id,
            KnowledgeBase.id == document.kb_id,
            KnowledgeBase.status == "ACTIVE",
            KnowledgeBase.deleted_at.is_(None),
        )
        .with_for_update(read=True)
    )
    grant = session.scalar(
        select(KnowledgeBaseGrant)
        .where(
            KnowledgeBaseGrant.tenant_id == document.tenant_id,
            KnowledgeBaseGrant.kb_id == document.kb_id,
            KnowledgeBaseGrant.permission.in_(["WRITE", "ADMIN"]),
            or_(
                KnowledgeBaseGrant.user_id == member.user_id,
                KnowledgeBaseGrant.department_id == member.department_id
                if member.department_id is not None
                else False,
            ),
        )
        .with_for_update(read=True)
    )
    if kb is None or grant is None:
        raise ProcessingError("ACCESS_REVOKED")


def verify_work_access(engine, lease: TaskLease) -> None:
    with Session(engine) as session, session.begin():
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
        if document is None:
            raise ProcessingError("REVISION_UNAVAILABLE")
        verify_access(session, document)


def publish_chunks(engine, lease: TaskLease, embedded, metrics: dict) -> None:
    with Session(engine) as session, session.begin():
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
        if document is None:
            raise ProcessingError("REVISION_UNAVAILABLE")
        verify_access(session, document)
        scope = ChunkScope(
            lease.tenant_id,
            document.kb_id,
            lease.document_id,
            lease.revision_id,
            lease.generation_id,
        )
        save_embedded_chunks(session, scope, embedded)
        generation = session.get(IndexGeneration, lease.generation_id)
        now = session.scalar(select(func.clock_timestamp()))
        generation.first_published_at = now
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
