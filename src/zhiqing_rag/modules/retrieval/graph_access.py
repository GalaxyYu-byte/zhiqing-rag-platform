"""图查询可见性门禁；Neo4j 查询必须使用返回的命名空间与当前批次。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from zhiqing_rag.modules.documents import Document
from zhiqing_rag.modules.knowledge.permissions import Identity, document_access_condition

from .graph_projection import GraphProjection


def active_graph_projection(
    session: Session, tenant_id: int, document_id: int
) -> GraphProjection | None:
    """供后台任务使用的发布状态查询；用户请求必须调用 authorized_graph_projection。"""
    return session.scalar(
        select(GraphProjection)
        .join(
            Document,
            (GraphProjection.tenant_id == Document.tenant_id)
            & (GraphProjection.document_id == Document.id)
            & (GraphProjection.generation_id == Document.active_generation_id),
        )
        .where(
            Document.tenant_id == tenant_id,
            Document.id == document_id,
            Document.status == "PUBLISHED",
            Document.deleted_at.is_(None),
            GraphProjection.status == "READY",
        )
    )


def authorized_graph_projection(
    session: Session, identity: Identity, document_id: int,
) -> GraphProjection | None:
    """返回当前用户可读的已发布图投影，检索必须限定其 namespace 和 generation。"""
    return session.scalar(
        select(GraphProjection)
        .join(
            Document,
            (GraphProjection.tenant_id == Document.tenant_id)
            & (GraphProjection.document_id == Document.id)
            & (GraphProjection.generation_id == Document.active_generation_id),
        )
        .where(
            Document.id == document_id,
            document_access_condition(identity, permission="READ", published_only=True),
            GraphProjection.status == "READY",
        )
    )
