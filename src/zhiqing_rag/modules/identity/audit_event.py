"""追加式关键操作审计 ORM 实体映射。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class AuditEvent(Base):
    """追加式关键操作审计"""

    __tablename__ = "audit_event"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "actor_member_id"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        CheckConstraint("jsonb_typeof(details) = 'object'"),
        Index("ix_audit_tenant_time", "tenant_id", text("occurred_at DESC")),
        Index("ix_audit_target", "tenant_id", "target_type", "target_id"),
        {"schema": SCHEMA, "comment": "追加式关键操作审计"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True, comment="数据库生成的主键"
    )
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tenant.id"),
        nullable=False,
        comment="事件归属租户",
    )
    actor_member_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="操作者；系统事件可为空"
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False, comment="操作代码")
    target_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="目标类型")
    target_id: Mapped[str | None] = mapped_column(
        String(128), comment="目标标识快照，不受业务对象删除影响"
    )
    request_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), comment="请求追踪 ID")
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="脱敏审计详情",
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="发生时间",
    )
