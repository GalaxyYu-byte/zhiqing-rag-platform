"""事务内待投递事件 ORM 实体映射。"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    FetchedValue,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class OutboxEvent(Base):
    """与业务变更同事务写入、由后台投递器可靠发布的事件。"""

    __tablename__ = "outbox_event"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "deduplication_key"),
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            [f"{SCHEMA}.background_task.tenant_id", f"{SCHEMA}.background_task.id"],
        ),
        CheckConstraint("jsonb_typeof(payload) = 'object'"),
        CheckConstraint("status IN ('PENDING','PUBLISHING','PUBLISHED','DEAD')"),
        CheckConstraint("attempt_count >= 0"),
        CheckConstraint(
            "(status='PUBLISHING' AND locked_by IS NOT NULL AND locked_until IS NOT NULL) OR "
            "(status<>'PUBLISHING' AND locked_by IS NULL AND locked_until IS NULL)"
        ),
        CheckConstraint("(status='PUBLISHED') = (published_at IS NOT NULL)"),
        Index("ix_outbox_pending", "available_at", "id", postgresql_where=text("status='PENDING'")),
        Index("ix_outbox_lease", "locked_until", postgresql_where=text("status='PUBLISHING'")),
        Index("ix_outbox_task", "tenant_id", "task_id"),
        {"schema": SCHEMA, "comment": "事务内创建的待投递事件"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True, comment="数据库生成的主键"
    )
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tenant.id"),
        nullable=False,
        comment="所属租户",
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, comment="事件类型")
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False, comment="聚合类型")
    aggregate_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="聚合 ID")
    task_id: Mapped[int | None] = mapped_column(BigInteger, comment="关联任务")
    deduplication_key: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="事件去重键"
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="可重放事件参数",
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'PENDING'"), comment="投递状态"
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"), comment="投递次数"
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="可投递时间",
    )
    locked_by: Mapped[str | None] = mapped_column(String(128), comment="投递租约持有者")
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="投递租约到期")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="投递完成时间")
    last_error_code: Mapped[str | None] = mapped_column(String(100), comment="最近错误码")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
        comment="更新时间",
    )
