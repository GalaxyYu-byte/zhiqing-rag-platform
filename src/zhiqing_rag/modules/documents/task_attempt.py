"""后台任务执行尝试记录 ORM 实体映射。"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class TaskAttempt(Base):
    """单次 Worker 执行及其租约、状态、错误和耗时统计。"""

    __tablename__ = "task_attempt"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "task_id", "attempt_no"),
        ForeignKeyConstraint(
            ["tenant_id", "task_id"],
            [f"{SCHEMA}.background_task.tenant_id", f"{SCHEMA}.background_task.id"],
        ),
        CheckConstraint("attempt_no > 0"),
        CheckConstraint("lease_epoch > 0"),
        CheckConstraint("status IN ('RUNNING','SUCCEEDED','FAILED','CANCELLED','LOST')"),
        CheckConstraint("(status <> 'RUNNING') = (finished_at IS NOT NULL)"),
        CheckConstraint("finished_at IS NULL OR finished_at >= started_at"),
        CheckConstraint("jsonb_typeof(metrics) = 'object'"),
        Index(
            "uq_task_running_attempt",
            "tenant_id",
            "task_id",
            unique=True,
            postgresql_where=text("status='RUNNING'"),
        ),
        {"schema": SCHEMA, "comment": "任务执行尝试记录"},
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
    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="业务任务")
    attempt_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="任务内尝试序号")
    lease_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="本次租约代次")
    worker_id: Mapped[str] = mapped_column(String(128), nullable=False, comment="Worker 标识")
    stage: Mapped[str | None] = mapped_column(String(50), comment="最后阶段")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'RUNNING'"), comment="本次尝试状态"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="开始时间",
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="结束时间")
    error_code: Mapped[str | None] = mapped_column(String(100), comment="脱敏错误码")
    error_detail: Mapped[str | None] = mapped_column(Text, comment="脱敏错误摘要")
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="耗时与使用量",
    )
