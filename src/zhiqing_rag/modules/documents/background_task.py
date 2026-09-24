"""可恢复后台业务任务 ORM 实体映射。"""

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
    SmallInteger,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class BackgroundTask(Base):
    """记录可重试、带租约且可恢复的异步业务任务。"""

    __tablename__ = "background_task"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "idempotency_key"),
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            [f"{SCHEMA}.document.tenant_id", f"{SCHEMA}.document.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "revision_id"],
            [
                f"{SCHEMA}.document_revision.tenant_id",
                f"{SCHEMA}.document_revision.document_id",
                f"{SCHEMA}.document_revision.id",
            ],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "revision_id", "generation_id"],
            [
                f"{SCHEMA}.index_generation.tenant_id",
                f"{SCHEMA}.index_generation.document_id",
                f"{SCHEMA}.index_generation.revision_id",
                f"{SCHEMA}.index_generation.id",
            ],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "chat_session_id"],
            [f"{SCHEMA}.chat_session.tenant_id", f"{SCHEMA}.chat_session.id"],
            name="fk_task_chat_session",
        ),
        CheckConstraint("task_type IN ('INDEX','GRAPH','SUMMARY','CLEANUP')"),
        CheckConstraint(
            "status IN ('PENDING','RUNNING','RETRY_WAIT','SUCCEEDED','FAILED','CANCELLED')"
        ),
        CheckConstraint("progress BETWEEN 0 AND 100"),
        CheckConstraint("max_attempts BETWEEN 1 AND 100"),
        CheckConstraint("lease_epoch >= 0"),
        CheckConstraint("attempt_count BETWEEN 0 AND max_attempts"),
        CheckConstraint("revision_id IS NULL OR document_id IS NOT NULL"),
        CheckConstraint("generation_id IS NULL OR revision_id IS NOT NULL"),
        CheckConstraint(
            "task_type NOT IN ('INDEX','GRAPH') OR "
            "(document_id IS NOT NULL AND revision_id IS NOT NULL AND generation_id IS NOT NULL "
            "AND chat_session_id IS NULL)"
        ),
        CheckConstraint(
            "task_type <> 'SUMMARY' OR "
            "(chat_session_id IS NOT NULL AND document_id IS NULL AND revision_id IS NULL "
            "AND generation_id IS NULL)"
        ),
        CheckConstraint(
            "(status = 'RUNNING' AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL "
            "AND lease_epoch > 0) OR "
            "(status <> 'RUNNING' AND lease_owner IS NULL AND lease_expires_at IS NULL)"
        ),
        CheckConstraint(
            "(status IN ('SUCCEEDED','FAILED','CANCELLED')) = (finished_at IS NOT NULL)"
        ),
        CheckConstraint("jsonb_typeof(payload) = 'object'"),
        Index(
            "ix_task_pending",
            "next_run_at",
            "id",
            postgresql_where=text("status IN ('PENDING','RETRY_WAIT')"),
        ),
        Index("ix_task_lease", "lease_expires_at", postgresql_where=text("status='RUNNING'")),
        Index("ix_task_document", "tenant_id", "document_id", text("created_at DESC")),
        {"schema": SCHEMA, "comment": "可恢复后台业务任务"},
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
    task_type: Mapped[str] = mapped_column(String(16), nullable=False, comment="任务类型")
    document_id: Mapped[int | None] = mapped_column(BigInteger, comment="目标文档")
    revision_id: Mapped[int | None] = mapped_column(BigInteger, comment="目标修订")
    generation_id: Mapped[int | None] = mapped_column(BigInteger, comment="目标批次")
    chat_session_id: Mapped[int | None] = mapped_column(BigInteger, comment="摘要任务目标会话")
    idempotency_key: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="服务端业务幂等键"
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'PENDING'"), comment="任务状态"
    )
    stage: Mapped[str | None] = mapped_column(String(50), comment="当前处理阶段")
    progress: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=text("0"), comment="进度百分比"
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"), comment="已经开始的执行次数"
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3"), comment="最大执行次数"
    )
    lease_epoch: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="执行租约代次，领取时递增"
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), comment="持有者")
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="租约到期"
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="心跳时间"
    )
    next_run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="最早可执行时间",
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="终态时间")
    error_code: Mapped[str | None] = mapped_column(String(100), comment="脱敏错误码")
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="最小任务参数，不包含凭据",
    )
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
