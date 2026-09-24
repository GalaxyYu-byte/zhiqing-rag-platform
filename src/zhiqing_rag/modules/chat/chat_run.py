"""一轮问答执行与幂等记录 ORM 实体映射。"""

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
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class ChatRun(Base):
    """一轮问答执行与幂等记录"""

    __tablename__ = "chat_run"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING','RUNNING','SUCCEEDED','FAILED','CANCELLED','ABANDONED')"
        ),
        CheckConstraint("jsonb_typeof(execution_snapshot) = 'object'"),
        CheckConstraint("input_tokens >= 0"),
        CheckConstraint("output_tokens >= 0"),
        CheckConstraint("lease_epoch >= 0"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "session_id", "id"),
        UniqueConstraint("tenant_id", "session_id", "client_request_id"),
        ForeignKeyConstraint(
            ["tenant_id", "session_id"],
            [f"{SCHEMA}.chat_session.tenant_id", f"{SCHEMA}.chat_session.id"],
        ),
        CheckConstraint("request_sha256 ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("btrim(question) <> '' AND btrim(client_request_id) <> ''"),
        CheckConstraint(
            "(status='RUNNING' AND lease_owner IS NOT NULL "
            "AND lease_expires_at IS NOT NULL AND lease_epoch > 0 "
            "AND started_at IS NOT NULL) OR "
            "(status<>'RUNNING' AND lease_owner IS NULL "
            "AND lease_expires_at IS NULL)"
        ),
        CheckConstraint(
            "(status IN ('SUCCEEDED','FAILED','CANCELLED','ABANDONED')) = (finished_at IS NOT NULL)"
        ),
        CheckConstraint("finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at"),
        Index(
            "uq_chat_one_active_run",
            "tenant_id",
            "session_id",
            unique=True,
            postgresql_where=text("status IN ('PENDING','RUNNING')"),
        ),
        Index("ix_chat_run_history", "tenant_id", "session_id", text("created_at DESC")),
        Index(
            "ix_chat_run_lease",
            "lease_expires_at",
            postgresql_where=text("status='RUNNING'"),
        ),
        {"schema": SCHEMA, "comment": "一轮问答执行与幂等记录"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
        nullable=False,
        comment="数据库生成的主键",
    )
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tenant.id"),
        nullable=False,
        comment="所属租户",
    )
    session_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="会话")
    client_request_id: Mapped[str] = mapped_column(
        String(128), nullable=False, comment="客户端稳定幂等标识"
    )
    request_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="规范化完整请求摘要"
    )
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="用户原问题")
    standalone_question: Mapped[str | None] = mapped_column(Text, comment="经过校验的独立问题")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'PENDING'"), comment="执行状态"
    )
    execution_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="查询计划、模型配置与授权版本快照；不能作为实时授权依据",
    )
    model_name: Mapped[str | None] = mapped_column(
        String(150), comment="生成模型，纯结构化回答可为空"
    )
    input_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="累计输入 token"
    )
    output_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="累计输出 token"
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), comment="执行进程标识")
    lease_epoch: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="执行代次"
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="运行租约"
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="开始时间")
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="终态时间"
    )
    error_code: Mapped[str | None] = mapped_column(String(100), comment="错误码")
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
