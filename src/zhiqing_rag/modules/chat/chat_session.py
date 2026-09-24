"""用户问答会话 ORM 实体映射。"""

from datetime import datetime

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
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class ChatSession(Base):
    """用户问答会话"""

    __tablename__ = "chat_session"
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE','ARCHIVED','DELETED')"),
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "owner_member_id"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        CheckConstraint("(status='DELETED') = (deleted_at IS NOT NULL)"),
        Index("ix_chat_owner", "tenant_id", "owner_member_id", text("last_active_at DESC")),
        {"schema": SCHEMA, "comment": "用户问答会话"},
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
    owner_member_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="会话拥有者成员"
    )
    title: Mapped[str] = mapped_column(
        String(300), nullable=False, server_default=text("'新会话'"), comment="标题"
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'ACTIVE'"), comment="会话状态"
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="最近活跃时间",
    )
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="软删除时间"
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
