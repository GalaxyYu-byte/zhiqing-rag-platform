"""可撤销登录会话 ORM 实体映射。"""

from datetime import datetime
from uuid import UUID

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
from sqlalchemy.dialects.postgresql import INET
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class AuthSession(Base):
    """可撤销登录会话"""

    __tablename__ = "auth_session"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "tenant_member_id"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        CheckConstraint("refresh_token_hash ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("rotation >= 0"),
        CheckConstraint("expires_at > created_at"),
        Index(
            "ix_auth_session_member",
            "tenant_id",
            "tenant_member_id",
            postgresql_where=text("revoked_at IS NULL"),
        ),
        Index("ix_auth_session_expiry", "expires_at"),
        {"schema": SCHEMA, "comment": "可撤销登录会话"},
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
    tenant_member_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="租户成员")
    session_key: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=False,
        unique=True,
        server_default=text("gen_random_uuid()"),
        comment="令牌携带的随机会话标识",
    )
    refresh_token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, comment="Refresh Token 的 SHA-256 摘要"
    )
    rotation: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="令牌轮换代次"
    )
    user_agent: Mapped[str | None] = mapped_column(String(500), comment="客户端标识")
    ip_address: Mapped[str | None] = mapped_column(INET, comment="最近使用 IP")
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="到期时间"
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="撤销时间")
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="最近使用时间"
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
