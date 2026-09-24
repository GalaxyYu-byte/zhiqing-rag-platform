"""知识库 ORM 实体映射。"""

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
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class KnowledgeBase(Base):
    """知识库"""

    __tablename__ = "knowledge_base"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "code"),
        ForeignKeyConstraint(
            ["tenant_id", "created_by"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        CheckConstraint("status IN ('ACTIVE','ARCHIVED','DELETED')"),
        CheckConstraint("acl_version > 0"),
        CheckConstraint("btrim(code) <> '' AND btrim(name) <> ''"),
        CheckConstraint("(status = 'DELETED') = (deleted_at IS NOT NULL)"),
        Index("ix_kb_owner", "tenant_id", "created_by"),
        {"schema": SCHEMA, "comment": "知识库"},
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
    code: Mapped[str] = mapped_column(String(64), nullable=False, comment="知识库代码")
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="名称")
    description: Mapped[str | None] = mapped_column(Text, comment="说明")
    created_by: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="创建者成员 ID")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'ACTIVE'"), comment="知识库状态"
    )
    acl_version: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("1"), comment="知识库权限版本"
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
