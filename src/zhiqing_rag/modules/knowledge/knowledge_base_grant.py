"""知识库用户或部门授权 ORM 实体映射。"""

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


class KnowledgeBaseGrant(Base):
    """知识库用户或部门授权"""

    __tablename__ = "knowledge_base_grant"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "kb_id"],
            [f"{SCHEMA}.knowledge_base.tenant_id", f"{SCHEMA}.knowledge_base.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "user_id"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.user_id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "department_id"],
            [f"{SCHEMA}.department.tenant_id", f"{SCHEMA}.department.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "granted_by"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        CheckConstraint("permission IN ('READ','WRITE','ADMIN')"),
        CheckConstraint("num_nonnulls(user_id, department_id) = 1"),
        Index(
            "uq_grant_user",
            "tenant_id",
            "kb_id",
            "user_id",
            unique=True,
            postgresql_where=text("user_id IS NOT NULL"),
        ),
        Index(
            "uq_grant_department",
            "tenant_id",
            "kb_id",
            "department_id",
            unique=True,
            postgresql_where=text("department_id IS NOT NULL"),
        ),
        Index("ix_grant_user_lookup", "tenant_id", "user_id"),
        Index("ix_grant_department_lookup", "tenant_id", "department_id"),
        {"schema": SCHEMA, "comment": "知识库用户或部门授权"},
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
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="知识库")
    user_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="授权用户，与 department_id 恰选一个"
    )
    department_id: Mapped[int | None] = mapped_column(BigInteger, comment="授权部门")
    permission: Mapped[str] = mapped_column(String(16), nullable=False, comment="权限等级")
    granted_by: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="授权操作者成员 ID")
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
