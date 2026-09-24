"""租户成员与部门角色 ORM 实体映射。"""

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
    SmallInteger,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class TenantMember(Base):
    """租户成员与部门角色"""

    __tablename__ = "tenant_member"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "user_id"),
        ForeignKeyConstraint(
            ["tenant_id", "department_id"],
            [f"{SCHEMA}.department.tenant_id", f"{SCHEMA}.department.id"],
        ),
        CheckConstraint("role IN ('MEMBER','ADMIN')"),
        CheckConstraint("clearance BETWEEN 0 AND 2"),
        CheckConstraint("status IN ('ACTIVE','DISABLED')"),
        CheckConstraint("auth_version > 0"),
        Index("ix_member_department", "tenant_id", "department_id"),
        Index("ix_member_user", "user_id"),
        {"schema": SCHEMA, "comment": "租户成员与部门角色"},
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
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.user_account.id"),
        nullable=False,
        comment="登录用户",
    )
    department_id: Mapped[int | None] = mapped_column(BigInteger, comment="所属部门")
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'MEMBER'"), comment="租户内角色"
    )
    clearance: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        server_default=text("0"),
        comment="密级：0 内部公开、1 部门内部、2 机密",
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'ACTIVE'"), comment="成员状态"
    )
    auth_version: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("1"), comment="成员授权版本"
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
