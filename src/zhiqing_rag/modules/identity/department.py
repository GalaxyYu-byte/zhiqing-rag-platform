"""租户内部门 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
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


class Department(Base):
    """租户内部门"""

    __tablename__ = "department"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "code"),
        ForeignKeyConstraint(
            ["tenant_id", "parent_id"],
            [f"{SCHEMA}.department.tenant_id", f"{SCHEMA}.department.id"],
        ),
        CheckConstraint("parent_id IS NULL OR parent_id <> id"),
        CheckConstraint("btrim(code) <> '' AND btrim(name) <> ''"),
        Index("ix_department_parent", "tenant_id", "parent_id"),
        {"schema": SCHEMA, "comment": "租户内部门"},
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
    code: Mapped[str] = mapped_column(String(64), nullable=False, comment="部门代码")
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="部门名称")
    parent_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="父部门，同租户；多级环路由服务层检查"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true"), comment="是否启用"
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
