"""租户 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    FetchedValue,
    Identity,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class Tenant(Base):
    """租户"""

    __tablename__ = "tenant"
    __table_args__ = (
        CheckConstraint("status IN ('ACTIVE','DISABLED')"),
        CheckConstraint("auth_version > 0"),
        CheckConstraint("btrim(code) <> '' AND btrim(name) <> ''"),
        {"schema": SCHEMA, "comment": "租户"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True, comment="数据库生成的主键"
    )
    code: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, comment="租户稳定代码，例如 DEFAULT"
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="租户名称")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'ACTIVE'"), comment="租户状态"
    )
    auth_version: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("1"), comment="租户授权版本，权限变更时递增"
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
