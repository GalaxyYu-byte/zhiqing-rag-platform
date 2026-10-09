"""文档跨部门显式共享，仍受知识库 ACL 和成员密级约束。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    FetchedValue,
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


class DocumentAccessGrant(Base):
    __tablename__ = "document_access_grant"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            [f"{SCHEMA}.document.tenant_id", f"{SCHEMA}.document.id"], ondelete="CASCADE",
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
        CheckConstraint("permission IN ('READ','WRITE')"),
        CheckConstraint("num_nonnulls(user_id, department_id) = 1"),
        Index("uq_doc_grant_user", "tenant_id", "document_id", "user_id", unique=True,
              postgresql_where=text("user_id IS NOT NULL")),
        Index("uq_doc_grant_department", "tenant_id", "document_id", "department_id",
              unique=True, postgresql_where=text("department_id IS NOT NULL")),
        Index("ix_doc_grant_user", "tenant_id", "user_id"),
        Index("ix_doc_grant_department", "tenant_id", "department_id"),
        {"schema": SCHEMA, "comment": "文档跨部门用户或部门共享"},
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    tenant_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    document_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    user_id: Mapped[int | None] = mapped_column(BigInteger)
    department_id: Mapped[int | None] = mapped_column(BigInteger)
    permission: Mapped[str] = mapped_column(String(16), nullable=False)
    granted_by: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP"),
        server_onupdate=FetchedValue(),
    )
