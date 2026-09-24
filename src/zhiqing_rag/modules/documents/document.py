"""业务文档及其当前发布指针 ORM 实体映射。"""

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class Document(Base):
    """知识库中的业务文档，维护期望修订与当前可检索索引批次。"""

    __tablename__ = "document"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "kb_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "kb_id"],
            [f"{SCHEMA}.knowledge_base.tenant_id", f"{SCHEMA}.knowledge_base.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "department_id"],
            [f"{SCHEMA}.department.tenant_id", f"{SCHEMA}.department.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "created_by"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "id", "desired_revision_id"],
            [
                f"{SCHEMA}.document_revision.tenant_id",
                f"{SCHEMA}.document_revision.document_id",
                f"{SCHEMA}.document_revision.id",
            ],
            name="fk_document_desired_revision",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "id", "active_generation_id"],
            [
                f"{SCHEMA}.index_generation.tenant_id",
                f"{SCHEMA}.index_generation.document_id",
                f"{SCHEMA}.index_generation.id",
            ],
            name="fk_document_active_generation",
        ),
        CheckConstraint("btrim(title) <> ''"),
        CheckConstraint("confidentiality BETWEEN 0 AND 2"),
        CheckConstraint(
            "business_status IN ('UNKNOWN','DRAFT','EFFECTIVE','ARCHIVED','REPEALED')"
        ),
        CheckConstraint("status IN ('DRAFT','PUBLISHED','WITHDRAWN','DELETED')"),
        CheckConstraint(
            "status <> 'PUBLISHED' OR "
            "(desired_revision_id IS NOT NULL AND active_generation_id IS NOT NULL)"
        ),
        CheckConstraint("(status = 'DELETED') = (deleted_at IS NOT NULL)"),
        CheckConstraint("jsonb_typeof(metadata) = 'object'"),
        Index(
            "uq_document_code",
            "tenant_id",
            "kb_id",
            "document_code",
            unique=True,
            postgresql_where=text("document_code IS NOT NULL AND deleted_at IS NULL"),
        ),
        Index("ix_document_listing", "tenant_id", "kb_id", "status", text("created_at DESC")),
        Index(
            "ix_document_metadata", "tenant_id", "department_id", "business_status", "event_date"
        ),
        {"schema": SCHEMA, "comment": "业务文档与发布指针"},
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
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="所属知识库")
    document_code: Mapped[str | None] = mapped_column(String(100), comment="业务编号")
    title: Mapped[str] = mapped_column(String(500), nullable=False, comment="文档标题")
    department_id: Mapped[int | None] = mapped_column(BigInteger, comment="归属部门")
    confidentiality: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        server_default=text("2"),
        comment="密级：0 内部公开、1 部门内部、2 机密；默认最严格",
    )
    business_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=text("'UNKNOWN'"),
        comment="业务状态，与索引状态独立",
    )
    event_date: Mapped[date | None] = mapped_column(Date, comment="业务生效或事件日期")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'DRAFT'"), comment="文档可见性状态"
    )
    desired_revision_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="期望发布的文件修订"
    )
    active_generation_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="当前可检索批次"
    )
    created_by: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="创建者成员 ID")
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="软删除时间")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="可扩展元数据，授权字段使用显式列",
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
