"""文档索引构建批次 ORM 实体映射。"""

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
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class IndexGeneration(Base):
    """一次不可变配置的索引构建批次，READY 后作为文档发布候选。"""

    __tablename__ = "index_generation"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "document_id", "id"),
        UniqueConstraint("tenant_id", "document_id", "revision_id", "id"),
        UniqueConstraint("tenant_id", "document_id", "generation_no"),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "revision_id"],
            [
                f"{SCHEMA}.document_revision.tenant_id",
                f"{SCHEMA}.document_revision.document_id",
                f"{SCHEMA}.document_revision.id",
            ],
        ),
        CheckConstraint("generation_no > 0"),
        CheckConstraint("jsonb_typeof(chunk_config) = 'object'"),
        CheckConstraint("status IN ('BUILDING','READY','FAILED','CANCELLED')"),
        CheckConstraint("chunk_count >= 0"),
        CheckConstraint("status <> 'READY' OR (chunk_count > 0 AND ready_at IS NOT NULL)"),
        Index("ix_generation_revision", "tenant_id", "document_id", "revision_id"),
        Index("ix_generation_profile", "embedding_profile_id"),
        {"schema": SCHEMA, "comment": "文档索引构建批次"},
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
    document_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="业务文档")
    revision_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="文件修订")
    embedding_profile_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.embedding_profile.id"),
        nullable=False,
        comment="本批次模型配置",
    )
    generation_no: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="文档内递增构建批次号"
    )
    parser_version: Mapped[str] = mapped_column(String(100), nullable=False, comment="解析器版本")
    cleaner_version: Mapped[str] = mapped_column(String(100), nullable=False, comment="清洗规则版本")
    chunker_version: Mapped[str] = mapped_column(String(100), nullable=False, comment="分块策略版本")
    chunk_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="分块参数快照",
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'BUILDING'"), comment="批次完整性状态"
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"), comment="批次预期分块总数"
    )
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="完成构建时间")
    first_published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="首次发布记录，当前发布由 document 指针决定"
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
