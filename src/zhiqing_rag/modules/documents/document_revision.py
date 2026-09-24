"""不可变文档文件修订 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class DocumentRevision(Base):
    """文档文件的不可变修订记录，文件正文存放于对象存储。"""

    __tablename__ = "document_revision"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "document_id", "id"),
        UniqueConstraint("tenant_id", "document_id", "revision_no"),
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            [f"{SCHEMA}.document.tenant_id", f"{SCHEMA}.document.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "uploaded_by"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "source_revision_id"],
            [
                f"{SCHEMA}.document_revision.tenant_id",
                f"{SCHEMA}.document_revision.document_id",
                f"{SCHEMA}.document_revision.id",
            ],
        ),
        CheckConstraint("revision_no > 0"),
        CheckConstraint("file_size > 0"),
        CheckConstraint("file_sha256 ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("btrim(bucket) <> '' AND btrim(object_key) <> ''"),
        CheckConstraint("source_revision_id IS NULL OR source_revision_id <> id"),
        Index("ix_revision_object", "bucket", "object_key"),
        {"schema": SCHEMA, "comment": "不可变文档文件修订"},
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
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="文档内递增修订号")
    original_filename: Mapped[str] = mapped_column(
        String(500), nullable=False, comment="原始文件名"
    )
    mime_type: Mapped[str] = mapped_column(String(150), nullable=False, comment="文件 MIME 类型")
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="字节数")
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False, comment="文件 SHA-256")
    bucket: Mapped[str] = mapped_column(String(63), nullable=False, comment="MinIO 桶名")
    object_key: Mapped[str] = mapped_column(
        Text, nullable=False, comment="项目/租户/知识库/文档/修订范围内对象键"
    )
    uploaded_by: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="上传者成员 ID")
    source_revision_id: Mapped[int | None] = mapped_column(
        BigInteger, comment="恢复操作的来源修订"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
