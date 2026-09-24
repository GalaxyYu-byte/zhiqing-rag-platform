"""回答证据来源快照 ORM 实体映射。"""

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


class AnswerCitation(Base):
    """回答证据来源快照"""

    __tablename__ = "answer_citation"
    __table_args__ = (
        CheckConstraint("citation_no > 0"),
        CheckConstraint("source_generation_id > 0"),
        CheckConstraint("source_chunk_id > 0"),
        CheckConstraint("page_number > 0"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "run_id", "citation_no"),
        ForeignKeyConstraint(
            ["tenant_id", "session_id", "run_id"],
            [
                f"{SCHEMA}.chat_run.tenant_id",
                f"{SCHEMA}.chat_run.session_id",
                f"{SCHEMA}.chat_run.id",
            ],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "revision_id"],
            [
                f"{SCHEMA}.document_revision.tenant_id",
                f"{SCHEMA}.document_revision.document_id",
                f"{SCHEMA}.document_revision.id",
            ],
        ),
        CheckConstraint("content_sha256 ~ '^[0-9a-f]{64}$'"),
        Index("ix_citation_document", "tenant_id", "document_id", "revision_id"),
        {"schema": SCHEMA, "comment": "回答证据来源快照"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
        nullable=False,
        comment="数据库生成的主键",
    )
    tenant_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey(f"{SCHEMA}.tenant.id"),
        nullable=False,
        comment="所属租户",
    )
    session_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="会话")
    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="问答轮次")
    citation_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="对应 S1 等引用序号")
    document_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="来源文档，保留元数据外键"
    )
    revision_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="来源文件修订")
    source_generation_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="来源索引批次快照，允许批次后续清理"
    )
    source_chunk_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="来源 chunk ID 快照，不阻止 chunk 清理"
    )
    content_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="证据正文哈希，用于核对"
    )
    page_number: Mapped[int | None] = mapped_column(Integer, comment="证据页码")
    section_title: Mapped[str | None] = mapped_column(Text, comment="证据章节")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
