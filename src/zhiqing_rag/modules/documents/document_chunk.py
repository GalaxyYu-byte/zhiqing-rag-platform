"""文档分块正文与向量 ORM 实体映射。"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
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
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class DocumentChunk(Base):
    """索引批次中的检索分块，保存正文、向量及来源定位信息。"""

    __tablename__ = "document_chunk"
    __table_args__ = (
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "generation_id", "chunk_index"),
        ForeignKeyConstraint(
            ["tenant_id", "kb_id", "document_id"],
            [f"{SCHEMA}.document.tenant_id", f"{SCHEMA}.document.kb_id", f"{SCHEMA}.document.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "revision_id", "generation_id"],
            [
                f"{SCHEMA}.index_generation.tenant_id",
                f"{SCHEMA}.index_generation.document_id",
                f"{SCHEMA}.index_generation.revision_id",
                f"{SCHEMA}.index_generation.id",
            ],
        ),
        CheckConstraint("chunk_index >= 0"),
        CheckConstraint("btrim(content) <> ''"),
        CheckConstraint("content_sha256 ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("token_count >= 0"),
        CheckConstraint("page_number IS NULL OR page_number > 0"),
        CheckConstraint("start_offset IS NULL OR start_offset >= 0"),
        CheckConstraint(
            "(start_offset IS NULL AND end_offset IS NULL) OR "
            "(start_offset IS NOT NULL AND end_offset IS NOT NULL AND end_offset >= start_offset)"
        ),
        CheckConstraint("jsonb_typeof(metadata) = 'object'"),
        Index("ix_chunk_scope", "tenant_id", "kb_id", "document_id", "generation_id"),
        Index(
            "ix_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "ix_chunk_bm25",
            "id",
            "content",
            "section_title",
            "tenant_id",
            "kb_id",
            "document_id",
            "generation_id",
            postgresql_using="bm25",
            postgresql_with={"key_field": "id"},
        ),
        {"schema": SCHEMA, "comment": "分块正文与 1024 维向量"},
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
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="知识库，服务于检索过滤")
    document_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="文档")
    revision_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="文件修订")
    generation_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="索引批次")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, comment="从 0 开始的块序号")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="原始块正文")
    embedding_content: Mapped[str] = mapped_column(
        Text, nullable=False, comment="向量化的实际输入，含必要结构上下文"
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, comment="正文 SHA-256")
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1024), comment="向量，构建期间可为空，发布前完整性校验要求非空"
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="块 token 数")
    page_number: Mapped[int | None] = mapped_column(Integer, comment="页码，从 1 开始")
    section_title: Mapped[str | None] = mapped_column(Text, comment="章节标题")
    start_offset: Mapped[int | None] = mapped_column(Integer, comment="正文起始偏移")
    end_offset: Mapped[int | None] = mapped_column(Integer, comment="正文结束偏移")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="表格位置等来源元数据",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
