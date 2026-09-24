"""Neo4j 图谱投影构建记录 ORM 实体映射。"""

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
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class GraphProjection(Base):
    """Neo4j 图谱投影构建记录"""

    __tablename__ = "graph_projection"
    __table_args__ = (
        CheckConstraint("projection_version > 0"),
        CheckConstraint("status IN ('BUILDING','READY','FAILED','STALE')"),
        CheckConstraint("entity_count >= 0"),
        CheckConstraint("relation_count >= 0"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "generation_id", "projection_version"),
        ForeignKeyConstraint(
            ["tenant_id", "document_id", "revision_id", "generation_id"],
            [
                f"{SCHEMA}.index_generation.tenant_id",
                f"{SCHEMA}.index_generation.document_id",
                f"{SCHEMA}.index_generation.revision_id",
                f"{SCHEMA}.index_generation.id",
            ],
        ),
        CheckConstraint("status <> 'READY' OR ready_at IS NOT NULL"),
        Index(
            "uq_graph_ready_generation",
            "tenant_id",
            "generation_id",
            unique=True,
            postgresql_where=text("status='READY'"),
        ),
        {"schema": SCHEMA, "comment": "Neo4j 图谱投影构建记录"},
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
    document_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="来源文档")
    revision_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="来源修订")
    generation_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="来源索引批次")
    projection_version: Mapped[int] = mapped_column(Integer, nullable=False, comment="图谱构建版本")
    graph_namespace: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        server_default=text("'zhiqing-rag-platform'"),
        comment="图数据库命名空间",
    )
    extractor_version: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="抽取规则版本"
    )
    model_name: Mapped[str] = mapped_column(String(150), nullable=False, comment="抽取模型")
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=text("'BUILDING'"),
        comment="构建状态",
    )
    entity_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="实体数量"
    )
    relation_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="关系数量"
    )
    ready_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="就绪时间")
    error_code: Mapped[str | None] = mapped_column(String(100), comment="错误码")
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
