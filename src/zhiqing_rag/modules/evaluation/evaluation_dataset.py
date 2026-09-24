"""版本化评估数据集 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    FetchedValue,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class EvaluationDataset(Base):
    """版本化评估数据集"""

    __tablename__ = "evaluation_dataset"
    __table_args__ = (
        CheckConstraint("status IN ('DRAFT','FROZEN','ARCHIVED')"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "code", "version"),
        ForeignKeyConstraint(
            ["tenant_id", "kb_id"],
            [f"{SCHEMA}.knowledge_base.tenant_id", f"{SCHEMA}.knowledge_base.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "created_by"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        CheckConstraint("dataset_sha256 IS NULL OR dataset_sha256 ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("status='DRAFT' OR (dataset_sha256 IS NOT NULL AND frozen_at IS NOT NULL)"),
        {"schema": SCHEMA, "comment": "版本化评估数据集"},
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
    kb_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="首期一个数据集对应一个知识库"
    )
    code: Mapped[str] = mapped_column(String(100), nullable=False, comment="数据集代码")
    version: Mapped[str] = mapped_column(String(50), nullable=False, comment="数据集版本")
    name: Mapped[str] = mapped_column(String(200), nullable=False, comment="名称")
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=text("'DRAFT'"),
        comment="草稿可编辑，冻结版本供评估",
    )
    dataset_sha256: Mapped[str | None] = mapped_column(String(64), comment="冻结题目集合哈希")
    created_by: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="创建者成员")
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="冻结时间")
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
