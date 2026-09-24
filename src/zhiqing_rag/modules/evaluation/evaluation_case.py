"""数据集中的评估题目 ORM 实体映射。"""

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
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class EvaluationCase(Base):
    """数据集中的评估题目"""

    __tablename__ = "evaluation_case"
    __table_args__ = (
        CheckConstraint("case_no > 0"),
        CheckConstraint("btrim(question) <> ''"),
        CheckConstraint("jsonb_typeof(expected_evidence) = 'array'"),
        CheckConstraint("jsonb_typeof(metadata) = 'object'"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "dataset_id", "id"),
        UniqueConstraint("tenant_id", "dataset_id", "case_no"),
        ForeignKeyConstraint(
            ["tenant_id", "dataset_id"],
            [
                f"{SCHEMA}.evaluation_dataset.tenant_id",
                f"{SCHEMA}.evaluation_dataset.id",
            ],
        ),
        {"schema": SCHEMA, "comment": "数据集中的评估题目"},
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
    dataset_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="所属数据集版本")
    case_no: Mapped[int] = mapped_column(Integer, nullable=False, comment="数据集内题号")
    question: Mapped[str] = mapped_column(Text, nullable=False, comment="问题")
    expected_answer: Mapped[str | None] = mapped_column(Text, comment="参考答案")
    expected_evidence: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment="期望证据 ID/修订的快照数组",
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        server_default=text("ARRAY[]::text[]"),
        comment="场景标签",
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="扩展评估要求",
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
