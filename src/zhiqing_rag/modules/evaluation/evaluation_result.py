"""评估执行中的逐题结果 ORM 实体映射。"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class EvaluationResult(Base):
    """评估执行中的逐题结果"""

    __tablename__ = "evaluation_result"
    __table_args__ = (
        CheckConstraint("status IN ('SUCCEEDED','FAILED','SKIPPED')"),
        CheckConstraint("jsonb_typeof(retrieved_evidence)='array'"),
        CheckConstraint("jsonb_typeof(metrics)='object'"),
        CheckConstraint("latency_ms >= 0"),
        CheckConstraint("input_tokens >= 0"),
        CheckConstraint("output_tokens >= 0"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "run_id", "case_id"),
        ForeignKeyConstraint(
            ["tenant_id", "dataset_id", "run_id"],
            [
                f"{SCHEMA}.evaluation_run.tenant_id",
                f"{SCHEMA}.evaluation_run.dataset_id",
                f"{SCHEMA}.evaluation_run.id",
            ],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "dataset_id", "case_id"],
            [
                f"{SCHEMA}.evaluation_case.tenant_id",
                f"{SCHEMA}.evaluation_case.dataset_id",
                f"{SCHEMA}.evaluation_case.id",
            ],
        ),
        Index("ix_eval_result_case", "tenant_id", "dataset_id", "case_id"),
        {"schema": SCHEMA, "comment": "评估执行中的逐题结果"},
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
    dataset_id: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="数据集版本，保障运行与题目一致"
    )
    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="评估运行")
    case_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="题目")
    status: Mapped[str] = mapped_column(String(16), nullable=False, comment="单题状态")
    answer: Mapped[str | None] = mapped_column(Text, comment="实际回答")
    retrieved_evidence: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'[]'::jsonb"),
        comment="召回证据快照",
    )
    metrics: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="逐题质量指标",
    )
    latency_ms: Mapped[int | None] = mapped_column(BigInteger, comment="端到端耗时")
    input_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="输入 token"
    )
    output_tokens: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="输出 token"
    )
    error_code: Mapped[str | None] = mapped_column(String(100), comment="失败错误码")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
