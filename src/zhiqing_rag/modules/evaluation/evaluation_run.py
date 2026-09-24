"""一次可复现评估执行 ORM 实体映射。"""

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
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class EvaluationRun(Base):
    """一次可复现评估执行"""

    __tablename__ = "evaluation_run"
    __table_args__ = (
        CheckConstraint("jsonb_typeof(config_snapshot)='object'"),
        CheckConstraint("status IN ('PENDING','RUNNING','SUCCEEDED','FAILED','CANCELLED')"),
        CheckConstraint("total_cases > 0"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "dataset_id", "id"),
        ForeignKeyConstraint(
            ["tenant_id", "dataset_id"],
            [
                f"{SCHEMA}.evaluation_dataset.tenant_id",
                f"{SCHEMA}.evaluation_dataset.id",
            ],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "created_by"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        CheckConstraint("dataset_sha256 ~ '^[0-9a-f]{64}$'"),
        CheckConstraint("completed_cases BETWEEN 0 AND total_cases"),
        CheckConstraint(
            "(status IN ('SUCCEEDED','FAILED','CANCELLED')) = (finished_at IS NOT NULL)"
        ),
        CheckConstraint("finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at"),
        {"schema": SCHEMA, "comment": "一次可复现评估执行"},
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
    dataset_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="冻结数据集版本")
    created_by: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="发起者成员")
    dataset_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="启动时的数据集哈希"
    )
    code_revision: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="代码提交号或发布版本"
    )
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, comment="模型、索引、权限范围与参数快照，禁止密钥"
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'PENDING'"), comment="执行状态"
    )
    total_cases: Mapped[int] = mapped_column(Integer, nullable=False, comment="题目总数")
    completed_cases: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"), comment="已完成数"
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), comment="开始时间")
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), comment="结束时间"
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
