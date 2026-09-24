"""用户回答反馈 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    FetchedValue,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class AnswerFeedback(Base):
    """用户回答反馈"""

    __tablename__ = "answer_feedback"
    __table_args__ = (
        CheckConstraint("rating IN (-1,1)"),
        CheckConstraint("length(comment) <= 4000"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "run_id", "member_id"),
        ForeignKeyConstraint(
            ["tenant_id", "session_id", "run_id"],
            [
                f"{SCHEMA}.chat_run.tenant_id",
                f"{SCHEMA}.chat_run.session_id",
                f"{SCHEMA}.chat_run.id",
            ],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "member_id"],
            [f"{SCHEMA}.tenant_member.tenant_id", f"{SCHEMA}.tenant_member.id"],
        ),
        {"schema": SCHEMA, "comment": "用户回答反馈"},
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
    member_id: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="反馈者成员")
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False, comment="-1 无帮助，1 有帮助")
    category: Mapped[str | None] = mapped_column(String(100), comment="问题分类")
    comment: Mapped[str | None] = mapped_column(Text, comment="反馈正文")
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
