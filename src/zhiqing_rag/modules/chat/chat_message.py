"""会话内持久消息 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class ChatMessage(Base):
    """会话内持久消息"""

    __tablename__ = "chat_message"
    __table_args__ = (
        CheckConstraint("role IN ('USER','ASSISTANT')"),
        CheckConstraint("sequence_no > 0"),
        CheckConstraint("token_count >= 0"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "session_id", "sequence_no"),
        UniqueConstraint("tenant_id", "run_id", "role"),
        ForeignKeyConstraint(
            ["tenant_id", "session_id", "run_id"],
            [
                f"{SCHEMA}.chat_run.tenant_id",
                f"{SCHEMA}.chat_run.session_id",
                f"{SCHEMA}.chat_run.id",
            ],
        ),
        {"schema": SCHEMA, "comment": "会话内持久消息"},
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
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="消息角色")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息正文")
    sequence_no: Mapped[int] = mapped_column(BigInteger, nullable=False, comment="会话内递增序号")
    token_count: Mapped[int] = mapped_column(
        BigInteger, nullable=False, server_default=text("0"), comment="正文 token 数"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
