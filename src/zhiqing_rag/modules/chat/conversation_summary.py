"""长会话摘要版本 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class ConversationSummary(Base):
    """长会话摘要版本"""

    __tablename__ = "conversation_summary"
    __table_args__ = (
        CheckConstraint("summary_version > 0"),
        CheckConstraint("start_sequence > 0"),
        CheckConstraint("btrim(summary) <> ''"),
        CheckConstraint("token_count >= 0"),
        CheckConstraint("status IN ('READY','INVALIDATED')"),
        UniqueConstraint("tenant_id", "id"),
        UniqueConstraint("tenant_id", "session_id", "summary_version"),
        ForeignKeyConstraint(
            ["tenant_id", "session_id"],
            [f"{SCHEMA}.chat_session.tenant_id", f"{SCHEMA}.chat_session.id"],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "session_id", "start_sequence"],
            [
                f"{SCHEMA}.chat_message.tenant_id",
                f"{SCHEMA}.chat_message.session_id",
                f"{SCHEMA}.chat_message.sequence_no",
            ],
        ),
        ForeignKeyConstraint(
            ["tenant_id", "session_id", "end_sequence"],
            [
                f"{SCHEMA}.chat_message.tenant_id",
                f"{SCHEMA}.chat_message.session_id",
                f"{SCHEMA}.chat_message.sequence_no",
            ],
        ),
        CheckConstraint("end_sequence >= start_sequence"),
        CheckConstraint("source_sha256 ~ '^[0-9a-f]{64}$'"),
        {"schema": SCHEMA, "comment": "长会话摘要版本"},
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
    summary_version: Mapped[int] = mapped_column(Integer, nullable=False, comment="摘要版本")
    start_sequence: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="覆盖首条消息序号"
    )
    end_sequence: Mapped[int] = mapped_column(
        BigInteger, nullable=False, comment="覆盖最后消息序号"
    )
    summary: Mapped[str] = mapped_column(
        Text, nullable=False, comment="摘要内容，使用前仍需检查来源权限"
    )
    source_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False, comment="覆盖消息的内容摘要"
    )
    model_name: Mapped[str] = mapped_column(String(150), nullable=False, comment="摘要模型")
    prompt_version: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="摘要提示词版本"
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, comment="摘要 token 数")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'READY'"), comment="有效性状态"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
