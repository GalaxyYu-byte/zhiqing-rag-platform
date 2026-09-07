"""对话会话、对话消息及回答反馈实体。

本模块对应聊天链路的三张表。会话保存用户选择的知识库范围，消息保存用户
问题和助手回答，反馈表则记录用户对回答质量的评价。会话使用字符串 UUID
作为主键，消息和反馈使用 PostgreSQL BIGSERIAL 主键。
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class ChatSession(Base):
    """对话会话实体，对应 ``kb_chat_session`` 表。

    ``kb_ids`` 按现有 SQL 设计保存 JSON 数组文本，例如 ``[1, 2, 3]``。这保留
    了当前表结构；如果未来需要频繁按知识库筛选会话，可以再单独设计关联表。
    """

    # 会话表使用字符串 UUID，而不是 BIGSERIAL。
    __tablename__ = "kb_chat_session"
    __table_args__ = (
        # 用户最近会话列表按 last_active_at 倒序排列，只索引未删除会话。
        Index(
            "idx_session_user",
            "user_id",
            text("last_active_at DESC"),
            postgresql_where=text("is_deleted = FALSE"),
        ),
    )

    # UUID 字符串主键，长度与 schema.sql 的 VARCHAR(36) 一致。
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # 会话所属用户 ID。
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 会话允许检索的知识库 ID JSON 数组，以文本形式存储。
    kb_ids: Mapped[str] = mapped_column(Text, nullable=False)
    # 会话标题，通常取第一条用户消息生成；可以为空。
    title: Mapped[str | None] = mapped_column(String(200))
    # 当前会话中的消息数量，默认 0；由业务层在新增消息时维护。
    message_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 会话创建时间。
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    # 最近一次活跃时间，用于会话列表排序。
    last_active_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    # 软删除标记，默认保留会话。
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )


class ChatMessage(Base):
    """对话消息实体，对应 ``kb_chat_message`` 表。

    ``role`` 用于区分 USER 和 ASSISTANT 消息。助手消息可以通过 ``sources``
    保存召回来源，来源结构示例为 ``docId``、``chunkId``、``pageNum``、
    ``excerpt`` 和 ``score`` 等字段。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_chat_message"
    __table_args__ = (
        # 按会话读取消息，并按创建时间保持对话顺序。
        Index("idx_message_session", "session_id", "created_at"),
    )

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 所属会话 UUID。
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    # 消息角色，约定值为 USER 或 ASSISTANT。
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    # 消息正文，用户问题或助手回答都存储在此字段。
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 助手回答引用的来源列表，使用 JSONB 便于保存结构化召回结果。
    # 用户消息通常不需要来源，因此该字段允许为空。
    sources: Mapped[list[dict[str, object]] | None] = mapped_column(JSONB)
    # 生成或处理该消息消耗的 Token 数，默认 0。
    token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 从请求开始到生成完成的耗时，单位为毫秒，默认 0。
    latency_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 对该消息的简单评价：1 表示好，-1 表示差，NULL 表示尚未评价。
    feedback: Mapped[int | None] = mapped_column(SmallInteger)
    # 消息创建时间，也用于恢复会话时的时间排序。
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )


class AnswerFeedback(Base):
    """回答反馈实体，对应 ``kb_answer_feedback`` 表。

    该表记录更完整的回答评价。一名用户对同一条消息只能提交一条反馈，
    联合唯一约束由数据库负责保证；如果用户修改评价，Service 层应更新原记录。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_answer_feedback"
    __table_args__ = (
        # 同一用户不能对同一条回答重复创建反馈记录。
        UniqueConstraint(
            "message_id",
            "user_id",
            name="uq_answer_feedback_message_user",
        ),
    )

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 被评价的助手消息 ID。
    message_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 提交反馈的用户 ID。
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 反馈结果：1 表示有用，-1 表示无用。
    feedback: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    # 用户补充的文字意见，可为空。
    comment: Mapped[str | None] = mapped_column(Text)
    # 反馈提交时间。
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
