"""全局用户身份 ORM 实体映射。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    FetchedValue,
    Identity,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class User(Base):
    """全局用户身份"""

    __tablename__ = "user_account"
    __table_args__ = (
        CheckConstraint("email = lower(btrim(email)) AND email <> ''"),
        CheckConstraint("btrim(password_hash) <> '' AND btrim(display_name) <> ''"),
        {"schema": SCHEMA, "comment": "全局用户身份"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True, comment="数据库生成的主键"
    )
    email: Mapped[str] = mapped_column(
        String(320), nullable=False, unique=True, comment="已规范化为小写的邮箱"
    )
    password_hash: Mapped[str] = mapped_column(
        Text, nullable=False, comment="密码哈希，禁止存储明文密码"
    )
    display_name: Mapped[str] = mapped_column(String(100), nullable=False, comment="显示名称")
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true"), comment="账号是否启用"
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
