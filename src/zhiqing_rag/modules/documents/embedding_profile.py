"""不可变向量模型配置 ORM 实体映射。"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, DateTime, Identity, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules.schema import SCHEMA


class EmbeddingProfile(Base):
    """记录向量供应商、模型版本、维度及不含凭据的请求配置快照。"""

    __tablename__ = "embedding_profile"
    __table_args__ = (
        CheckConstraint("dimensions = 1024"),
        CheckConstraint("distance_metric = 'cosine'"),
        CheckConstraint("jsonb_typeof(config) = 'object'"),
        {"schema": SCHEMA, "comment": "不可变向量模型配置"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), primary_key=True, comment="数据库生成的主键"
    )
    code: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, comment="模型配置唯一代码")
    provider: Mapped[str] = mapped_column(String(100), nullable=False, comment="服务提供方")
    model_name: Mapped[str] = mapped_column(String(150), nullable=False, comment="模型名称")
    model_revision: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="供应商版本或本地快照版本"
    )
    dimensions: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1024"), comment="当前 SQL 的向量维度固定 1024"
    )
    distance_metric: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=text("'cosine'"),
        comment="与当前 HNSW operator class 一致",
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        comment="请求配置快照，不保存 API Key",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="创建时间",
    )
