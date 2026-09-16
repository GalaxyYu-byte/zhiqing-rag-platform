"""独立 Graph RAG 构建任务实体。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class GraphTask(Base):
    """一个文档版本对应一次可重试的候选图构建。"""

    __tablename__ = "kb_graph_task"
    __table_args__ = (
        UniqueConstraint(
            "doc_id",
            "doc_version",
            name="uq_graph_task_doc_version",
        ),
        Index("idx_graph_task_status", "status", "created_at"),
        Index("idx_graph_task_doc_id", "doc_id", "created_at"),
        Index(
            "uq_graph_task_active_doc",
            "doc_id",
            unique=True,
            postgresql_where=text("status IN ('PENDING', 'PROCESSING')"),
        ),
        CheckConstraint(
            "progress_percent BETWEEN 0 AND 100",
            name="ck_graph_task_progress_percent",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    doc_version: Mapped[int] = mapped_column(Integer, nullable=False)
    extractor_version: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'PENDING'")
    )
    stage: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("'PENDING'")
    )
    progress_percent: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    processed_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    extracted_entities: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    extracted_claims: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    max_retry: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3")
    )
    error_msg: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(String(200))
    heartbeat_at: Mapped[datetime | None] = mapped_column()
    lease_expires_at: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    started_at: Mapped[datetime | None] = mapped_column()
    finished_at: Mapped[datetime | None] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
