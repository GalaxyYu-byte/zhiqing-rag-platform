"""文档、文档分块及索引任务实体。

本模块对应三个文档处理阶段使用的表：

* ``kb_document``：原始文件和索引状态；
* ``kb_document_version``：每次上传或重建产生的候选文件版本；
* ``kb_doc_chunk``：切分后的文本块和向量；
* ``kb_index_task``：异步索引任务及重试状态。

实体中的 ``doc_id``、``kb_id`` 没有声明 ForeignKey，是因为当前 schema.sql
也没有创建外键约束。这样可以保留现有数据库设计，同时避免迁移工具意外
生成新的外键约束。
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class Document(Base):
    """知识库文档实体，对应 ``kb_document`` 表。

    一条文档记录代表上传到 MinIO 的一个原始文件。文档的解析、切分、Embedding
    和向量入库由后台索引任务完成，``status`` 用于向接口层报告处理进度。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_document"
    __table_args__ = (
        # 文档列表通常按知识库过滤；软删除记录不参与正常查询。
        Index(
            "idx_doc_kb_id",
            "kb_id",
            postgresql_where=text("is_deleted = FALSE"),
        ),
        # 后台任务和前端进度查询通常按 status 筛选未删除文档。
        Index(
            "idx_doc_status",
            "status",
            postgresql_where=text("is_deleted = FALSE"),
        ),
    )

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 文档所属知识库 ID。
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 用户上传时的原始文件名，例如 manual.pdf。
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # 文件类型，约定值为 PDF、DOCX、MD 或 TXT。
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # 原始文件大小，单位为字节。
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 文件在 MinIO Bucket 中的对象路径，不是本地文件系统路径。
    minio_path: Mapped[str] = mapped_column(String(500), nullable=False)
    # 索引状态：PENDING（待处理）、PROCESSING（处理中）、DONE（完成）、
    # FAILED（失败）。新记录默认为 PENDING。
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'PENDING'")
    )
    # 最近一次索引失败的错误信息；成功或尚未失败时为空。
    error_msg: Mapped[str | None] = mapped_column(Text)
    # 最近一次索引生成的分块数量，默认 0。
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 本次文档向量化估算消耗的 Token 数，默认 0。
    token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 文档索引版本。每次重建索引前递增，旧分块通过该值识别和清理。
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    # 上传者用户 ID，用于审计和权限校验。
    uploaded_by: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 文件上传时间，由数据库自动写入。
    uploaded_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    # 最近一次索引成功完成的时间；未完成索引时为空。
    indexed_at: Mapped[datetime | None] = mapped_column()
    # 软删除标记。删除文档时保留原始记录和索引历史，正常查询过滤掉它。
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )


class DocumentVersion(Base):
    """文档文件版本；只有 READY 版本才能切换为正式版本。"""

    __tablename__ = "kb_document_version"
    __table_args__ = (
        UniqueConstraint("doc_id", "version", name="uq_doc_version"),
        Index("idx_doc_version_status", "doc_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64))
    minio_path: Mapped[str] = mapped_column(String(500), nullable=False)
    # 版本产生方式：UPLOAD、UPDATE、REINDEX 或 RESTORE。
    operation_type: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'UPLOAD'")
    )
    # RESTORE 版本所引用的历史版本号；其他操作为空。
    source_version: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'PENDING'")
    )
    uploaded_by: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    indexed_at: Mapped[datetime | None] = mapped_column()
    error_msg: Mapped[str | None] = mapped_column(Text)


class DocChunk(Base):
    """文档分块实体，对应 ``kb_doc_chunk`` 表。

    这是 RAG 检索的核心实体。一条记录是一个可独立召回的文本块，同时保存
    原文、全文检索字段和 1024 维向量。``content_tsv`` 由数据库触发器根据
    ``content`` 自动生成，应用代码通常不需要手动赋值。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_doc_chunk"
    __table_args__ = (
        # 至少执行一次的任务可能重复写入；业务唯一键让 Upsert 保持幂等。
        UniqueConstraint(
            "doc_id",
            "doc_version",
            "chunk_index",
            name="uq_chunk_doc_version_index",
        ),
        # HNSW 向量索引，使用余弦距离算子，服务向量相似度检索。
        # m 和 ef_construction 与 schema.sql 中的索引参数保持一致。
        Index(
            "idx_chunk_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 128},
        ),
        # GIN 全文索引，服务关键词检索；中文全文检索效果有限，主要依靠向量检索。
        Index(
            "idx_chunk_content_tsv",
            "content_tsv",
            postgresql_using="gin",
        ),
        # 多租户检索时按知识库过滤，文档详情查询时按文档过滤。
        Index("idx_chunk_kb_id", "kb_id"),
        Index("idx_chunk_doc_id", "doc_id"),
    )

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 所属文档 ID。
    doc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 冗余保存知识库 ID，检索时可以直接按知识库过滤，减少 JOIN。
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 分块在原文中的顺序，从 0 开始；用于按原文顺序重组上下文。
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    # 分块原文，是发送给 Embedding 模型和大模型的主要文本内容。
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # PostgreSQL TSVECTOR 全文检索向量，由 trigger_chunk_tsv 触发器自动维护。
    content_tsv: Mapped[object | None] = mapped_column(TSVECTOR)
    # 1024 维 Embedding 向量，对应 schema.sql 中的 VECTOR(1024)。
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    # 文本块所在页码，主要用于 PDF；无法确定时为空。
    page_num: Mapped[int | None] = mapped_column(Integer)
    # 文本块所在章节标题，无法识别章节时为空。
    section_title: Mapped[str | None] = mapped_column(String(500))
    # 当前文本块的 Token 数估算值，默认 0。
    token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 该分块所属的文档版本，用于重建索引时删除旧版本分块。
    doc_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # 分块写入数据库的时间。
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )


class IndexTask(Base):
    """异步索引任务实体，对应 ``kb_index_task`` 表。

    文档上传接口只负责创建任务并快速返回，后台线程池负责执行真正的索引工作。
    任务失败后可以依据 ``retry_count`` 和 ``max_retry`` 决定是否重试，具体调度
    逻辑放在 Service 层而不是实体类中。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_index_task"
    __table_args__ = (
        # 按状态和创建时间查找待处理任务，便于任务调度器取出任务。
        Index("idx_task_status", "status", "created_at"),
        # 按文档查询索引历史和当前任务。
        Index("idx_task_doc_id", "doc_id"),
        # 同一个文档同一时间最多保留一个活跃索引任务。
        Index(
            "uq_task_active_doc",
            "doc_id",
            unique=True,
            postgresql_where=text("status IN ('PENDING', 'PROCESSING')"),
        ),
        CheckConstraint(
            "progress_percent BETWEEN 0 AND 100",
            name="ck_task_progress_percent",
        ),
    )

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 本次索引任务对应的文档 ID。
    doc_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 任务正在构建的目标文档版本；成功前不会修改文档正式版本。
    doc_version: Mapped[int] = mapped_column(Integer, nullable=False)
    # 任务类型：INDEX 首次索引，REINDEX 原文件重建，UPDATE 新文件更新，
    # RESTORE 从历史文件创建新的正式版本。
    task_type: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'INDEX'")
    )
    # 任务状态，默认 PENDING；执行过程中通常流转为 PROCESSING、DONE 或 FAILED。
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'PENDING'")
    )
    # 当前处理阶段，用于展示精确进度和定位卡点。
    stage: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("'PENDING'")
    )
    progress_percent: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    embedded_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    persisted_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cache_hit_chunks: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    total_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 已经重试的次数，默认 0。
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # 允许的最大重试次数，默认 3；达到上限后应进入最终失败状态。
    max_retry: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3")
    )
    # 最近一次任务失败原因；任务成功或尚未失败时为空。
    error_msg: Mapped[str | None] = mapped_column(Text)
    # 多节点 Worker 归属和租约；租约失效后其他 Worker 可以接管任务。
    worker_id: Mapped[str | None] = mapped_column(String(200))
    heartbeat_at: Mapped[datetime | None] = mapped_column()
    lease_expires_at: Mapped[datetime | None] = mapped_column()
    # 任务创建时间。
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    # 任务实际开始执行的时间，排队期间为空。
    started_at: Mapped[datetime | None] = mapped_column()
    # 任务结束时间，未结束时为空。
    finished_at: Mapped[datetime | None] = mapped_column()
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
