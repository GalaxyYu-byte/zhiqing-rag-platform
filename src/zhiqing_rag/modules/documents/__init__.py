"""文档、索引及后台处理 ORM 实体。"""

# 注册复合外键所引用的租户、知识库和部门表。
from zhiqing_rag.modules import identity as _identity  # noqa: F401
from zhiqing_rag.modules import knowledge as _knowledge  # noqa: F401

from .background_task import BackgroundTask
from .document import Document
from .document_chunk import DocumentChunk
from .document_revision import DocumentRevision
from .embedding_profile import EmbeddingProfile
from .index_generation import IndexGeneration
from .outbox_event import OutboxEvent
from .task_attempt import TaskAttempt

__all__ = [
    "Document",
    "DocumentRevision",
    "EmbeddingProfile",
    "IndexGeneration",
    "DocumentChunk",
    "BackgroundTask",
    "TaskAttempt",
    "OutboxEvent",
]
