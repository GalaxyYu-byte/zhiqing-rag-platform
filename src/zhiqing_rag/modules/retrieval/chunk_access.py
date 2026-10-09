"""供后续向量、关键词和引用查询复用的已发布分块授权入口。"""

from sqlalchemy import select

from zhiqing_rag.modules.documents import Document, DocumentChunk, IndexGeneration
from zhiqing_rag.modules.knowledge.permissions import Identity, document_access_condition


def authorized_chunk_statement(identity: Identity):
    """先限制可见文档与活动批次，再由调用者追加检索条件、排序和数量。"""
    return (
        select(DocumentChunk)
        .join(
            Document,
            (DocumentChunk.tenant_id == Document.tenant_id)
            & (DocumentChunk.document_id == Document.id)
            & (DocumentChunk.kb_id == Document.kb_id)
            & (DocumentChunk.generation_id == Document.active_generation_id),
        )
        .join(
            IndexGeneration,
            (IndexGeneration.tenant_id == DocumentChunk.tenant_id)
            & (IndexGeneration.id == DocumentChunk.generation_id)
            & (IndexGeneration.document_id == Document.id)
            & (IndexGeneration.revision_id == DocumentChunk.revision_id),
        )
        .where(
            document_access_condition(identity, permission="READ", published_only=True),
            IndexGeneration.status == "READY",
        )
    )
