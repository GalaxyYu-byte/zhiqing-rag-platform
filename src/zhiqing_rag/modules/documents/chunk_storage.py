"""在调用方事务内保存完整批次，向量生成应在开启写事务前完成。"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zhiqing_rag.document_processing.embedding import EmbeddedChunk, validate_embedding

from .document import Document
from .document_chunk import DocumentChunk
from .embedding_profile import EmbeddingProfile
from .index_generation import IndexGeneration


@dataclass(frozen=True, slots=True)
class ChunkScope:
    tenant_id: int
    kb_id: int
    document_id: int
    revision_id: int
    generation_id: int


def save_embedded_chunks(
    session: Session, scope: ChunkScope, chunks: Sequence[EmbeddedChunk]
) -> tuple[int, ...]:
    """保存完整批次并置为 READY，返回按分块序号排列的 ID；不提交、不发布。"""
    chunks = tuple(chunks)
    if not chunks or [item.chunk.chunk_index for item in chunks] != list(range(len(chunks))):
        raise ValueError("完整批次必须非空，chunk_index 必须从 0 开始连续排列")
    for item in chunks:
        validate_embedding(item.embedding)
    generation = session.scalar(
        select(IndexGeneration)
        .where(
            IndexGeneration.id == scope.generation_id,
            IndexGeneration.tenant_id == scope.tenant_id,
            IndexGeneration.document_id == scope.document_id,
            IndexGeneration.revision_id == scope.revision_id,
        )
        .with_for_update()
    )
    if generation is None or generation.status != "BUILDING":
        raise ValueError("目标批次不存在、归属不匹配或不是 BUILDING")
    document = session.scalar(
        select(Document).where(
            Document.id == scope.document_id,
            Document.tenant_id == scope.tenant_id,
            Document.kb_id == scope.kb_id,
            Document.deleted_at.is_(None),
        )
    )
    if document is None:
        raise ValueError("文档与知识库归属不匹配或文档已删除")
    profile = session.get(EmbeddingProfile, generation.embedding_profile_id)
    if (
        profile is None
        or profile.provider != "dashscope"
        or profile.dimensions != 1024
        or any(item.model_name != profile.model_name for item in chunks)
    ):
        raise ValueError("向量模型与索引批次的 embedding_profile 不匹配")
    existing = session.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(
            DocumentChunk.tenant_id == scope.tenant_id,
            DocumentChunk.generation_id == scope.generation_id,
        )
    )
    if existing:
        raise ValueError("目标批次已有分块，请使用新的索引批次")
    if generation.chunk_count not in (0, len(chunks)):
        raise ValueError("分块数量与批次预期不匹配")
    rows = []
    for item in chunks:
        chunk = item.chunk
        rows.append(
            DocumentChunk(
                tenant_id=scope.tenant_id,
                kb_id=scope.kb_id,
                document_id=scope.document_id,
                revision_id=scope.revision_id,
                generation_id=scope.generation_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding_content=chunk.embedding_content,
                content_sha256=chunk.content_sha256,
                embedding=list(item.embedding),
                token_count=chunk.token_count,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                metadata_={**chunk.metadata, "source_refs": list(chunk.source_refs)},
            )
        )
    session.add_all(rows)
    # 分块先写入，数据库触发器随后检查 READY 批次的数量和向量完整性。
    session.flush()
    generation.chunk_count = len(rows)
    generation.ready_at = datetime.now(UTC)
    generation.status = "READY"
    session.flush()
    return tuple(row.id for row in rows)
