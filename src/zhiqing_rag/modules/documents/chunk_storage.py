"""在调用方事务内保存完整批次，向量生成应在开启写事务前完成。"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zhiqing_rag.document_processing.chunking import DocumentChunkDraft
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


def save_raw_chunks(
    session: Session, scope: ChunkScope, chunks: Sequence[DocumentChunkDraft]
) -> tuple[int, ...]:
    """先持久化无向量的完整分块，供图谱与向量流程并行读取。"""
    chunks = tuple(chunks)
    if not chunks or [chunk.chunk_index for chunk in chunks] != list(range(len(chunks))):
        raise ValueError("完整批次必须非空，chunk_index 必须从 0 开始连续排列")
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
    if (
        session.scalar(
            select(Document.id).where(
                Document.id == scope.document_id,
                Document.tenant_id == scope.tenant_id,
                Document.kb_id == scope.kb_id,
                Document.deleted_at.is_(None),
            )
        )
        is None
    ):
        raise ValueError("文档与知识库归属不匹配或文档已删除")
    rows = session.scalars(
        select(DocumentChunk)
        .where(
            DocumentChunk.tenant_id == scope.tenant_id,
            DocumentChunk.generation_id == scope.generation_id,
        )
        .order_by(DocumentChunk.chunk_index)
    ).all()
    if rows:
        if len(rows) != len(chunks) or any(
            row.chunk_index != chunk.chunk_index
            or row.content_sha256 != chunk.content_sha256
            or row.content != chunk.content
            or row.embedding_content != chunk.embedding_content
            for row, chunk in zip(rows, chunks, strict=True)
        ):
            raise ValueError("已保存分块与重试结果不一致")
        return tuple(row.id for row in rows)
    if generation.chunk_count not in (0, len(chunks)):
        raise ValueError("分块数量与批次预期不匹配")
    rows = [
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
            embedding=None,
            token_count=chunk.token_count,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            metadata_={**chunk.metadata, "source_refs": list(chunk.source_refs)},
        )
        for chunk in chunks
    ]
    session.add_all(rows)
    session.flush()
    generation.chunk_count = len(rows)
    session.flush()
    return tuple(row.id for row in rows)


def complete_embedded_chunks(
    session: Session, scope: ChunkScope, chunks: Sequence[EmbeddedChunk]
) -> tuple[int, ...]:
    """为已保存分块补写向量；完整性通过后才将批次置为 READY。"""
    chunks = tuple(chunks)
    if not chunks or [item.chunk.chunk_index for item in chunks] != list(range(len(chunks))):
        raise ValueError("向量批次必须非空且按 chunk_index 连续排列")
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
    if (
        generation is None
        or generation.status != "BUILDING"
        or generation.chunk_count != len(chunks)
    ):
        raise ValueError("索引批次尚未完成分块持久化")
    profile = session.get(EmbeddingProfile, generation.embedding_profile_id)
    if (
        profile is None
        or profile.provider != "dashscope"
        or profile.dimensions != 1024
        or any(item.model_name != profile.model_name for item in chunks)
    ):
        raise ValueError("向量模型与索引批次的 embedding_profile 不匹配")
    rows = session.scalars(
        select(DocumentChunk)
        .where(
            DocumentChunk.tenant_id == scope.tenant_id,
            DocumentChunk.generation_id == scope.generation_id,
        )
        .order_by(DocumentChunk.chunk_index)
        .with_for_update()
    ).all()
    if len(rows) != len(chunks):
        raise ValueError("已保存分块数量不匹配")
    for row, item in zip(rows, chunks, strict=True):
        if (
            row.chunk_index != item.chunk.chunk_index
            or row.content_sha256 != item.chunk.content_sha256
            or row.content != item.chunk.content
            or row.embedding_content != item.chunk.embedding_content
        ):
            raise ValueError("向量输入与已保存分块不匹配")
        row.embedding = list(item.embedding)
    session.flush()
    generation.ready_at = datetime.now(UTC)
    generation.status = "READY"
    session.flush()
    return tuple(row.id for row in rows)


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
