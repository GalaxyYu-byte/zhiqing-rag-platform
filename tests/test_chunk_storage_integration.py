"""真实 PostgreSQL 入库与事务回滚验证，默认跳过且所有测试数据回滚。"""

import os
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_embedding import make_chunks

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.document_processing.embedding import EmbeddedChunk
from zhiqing_rag.modules.documents import (
    Document,
    DocumentChunk,
    DocumentRevision,
    EmbeddingProfile,
    IndexGeneration,
)
from zhiqing_rag.modules.documents.chunk_storage import ChunkScope, save_embedded_chunks
from zhiqing_rag.modules.knowledge import KnowledgeBase

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EMBEDDING_DB_INTEGRATION") != "1",
    reason="设置 RUN_EMBEDDING_DB_INTEGRATION=1 验证真实数据库；数据自动回滚",
)


@pytest.fixture
def target():
    settings = get_settings()
    engine = create_engine(settings.database_url, connect_args={"connect_timeout": 5})
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
                kb = session.scalar(
                    select(KnowledgeBase)
                    .where(KnowledgeBase.code == "QA_PUBLIC", KnowledgeBase.status == "ACTIVE")
                    .order_by(KnowledgeBase.id)
                    .limit(1)
                )
                profile = session.scalar(
                    select(EmbeddingProfile)
                    .where(
                        EmbeddingProfile.provider == "dashscope",
                        EmbeddingProfile.model_name == "text-embedding-v3",
                        EmbeddingProfile.dimensions == 1024,
                    )
                    .limit(1)
                )
                assert kb is not None and profile is not None, "需先准备知识库和模型调试数据"
                document = Document(
                    tenant_id=kb.tenant_id, kb_id=kb.id, title="回滚测试", created_by=kb.created_by
                )
                session.add(document)
                session.flush()
                revision = DocumentRevision(
                    tenant_id=kb.tenant_id,
                    document_id=document.id,
                    revision_no=1,
                    original_filename="rollback.md",
                    mime_type="text/markdown",
                    file_size=1,
                    file_sha256="0" * 64,
                    bucket=settings.minio_bucket,
                    object_key=f"rollback-test/{uuid4().hex}",
                    uploaded_by=kb.created_by,
                )
                session.add(revision)
                session.flush()
                generation = IndexGeneration(
                    tenant_id=kb.tenant_id,
                    document_id=document.id,
                    revision_id=revision.id,
                    embedding_profile_id=profile.id,
                    generation_no=1,
                    parser_version="test",
                    cleaner_version="test",
                    chunker_version="test",
                    chunk_config={},
                    chunk_count=20,
                )
                session.add(generation)
                session.flush()
                scope = ChunkScope(kb.tenant_id, kb.id, document.id, revision.id, generation.id)
                yield session, scope
        finally:
            transaction.rollback()
    engine.dispose()


def embedded():
    return tuple(
        EmbeddedChunk(chunk, (float(chunk.chunk_index + 1),) * 1024, "text-embedding-v3")
        for chunk in make_chunks()
    )


def test_stores_20_chunks_preserves_mapping_and_marks_ready(target):
    session, scope = target
    items = embedded()
    ids = save_embedded_chunks(session, scope, items)
    session.expire_all()
    rows = session.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.generation_id == scope.generation_id)
        .order_by(DocumentChunk.chunk_index)
    ).all()
    assert len(ids) == len(rows) == 20
    for row, item in zip(rows, items, strict=True):
        assert row.content == item.chunk.content
        assert row.embedding_content == item.chunk.embedding_content
        assert row.content_sha256 == item.chunk.content_sha256
        assert row.page_number == item.chunk.page_number
        assert row.section_title == item.chunk.section_title
        assert row.metadata_["source_refs"] == list(item.chunk.source_refs)
        assert row.metadata_["heading_path"] == item.chunk.metadata["heading_path"]
        assert list(row.embedding) == list(item.embedding)
    generation = session.get(IndexGeneration, scope.generation_id)
    assert generation.status == "READY" and generation.chunk_count == 20
    assert session.get(Document, scope.document_id).active_generation_id is None
    with pytest.raises(ValueError, match="BUILDING"):
        save_embedded_chunks(session, scope, items)


@pytest.mark.parametrize("invalid", ["scope", "model", "indices", "count"])
def test_rejects_invalid_target_before_writing(target, invalid):
    session, scope = target
    items = embedded()
    if invalid == "scope":
        scope = replace(scope, kb_id=99999999)
    elif invalid == "model":
        items = tuple(replace(item, model_name="other-model") for item in items)
    elif invalid == "indices":
        items = tuple(reversed(items))
    else:
        items = items[:-1]
    with pytest.raises(ValueError):
        save_embedded_chunks(session, scope, items)
    assert (
        session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.generation_id == scope.generation_id)
        )
        == 0
    )
    assert session.get(IndexGeneration, scope.generation_id).status == "BUILDING"


def test_db_failure_rolls_back_entire_chunk_batch(target):
    session, scope = target
    items = list(embedded())
    items[15] = replace(items[15], chunk=replace(items[15].chunk, content_sha256="bad-hash"))
    with pytest.raises(IntegrityError), session.begin_nested():
        save_embedded_chunks(session, scope, items)
    assert (
        session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.generation_id == scope.generation_id)
        )
        == 0
    )
    assert session.get(IndexGeneration, scope.generation_id).status == "BUILDING"
