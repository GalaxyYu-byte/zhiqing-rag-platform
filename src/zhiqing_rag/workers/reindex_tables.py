"""只为实际受旧表头切块影响的当前文档创建新的索引批次。"""

import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.documents import (
    BackgroundTask,
    Document,
    DocumentChunk,
    DocumentRevision,
    EmbeddingProfile,
    IndexGeneration,
)
from zhiqing_rag.workers.backfill_graph import make_engine
from zhiqing_rag.workers.index_snapshot import chunk_fingerprint, chunk_record
from zhiqing_rag.workers.ingestion import download_source, parse_chunks
from zhiqing_rag.workers.publication import lock_work_knowledge_base, verify_access

REINDEX_VERSION = "chunking-v3-table-header"
ACTIVE_STATUSES = ("PENDING", "RETRY_WAIT", "RUNNING")


def locked_document(session: Session, tenant_id: int, document_id: int) -> Document:
    lock_work_knowledge_base(session, tenant_id, document_id)
    # 与上传、删除、发布统一为知识库 -> 任务 -> 文档的锁顺序。
    session.scalars(
        select(BackgroundTask)
        .where(BackgroundTask.tenant_id == tenant_id, BackgroundTask.document_id == document_id)
        .order_by(BackgroundTask.id)
        .with_for_update()
    ).all()
    document = session.scalar(
        select(Document)
        .where(
            Document.tenant_id == tenant_id,
            Document.id == document_id,
            Document.status == "PUBLISHED",
            Document.deleted_at.is_(None),
        )
        .with_for_update()
    )
    if document is None:
        raise ValueError("PUBLISHED_DOCUMENT_REQUIRED")
    verify_access(session, document, document.desired_revision_id)
    return document


def current_chunks(session: Session, document: Document):
    generation = session.scalar(
        select(IndexGeneration).where(
            IndexGeneration.tenant_id == document.tenant_id,
            IndexGeneration.id == document.active_generation_id,
            IndexGeneration.document_id == document.id,
            IndexGeneration.revision_id == document.desired_revision_id,
            IndexGeneration.status == "READY",
        )
    )
    if generation is None:
        raise ValueError("REVISION_UNAVAILABLE")
    chunks = session.scalars(
        select(DocumentChunk)
        .where(
            DocumentChunk.tenant_id == document.tenant_id,
            DocumentChunk.generation_id == generation.id,
        )
        .order_by(DocumentChunk.chunk_index)
    ).all()
    if len(chunks) != generation.chunk_count:
        raise ValueError("CHUNK_INTEGRITY_INVALID")
    chunk_fingerprint(chunks)
    return generation, chunks


def capture(engine, settings: Settings, tenant_id: int, document_id: int):
    """事务仅捕获一致快照；下载与解析在事务关闭后进行。"""
    with Session(engine) as session, session.begin():
        document = locked_document(session, tenant_id, document_id)
        generation, chunks = current_chunks(session, document)
        revision = session.scalar(
            select(DocumentRevision).where(
                DocumentRevision.tenant_id == tenant_id,
                DocumentRevision.document_id == document_id,
                DocumentRevision.id == generation.revision_id,
            )
        )
        profile = session.get(EmbeddingProfile, generation.embedding_profile_id)
        if revision is None:
            raise ValueError("REVISION_UNAVAILABLE")
        if (
            profile is None
            or profile.provider != "dashscope"
            or profile.model_name != settings.embedding_model
            or profile.dimensions != settings.embedding_dimensions
            or profile.dimensions != 1024
        ):
            raise ValueError("MODEL_CONFIG_CHANGED")
        item = {
            "tenant_id": tenant_id,
            "document_id": document_id,
            "title": document.title,
            "revision_id": generation.revision_id,
            "base_generation_id": generation.id,
            "base_chunk_count": len(chunks),
            "base_fingerprint": chunk_fingerprint(chunks),
            "chunker_version": REINDEX_VERSION,
        }
        context = {
            key: getattr(revision, key)
            for key in ("bucket", "object_key", "original_filename", "file_size", "file_sha256")
        }
        context["chunk_config"] = dict(generation.chunk_config)
        # 已加载的输入供离线比较；不向报告暴露对象存储定位。
        session.expunge_all()
        return item, tuple(chunks), context


def compare_table_chunks(old_chunks, new_chunks) -> dict:
    """仅诊断当前已修复的稀疏 XLSX 前言/列头问题，其他差异不触发重建。"""
    old = [chunk_record(chunk) for chunk in old_chunks]
    new = [chunk_record(chunk) for chunk in new_chunks]
    corrected_refs = {
        ref
        for chunk in new
        if chunk["metadata"].get("kind") == "table"
        and chunk["metadata"].get("table_header_row", 1) > 1
        for ref in chunk["metadata"].get("source_refs", [])
    }
    affected = []
    for ref in sorted(corrected_refs):
        before = [
            row
            for row in old
            if row["metadata"].get("kind") == "table"
            and ref in row["metadata"].get("source_refs", [])
        ]
        after = [
            row
            for row in new
            if row["metadata"].get("kind") == "table"
            and ref in row["metadata"].get("source_refs", [])
        ]

        # 序号会受前面表格切分变化影响；不能仅因此判断后面的表格损坏。
        def comparable(rows):
            return [{k: v for k, v in row.items() if k != "chunk_index"} for row in rows]

        if comparable(before) != comparable(after):
            affected.append(
                {
                    "source_ref": ref,
                    "old_chunk_indices": [row["chunk_index"] for row in before],
                    "new_chunk_indices": [row["chunk_index"] for row in after],
                    "header_rows": sorted(
                        {row["metadata"].get("table_header_row", 1) for row in after}
                    ),
                }
            )
    return {
        "needs_reindex": bool(affected),
        "reason": "SPARSE_TABLE_HEADER_CHANGED" if affected else "NO_AFFECTED_TABLE_CHANGE",
        "target_chunk_count": len(new),
        "target_fingerprint": chunk_fingerprint(new_chunks),
        "affected_tables": affected,
    }


async def preview(engine, settings: Settings, tenant_id: int, document_ids: list[int]):
    if tenant_id <= 0 or not document_ids or len(set(document_ids)) > 20:
        raise ValueError("请明确指定租户与 1～20 份文档")
    if any(document_id <= 0 for document_id in document_ids):
        raise ValueError("DOCUMENT_ID_INVALID")
    result = []
    for document_id in sorted(set(document_ids)):
        item, old_chunks, context = await asyncio.to_thread(
            capture,
            engine,
            settings,
            tenant_id,
            document_id,
        )
        with TemporaryDirectory(prefix="zhiqing-table-preview-") as directory:
            path = Path(directory) / "source.tmp"
            await asyncio.to_thread(download_source, settings, context, path)
            async with asyncio.timeout(settings.ingestion_timeout_seconds):
                new_chunks = await parse_chunks(settings, context, path)
        result.append({**item, **compare_table_chunks(old_chunks, new_chunks)})
    return result


def enqueue(engine, settings: Settings, item: dict) -> tuple[int, int] | None:
    """幂等新增批次；保留修订、发布指针和所有历史分块。"""
    if not item["needs_reindex"]:
        return None
    key = (
        f"table-reindex:{item['base_generation_id']}:"
        f"{item['chunker_version']}:{item['target_fingerprint'][:24]}"
    )
    with Session(engine) as session, session.begin():
        document = locked_document(session, item["tenant_id"], item["document_id"])
        if document.desired_revision_id != item["revision_id"]:
            raise ValueError("REVISION_UNAVAILABLE")
        existing = session.scalar(
            select(BackgroundTask).where(
                BackgroundTask.tenant_id == item["tenant_id"],
                BackgroundTask.idempotency_key == key,
            )
        )
        if existing is not None:
            if document.active_generation_id not in (
                item["base_generation_id"],
                existing.generation_id,
            ):
                raise ValueError("BASE_GENERATION_CHANGED")
            return existing.generation_id, existing.id
        if document.active_generation_id != item["base_generation_id"]:
            raise ValueError("BASE_GENERATION_CHANGED")
        base, chunks = current_chunks(session, document)
        if chunk_fingerprint(chunks) != item["base_fingerprint"]:
            raise ValueError("CHUNK_INTEGRITY_INVALID")
        if (
            session.scalar(
                select(BackgroundTask.id).where(
                    BackgroundTask.tenant_id == item["tenant_id"],
                    BackgroundTask.document_id == item["document_id"],
                    BackgroundTask.task_type.in_(["INDEX", "GRAPH"]),
                    BackgroundTask.status.in_(ACTIVE_STATUSES),
                )
            )
            is not None
        ):
            raise ValueError("ACTIVE_WORK_EXISTS")
        next_no = 1 + session.scalar(
            select(func.max(IndexGeneration.generation_no)).where(
                IndexGeneration.tenant_id == item["tenant_id"],
                IndexGeneration.document_id == item["document_id"],
            )
        )
        generation = IndexGeneration(
            tenant_id=base.tenant_id,
            document_id=base.document_id,
            revision_id=base.revision_id,
            embedding_profile_id=base.embedding_profile_id,
            generation_no=next_no,
            parser_version="document-processing-v1",
            cleaner_version="cleaning-v1",
            chunker_version=REINDEX_VERSION,
            chunk_config=dict(base.chunk_config),
        )
        session.add(generation)
        session.flush()
        task = BackgroundTask(
            tenant_id=base.tenant_id,
            document_id=base.document_id,
            revision_id=base.revision_id,
            generation_id=generation.id,
            task_type="INDEX",
            stage="QUEUED",
            idempotency_key=key,
            max_attempts=settings.ingestion_max_attempts,
            payload={
                "pipeline": "document-ingestion-v1",
                "reindex_reason": item["reason"],
                "expected_active_generation_id": base.id,
                "expected_chunks_fingerprint": item["target_fingerprint"],
                "expected_chunk_count": item["target_chunk_count"],
            },
        )
        session.add(task)
        session.flush()
        return generation.id, task.id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--document-id", type=int, action="append", required=True)
    parser.add_argument("--enqueue", action="store_true", help="预览后为实际受影响文档入队")
    args = parser.parse_args()
    settings = get_settings()
    engine = make_engine(settings)
    try:
        items = asyncio.run(preview(engine, settings, args.tenant_id, args.document_id))
        if args.enqueue:
            for item in items:
                scheduled = enqueue(engine, settings, item)
                if scheduled:
                    item.update(generation_id=scheduled[0], task_id=scheduled[1])
        print(
            json.dumps(
                {"mode": "enqueue" if args.enqueue else "preview", "documents": items},
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as error:
        # 不输出连接串、对象键或提供方异常原文。
        code = str(error) if isinstance(error, ValueError) else type(error).__name__
        print(json.dumps({"error_code": code}, ensure_ascii=False))
        raise SystemExit(1) from None
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
