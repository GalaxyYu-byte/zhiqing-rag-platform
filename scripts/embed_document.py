"""真实模型向量化并入库；--demo-20 生成独立的 20 块测试文档。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import math
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import urllib3
from minio import Minio
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from zhiqing_rag.core.config import PROJECT_ROOT, get_settings
from zhiqing_rag.document_processing import (
    ChunkConfig,
    EmbeddingStats,
    embed_chunks,
    process_document,
)
from zhiqing_rag.document_processing.format_validation import validate_document_format
from zhiqing_rag.modules.documents import (
    Document,
    DocumentChunk,
    DocumentRevision,
    EmbeddingProfile,
    IndexGeneration,
)
from zhiqing_rag.modules.documents.chunk_storage import ChunkScope, save_embedded_chunks
from zhiqing_rag.modules.identity import Tenant, TenantMember
from zhiqing_rag.modules.knowledge import KnowledgeBase


def _demo_file(run_id: str) -> Path:
    path = PROJECT_ROOT / "data" / "embedding-demo" / f"{run_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(
            f"# 测试事项 {index + 1:02d}\n\n"
            f"这是向量入库测试文档的第 {index + 1} 项。"
            f"事项编号为 EMBED-{index + 1:02d}，办理时限为 {index + 1} 个工作日。"
            "申请人需要提交申请表和身份证明，工作人员核对材料后登记受理。"
            for index in range(20)
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--demo-20", action="store_true")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--member-id", type=int, required=True)
    args = parser.parse_args()
    settings = get_settings()
    run_id = uuid4().hex
    path = _demo_file(run_id) if args.demo_20 else args.file.resolve()
    chunk_config = ChunkConfig(min_chunk_tokens=0) if args.demo_20 else ChunkConfig()
    # 解析与上传同一份快照，避免原文件在模型调用期间被修改。
    payload = path.read_bytes()
    snapshot = PROJECT_ROOT / "data" / "embedding-inputs" / run_id / path.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(payload)
    validated = validate_document_format(snapshot)
    processed = process_document(snapshot, chunk_config=chunk_config)
    if not processed.cleaned.is_complete:
        raise ValueError("解析结果不完整，请先处理解析告警")
    if not processed.chunks or (args.demo_20 and len(processed.chunks) != 20):
        raise ValueError("文档分块数量不符合预期")
    engine = create_engine(settings.database_url, connect_args={"connect_timeout": 5})
    try:
        # 调模型前检查目标归属和模型配置，避免为错误的入库目标付费。
        with Session(engine) as session:
            kb = session.get(KnowledgeBase, args.kb_id)
            member = session.get(TenantMember, args.member_id)
            if kb is None or kb.status != "ACTIVE":
                raise ValueError("目标知识库不存在或未启用")
            tenant_id = kb.tenant_id
            tenant = session.get(Tenant, tenant_id)
            if (
                tenant is None
                or tenant.status != "ACTIVE"
                or member is None
                or member.tenant_id != tenant_id
                or member.status != "ACTIVE"
            ):
                raise ValueError("租户或成员归属不匹配、未启用")
            profile_id = session.scalar(
                select(EmbeddingProfile.id)
                .where(
                    EmbeddingProfile.provider == "dashscope",
                    EmbeddingProfile.model_name == settings.embedding_model,
                    EmbeddingProfile.dimensions == settings.embedding_dimensions,
                )
                .order_by(EmbeddingProfile.id.desc())
                .limit(1)
            )
            if profile_id is None:
                raise ValueError("数据库中没有与当前模型一致的 embedding_profile")
        print(
            f"分块数: {len(processed.chunks)}，每批至多 10 块，开始缓存查询与向量生成", flush=True
        )
        stats = EmbeddingStats()
        embedded = asyncio.run(embed_chunks(processed.chunks, settings=settings, stats=stats))
        print(f"已返回并校验 {len(embedded)} 个 EmbeddedChunk，开始上传原文件和入库", flush=True)

        endpoint = urlsplit(settings.minio_endpoint)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
            raise ValueError("MINIO_ENDPOINT 必须包含 http:// 或 https://")
        object_key = (
            f"{settings.minio_object_prefix.rstrip('/')}/{tenant_id}/{args.kb_id}"
            f"/embedding-import/{run_id}/{path.name}"
        )
        pool = urllib3.PoolManager(timeout=urllib3.Timeout(total=30), retries=False)
        try:
            minio = Minio(
                endpoint.netloc,
                access_key=settings.minio_access_key.get_secret_value(),
                secret_key=settings.minio_secret_key.get_secret_value(),
                secure=endpoint.scheme == "https",
                http_client=pool,
            )
            minio.put_object(
                settings.minio_bucket,
                object_key,
                io.BytesIO(payload),
                len(payload),
                content_type=validated.mime_type,
            )
        finally:
            pool.clear()

        with Session(engine) as session, session.begin():
            document = Document(
                tenant_id=tenant_id,
                kb_id=args.kb_id,
                title=path.stem,
                created_by=args.member_id,
                metadata_={"embedding_import_run": run_id, "demo": args.demo_20},
            )
            session.add(document)
            session.flush()
            revision = DocumentRevision(
                tenant_id=tenant_id,
                document_id=document.id,
                revision_no=1,
                original_filename=path.name,
                mime_type=validated.mime_type,
                file_size=len(payload),
                file_sha256=hashlib.sha256(payload).hexdigest(),
                bucket=settings.minio_bucket,
                object_key=object_key,
                uploaded_by=args.member_id,
            )
            session.add(revision)
            session.flush()
            document.desired_revision_id = revision.id
            generation = IndexGeneration(
                tenant_id=tenant_id,
                document_id=document.id,
                revision_id=revision.id,
                embedding_profile_id=profile_id,
                generation_no=1,
                parser_version="document-processing-v1",
                cleaner_version="cleaning-v1",
                chunker_version="chunking-v1",
                chunk_config=asdict(chunk_config),
                chunk_count=len(embedded),
            )
            session.add(generation)
            session.flush()
            scope = ChunkScope(tenant_id, args.kb_id, document.id, revision.id, generation.id)
            chunk_ids = save_embedded_chunks(session, scope, embedded)

        # 用新会话读取已提交结果，核对正文、来源、顺序及 float32 向量。
        with Session(engine) as session:
            stored = session.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.tenant_id == tenant_id,
                    DocumentChunk.generation_id == scope.generation_id,
                )
                .order_by(DocumentChunk.chunk_index)
            ).all()
            if len(stored) != len(embedded):
                raise ValueError("提交后读取到的分块数量不匹配")
            for row, item in zip(stored, embedded, strict=True):
                chunk = item.chunk
                if (
                    row.chunk_index != chunk.chunk_index
                    or row.content != chunk.content
                    or row.embedding_content != chunk.embedding_content
                    or row.content_sha256 != chunk.content_sha256
                    or row.metadata_.get("source_refs") != list(chunk.source_refs)
                    or row.embedding is None
                    or len(row.embedding) != 1024
                    or not all(
                        math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-8)
                        for actual, expected in zip(row.embedding, item.embedding, strict=True)
                    )
                ):
                    raise ValueError(f"分块 {chunk.chunk_index} 入库后数据不匹配")
            if session.get(IndexGeneration, scope.generation_id).status != "READY":
                raise ValueError("入库后索引批次未就绪")
        report = {
            "run_id": run_id,
            "model": settings.embedding_model,
            "dimensions": 1024,
            "batch_count": stats.model_batches,
            "cache_hits": stats.cache_hits,
            "model_chunks": stats.model_chunks,
            "embedded_count": len(embedded),
            "stored_count": len(stored),
            "scope": asdict(scope),
            "chunk_ids": chunk_ids,
            "generation_status": "READY",
            "document_status": "DRAFT",
            "verified_after_commit": True,
            "source_snapshot": str(snapshot),
            "bucket": settings.minio_bucket,
            "object_key": object_key,
        }
        report_path = snapshot.parent / "result.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"验证报告: {report_path}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 模型和数据库异常可能包含请求正文或连接信息，仅打印异常类型。
        print(f"导入失败: {type(exc).__name__}", flush=True)
        raise SystemExit(1) from None
