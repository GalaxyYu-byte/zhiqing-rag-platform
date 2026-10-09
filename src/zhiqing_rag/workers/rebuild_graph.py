"""选定文档的候选图谱构建、校验、切换与回退；保留旧版本。"""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.documents import BackgroundTask, Document, DocumentChunk, IndexGeneration
from zhiqing_rag.modules.retrieval import GraphProjection, active_graph_projection
from zhiqing_rag.workers.backfill_graph import make_engine, process_task
from zhiqing_rag.workers.graph import GraphWorker
from zhiqing_rag.workers.graph_extraction import (
    ENDPOINTS,
    EXTRACTOR_VERSION,
    Extraction,
    stable_key,
    validate_evidence,
)
from zhiqing_rag.workers.graph_queue import REBUILD_PIPELINE
from zhiqing_rag.workers.graph_storage import GraphStore
from zhiqing_rag.workers.publication import lock_work_knowledge_base, verify_access


def fingerprint(chunks: list[dict]) -> str:
    records = []
    for index, chunk in enumerate(chunks):
        content_hash = hashlib.sha256(chunk["content"].encode()).hexdigest()
        if chunk["chunk_index"] != index or chunk["content_sha256"] != content_hash:
            raise ValueError("CHUNK_INTEGRITY_INVALID")
        records.append((chunk["id"], index, content_hash))
    return stable_key(records)


def chunk_rows(session: Session, tenant_id: int, generation_id: int) -> list[dict]:
    return [
        {
            field: getattr(chunk, field)
            for field in (
                "id",
                "chunk_index",
                "content",
                "content_sha256",
                "page_number",
                "section_title",
                "start_offset",
                "end_offset",
                "metadata_",
            )
        }
        for chunk in session.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.generation_id == generation_id,
            )
            .order_by(DocumentChunk.chunk_index)
        )
    ]


def locked_document(session: Session, tenant_id: int, document_id: int) -> Document:
    lock_work_knowledge_base(session, tenant_id, document_id)
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


def require_current(document, projection):
    if (
        document.active_generation_id != projection.generation_id
        or document.desired_revision_id != projection.revision_id
    ):
        raise ValueError("REVISION_UNAVAILABLE")


def plan(engine, settings: Settings, tenant_id: int, document_ids: list[int]) -> list[dict]:
    if not document_ids or len(set(document_ids)) > 3:
        raise ValueError("请明确选择 1～3 份文档")
    plans = []
    with Session(engine) as session:
        for document_id in sorted(set(document_ids)):
            document = locked_document(session, tenant_id, document_id)
            base = active_graph_projection(session, tenant_id, document_id)
            if base is None:
                raise ValueError("READY_BASE_GRAPH_REQUIRED")
            require_current(document, base)
            generation = session.get(IndexGeneration, base.generation_id)
            chunks = chunk_rows(session, tenant_id, base.generation_id)
            if generation.status != "READY" or not chunks or len(chunks) != generation.chunk_count:
                raise ValueError("CHUNK_INTEGRITY_INVALID")
            plans.append(
                {
                    "tenant_id": tenant_id,
                    "document_id": document_id,
                    "title": document.title,
                    "kb_id": document.kb_id,
                    "revision_id": base.revision_id,
                    "generation_id": base.generation_id,
                    "base_projection_id": base.id,
                    "base_namespace": base.graph_namespace,
                    "base_extractor": base.extractor_version,
                    "chunk_count": len(chunks),
                    "input_fingerprint": fingerprint(chunks),
                    "extractor_version": EXTRACTOR_VERSION,
                    "model": settings.graph_extraction_model,
                }
            )
    return plans


def enqueue(engine, settings: Settings, item: dict) -> tuple[int, int]:
    with Session(engine) as session, session.begin():
        document = locked_document(session, item["tenant_id"], item["document_id"])
        base = active_graph_projection(session, item["tenant_id"], item["document_id"])
        if base is None or base.id != item["base_projection_id"]:
            raise ValueError("BASE_GRAPH_CHANGED")
        require_current(document, base)
        chunks = chunk_rows(session, base.tenant_id, base.generation_id)
        if fingerprint(chunks) != item["input_fingerprint"]:
            raise ValueError("CHUNK_INTEGRITY_INVALID")
        pending = session.scalar(
            select(BackgroundTask.id).where(
                BackgroundTask.tenant_id == base.tenant_id,
                BackgroundTask.generation_id == base.generation_id,
                BackgroundTask.payload["pipeline"].as_string() == REBUILD_PIPELINE,
                BackgroundTask.status.in_(["PENDING", "RETRY_WAIT", "RUNNING"]),
            )
        )
        if pending is not None:
            raise ValueError("REBUILD_ALREADY_RUNNING")
        version = 1 + session.scalar(
            select(func.max(GraphProjection.projection_version)).where(
                GraphProjection.tenant_id == base.tenant_id,
                GraphProjection.generation_id == base.generation_id,
            )
        )
        # 每次候选独立命名空间，失败、重放和审核均不能改写旧版共享节点。
        namespace = f"{settings.graph_namespace[:50]}:rebuild:{uuid4().hex}"
        projection = GraphProjection(
            tenant_id=base.tenant_id,
            document_id=base.document_id,
            revision_id=base.revision_id,
            generation_id=base.generation_id,
            projection_version=version,
            graph_namespace=namespace,
            extractor_version=EXTRACTOR_VERSION,
            model_name=settings.graph_extraction_model,
            status="BUILDING",
        )
        task = BackgroundTask(
            tenant_id=base.tenant_id,
            task_type="GRAPH",
            document_id=base.document_id,
            revision_id=base.revision_id,
            generation_id=base.generation_id,
            idempotency_key=f"graph-rebuild:{base.generation_id}:v{version}",
            stage="QUEUED",
            max_attempts=settings.graph_max_attempts,
            payload={
                "pipeline": REBUILD_PIPELINE,
                "projection_version": version,
                "base_projection_id": base.id,
                "input_fingerprint": item["input_fingerprint"],
            },
        )
        session.add_all([projection, task])
        session.flush()
        return projection.id, task.id


def projection_context(engine, tenant_id: int, projection_id: int) -> dict:
    with Session(engine) as session:
        projection = session.scalar(
            select(GraphProjection).where(
                GraphProjection.tenant_id == tenant_id,
                GraphProjection.id == projection_id,
            )
        )
        if projection is None:
            raise ValueError("PROJECTION_NOT_FOUND")
        document = locked_document(session, tenant_id, projection.document_id)
        require_current(document, projection)
        task = session.scalar(
            select(BackgroundTask).where(
                BackgroundTask.tenant_id == tenant_id,
                BackgroundTask.idempotency_key
                == f"graph-rebuild:{projection.generation_id}:v{projection.projection_version}",
            )
        )
        return {
            "projection_id": projection.id,
            "projection_version": projection.projection_version,
            "namespace": projection.graph_namespace,
            "model": projection.model_name,
            "extractor_version": projection.extractor_version,
            "status": projection.status,
            "ready_at": projection.ready_at.isoformat() if projection.ready_at else None,
            "entity_count": projection.entity_count,
            "relation_count": projection.relation_count,
            "scope": {
                "tenant_id": tenant_id,
                "kb_id": document.kb_id,
                "document_id": document.id,
                "revision_id": projection.revision_id,
                "generation_id": projection.generation_id,
            },
            "task_id": task.id if task else None,
            "task_status": task.status if task else None,
            "task_payload": dict(task.payload) if task else {},
            "chunks": chunk_rows(session, tenant_id, projection.generation_id),
        }


def check_snapshot(context: dict, snapshot: dict) -> dict:
    """结构校验与固定事实抽样；逐字可定位不自动裁决语义支持。"""
    issues = []
    chunks = {chunk["id"]: chunk for chunk in context["chunks"]}
    input_hash = fingerprint(context["chunks"])
    payload = context["task_payload"]
    if context["task_id"] and (
        context["task_status"] != "SUCCEEDED" or payload.get("input_fingerprint") != input_hash
    ):
        issues.append("BUILD_INCOMPLETE_OR_INPUT_CHANGED")
    if not context["ready_at"]:
        issues.append("BUILD_INCOMPLETE")
    entities = {row["key"]: row for row in snapshot["entities"]}
    evidence = {row["key"]: row for row in snapshot["evidence"]}
    assertions = {row["key"]: row for row in snapshot["assertions"]}
    for kind, rows in snapshot.items():
        if len({row["key"] for row in rows}) != len(rows):
            issues.append(f"DUPLICATE_{kind.upper()}_KEY")
    if len(entities) != context["entity_count"] or len(assertions) != context["relation_count"]:
        issues.append("PROJECTION_COUNT_MISMATCH")
    if not entities:
        issues.append("EMPTY_GRAPH_REQUIRES_REVIEW")
    for row in [*entities.values(), *evidence.values(), *assertions.values()]:
        if (
            row.get("namespace") != context["namespace"]
            or row.get("tenant_id") != context["scope"]["tenant_id"]
            or row.get("kb_id") != context["scope"]["kb_id"]
        ):
            issues.append("NODE_SCOPE_INVALID")
    for row in evidence.values():
        chunk = chunks.get(row.get("chunk_id"))
        if (
            chunk is None
            or any(
                row.get(field) != context["scope"][field]
                for field in ("document_id", "revision_id", "generation_id")
            )
            or row.get("content_sha256") != chunk["content_sha256"]
            or row.get("chunk_index") != chunk["chunk_index"]
            or not isinstance(row.get("quote_start"), int)
            or not isinstance(row.get("quote_end"), int)
            or not row.get("quote")
            or not 0 <= row["quote_start"] < row["quote_end"] <= len(chunk["content"])
            or chunk["content"][row["quote_start"] : row["quote_end"]] != row["quote"]
        ):
            issues.append("EVIDENCE_LOCATION_INVALID")
    for row in assertions.values():
        if len(row["subjects"]) != 1 or len(row["objects"]) != 1:
            issues.append("INVALID_RELATION_ENDPOINT")
            continue
        source = entities.get(row["subjects"][0])
        target = entities.get(row["objects"][0])
        allowed = ENDPOINTS.get(row.get("predicate"))
        if (
            not source
            or not target
            or not allowed
            or source["type"] not in allowed[0]
            or target["type"] not in allowed[1]
        ):
            issues.append("INVALID_RELATION_ENDPOINT")
        if not row["evidence"] or any(key not in evidence for key in row["evidence"]):
            issues.append("FACT_WITHOUT_VALID_EVIDENCE")
        if len(row["evidence"]) != 1:
            issues.append("FACT_EVIDENCE_CARDINALITY_INVALID")
        elif row["key"] != stable_key(
            row["evidence"][0],
            row["subjects"][0],
            row["predicate"],
            row["objects"][0],
            row.get("condition_text"),
        ):
            issues.append("FACT_STABLE_KEY_INVALID")
        if any(
            row.get(field) != context["scope"][field] for field in ("document_id", "generation_id")
        ):
            issues.append("ASSERTION_SCOPE_INVALID")
        for key in row["evidence"]:
            ev = evidence.get(key)
            if ev and row.get("condition_text"):
                chunk = chunks.get(ev["chunk_id"])
                if not chunk or row["condition_text"] not in chunk["content"]:
                    issues.append("CONDITION_NOT_IN_CHUNK")
    # 候选缓存逐条推导稳定键，对照真实 Neo4j，检查抽取到投影的数据损失或污染。
    if context["task_id"]:
        store = object.__new__(GraphStore)
        store.settings = Settings(_env_file=None, graph_namespace=context["namespace"])
        expected = {"entities": set(), "evidence": set(), "assertions": set()}
        expected_entities = {}
        exclusions = payload.get("fact_exclusions", {})
        review = payload.get("fact_review", {})
        if exclusions and (
            not isinstance(exclusions, dict)
            or not review.get("reviewed_by", "").strip()
            or review.get("raw_cache_fingerprint") != stable_key(payload.get("extraction_cache"))
            or any(
                not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()
            )
        ):
            issues.append("FACT_EXCLUSION_REVIEW_INVALID")
        applied_exclusions = set()
        for chunk in chunks.values():
            cache = payload.get("extraction_cache", {}).get(str(chunk["id"]), {})
            try:
                if (
                    cache["model"] != context["model"]
                    or cache["extractor_version"] != context["extractor_version"]
                    or cache["content_sha256"] != chunk["content_sha256"]
                ):
                    raise ValueError("CACHE_VERSION_INVALID")
                extraction = Extraction.model_validate(cache["extraction"])
                validate_evidence(extraction, chunk["content"])
                nodes, facts = store.chunk_records(context["scope"], chunk, extraction)
            except (KeyError, ValueError, TypeError):
                issues.append("RAW_EXTRACTION_CACHE_INVALID")
                continue
            applied_exclusions.update(row["key"] for row in facts if row["key"] in exclusions)
            facts = [row for row in facts if row["key"] not in exclusions]
            expected["entities"].update(row["key"] for row in nodes)
            expected_entities.update({row["key"]: row for row in nodes})
            expected["assertions"].update(row["key"] for row in facts)
            expected["evidence"].update(row["evidence"]["key"] for row in [*nodes, *facts])
        for kind, keys in expected.items():
            if keys != {row["key"] for row in snapshot[kind]}:
                issues.append(f"RAW_PROJECTION_{kind.upper()}_MISMATCH")
        for key, row in expected_entities.items():
            actual = entities.get(key)
            if actual and any(
                actual.get(field) != row[field] for field in ("type", "normalized_name")
            ):
                issues.append("ENTITY_PROPERTIES_INVALID")
        if applied_exclusions != set(exclusions):
            issues.append("FACT_EXCLUSION_KEY_INVALID")
    samples = []
    # 固定键排序，优先覆盖不同 Chunk，不把最前几行当作整份文档的代表。
    candidates = sorted(
        assertions.values(), key=lambda row: stable_key("graph-rebuild-review", row["key"])
    )
    selected_chunks = set()
    ordered = []
    for row in candidates:
        ev = evidence.get(row["evidence"][0]) if row["evidence"] else None
        if ev and ev["chunk_id"] not in selected_chunks:
            ordered.append(row)
            selected_chunks.add(ev["chunk_id"])
    ordered.extend(row for row in candidates if row not in ordered)
    for row in ordered[:5]:
        if len(row["subjects"]) != 1 or len(row["objects"]) != 1 or not row["evidence"]:
            continue
        ev = evidence.get(row["evidence"][0])
        source, target = entities.get(row["subjects"][0]), entities.get(row["objects"][0])
        if not ev or not source or not target or ev["chunk_id"] not in chunks:
            continue
        samples.append(
            {
                "fact_key": row["key"],
                "source": source["name"],
                "predicate": row["predicate"],
                "target": target["name"],
                "condition_text": row.get("condition_text"),
                "quote": ev["quote"],
                "chunk_id": ev["chunk_id"],
                "content": chunks[ev["chunk_id"]]["content"],
                "verdict": "PENDING",
                "reason": "",
            }
        )
    if not samples:
        issues.append("NO_FACT_SAMPLE_REQUIRES_REVIEW")
    return {
        "schema": "graph-rebuild-review-v1",
        "projection_id": context["projection_id"],
        "scope": context["scope"],
        "namespace": context["namespace"],
        "input_fingerprint": input_hash,
        "snapshot_fingerprint": stable_key(snapshot),
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "counts": {kind: len(rows) for kind, rows in snapshot.items()},
        "issues": sorted(set(issues)),
        "structural_passed": not issues,
        "reviewed_by": "",
        "samples": samples,
    }


async def inspect(engine, settings: Settings, tenant_id: int, projection_id: int) -> dict:
    context = projection_context(engine, tenant_id, projection_id)
    store = GraphStore(settings.model_copy(update={"graph_namespace": context["namespace"]}))
    try:
        snapshot = await store.snapshot(context["scope"])
    finally:
        await store.close()
    return check_snapshot(context, snapshot)


async def filter_candidate(
    engine, settings: Settings, tenant_id: int, projection_id: int,
    exclusions: dict[str, str], reviewed_by: str,
) -> dict:
    """仅剔除已完成候选中的明确错误；原始模型缓存和旧活动图保持可核对。"""
    context = projection_context(engine, tenant_id, projection_id)
    if context["status"] != "BUILDING" or context["task_status"] != "SUCCEEDED":
        raise ValueError("CANDIDATE_NOT_COMPLETE")
    if not reviewed_by.strip() or not exclusions or any(
        not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()
    ):
        raise ValueError("FACT_EXCLUSION_REVIEW_REQUIRED")
    store = GraphStore(settings.model_copy(update={"graph_namespace": context["namespace"]}))
    cache_hash = stable_key(context["task_payload"]["extraction_cache"])
    records = []
    known_keys = set()
    try:
        for chunk in context["chunks"]:
            raw = context["task_payload"]["extraction_cache"][str(chunk["id"])]["extraction"]
            extraction = Extraction.model_validate(raw)
            validate_evidence(extraction, chunk["content"])
            _, facts = store.chunk_records(context["scope"], chunk, extraction)
            known_keys.update(row["key"] for row in facts)
            filtered = extraction.model_copy(update={"relations": [
                relation for relation, fact in zip(extraction.relations, facts, strict=True)
                if fact["key"] not in exclusions
            ]})
            records.append((chunk, filtered))
        if not set(exclusions) <= known_keys or not known_keys - set(exclusions):
            raise ValueError("FACT_EXCLUSION_KEY_INVALID")

        def save_state(entity_keys=None, relation_keys=None):
            with Session(engine) as session, session.begin():
                lock_work_knowledge_base(session, tenant_id, context["scope"]["document_id"])
                task = session.scalar(select(BackgroundTask).where(
                    BackgroundTask.tenant_id == tenant_id, BackgroundTask.id == context["task_id"]
                ).with_for_update())
                document = locked_document(session, tenant_id, context["scope"]["document_id"])
                projection = session.scalar(select(GraphProjection).where(
                    GraphProjection.tenant_id == tenant_id, GraphProjection.id == projection_id
                ).with_for_update())
                if projection is None or task is None:
                    raise ValueError("CANDIDATE_NOT_COMPLETE")
                require_current(document, projection)
                if projection.status != "BUILDING" or task.status != "SUCCEEDED":
                    raise ValueError("CANDIDATE_NOT_COMPLETE")
                if (
                    fingerprint(chunk_rows(session, tenant_id, projection.generation_id))
                    != context["task_payload"]["input_fingerprint"]
                    or stable_key(task.payload["extraction_cache"]) != cache_hash
                ):
                    raise ValueError("REVIEW_STALE_OR_WRONG_SCOPE")
                task.payload = {
                    **task.payload,
                    "fact_exclusions": exclusions,
                    "fact_review": {
                        "reviewed_by": reviewed_by, "raw_cache_fingerprint": cache_hash,
                        "at_utc": datetime.now(UTC).isoformat(),
                    },
                }
                if entity_keys is not None:
                    projection.entity_count = len(entity_keys)
                    projection.relation_count = len(relation_keys)
                    task.payload = {
                        **task.payload, "entity_keys": sorted(entity_keys),
                        "relation_keys": sorted(relation_keys),
                    }

        # 先记录裁决，写图失败时结构/计数检查会阻止切换，可用同一裁决重放。
        save_state()
        entity_keys, relation_keys = set(), set()
        for chunk, extraction in records:
            nodes, facts = await store.write_chunk(context["scope"], chunk, extraction)
            entity_keys.update(nodes)
            relation_keys.update(facts)
        save_state(entity_keys, relation_keys)
    finally:
        await store.close()
    return await inspect(engine, settings, tenant_id, projection_id)


def approve_review(saved: dict, fresh: dict):
    if not fresh["structural_passed"]:
        raise ValueError("GRAPH_STRUCTURE_INVALID")
    for field in (
        "schema",
        "projection_id",
        "scope",
        "namespace",
        "input_fingerprint",
        "snapshot_fingerprint",
    ):
        if saved.get(field) != fresh[field]:
            raise ValueError("REVIEW_STALE_OR_WRONG_SCOPE")
    if not saved.get("reviewed_by", "").strip():
        raise ValueError("EVIDENCE_REVIEW_REQUIRED")
    saved_samples = {row["fact_key"]: row for row in saved.get("samples", [])}
    if len(saved_samples) != len(saved.get("samples", [])) or set(saved_samples) != {
        row["fact_key"] for row in fresh["samples"]
    }:
        raise ValueError("EVIDENCE_REVIEW_INCOMPLETE")
    for row in fresh["samples"]:
        saved_row = saved_samples[row["fact_key"]]
        if any(
            saved_row.get(key) != value
            for key, value in row.items()
            if key not in {"verdict", "reason"}
        ):
            raise ValueError("REVIEW_STALE_OR_WRONG_SCOPE")
        if saved_row.get("verdict") != "SUPPORT" or not saved_row.get("reason", "").strip():
            raise ValueError("EVIDENCE_REVIEW_NOT_PASSED")


def switch(
    engine,
    tenant_id: int,
    projection_id: int,
    fresh: dict,
    *,
    rollback: bool = False,
    expected_active_id: int | None = None,
):
    """同一 PostgreSQL 事务切换可见版本；保留旧节点、旧投影及旧抽取缓存。"""
    with Session(engine) as session, session.begin():
        target = session.scalar(
            select(GraphProjection).where(
                GraphProjection.tenant_id == tenant_id,
                GraphProjection.id == projection_id,
            )
        )
        if target is None:
            raise ValueError("PROJECTION_NOT_FOUND")
        lock_work_knowledge_base(session, tenant_id, target.document_id)
        task = session.scalar(
            select(BackgroundTask)
            .where(
                BackgroundTask.tenant_id == tenant_id,
                BackgroundTask.idempotency_key
                == f"graph-rebuild:{target.generation_id}:v{target.projection_version}",
            )
            .with_for_update()
        )
        document = locked_document(session, tenant_id, target.document_id)
        require_current(document, target)
        projections = list(
            session.scalars(
                select(GraphProjection)
                .where(
                    GraphProjection.tenant_id == tenant_id,
                    GraphProjection.generation_id == target.generation_id,
                )
                .order_by(GraphProjection.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        )
        target = next(row for row in projections if row.id == projection_id)
        active = next((row for row in projections if row.status == "READY"), None)
        if active and active.id == target.id:
            return
        if rollback:
            if expected_active_id is None or not active or active.id != expected_active_id:
                raise ValueError("BASE_GRAPH_CHANGED")
            if target.status != "STALE" or target.ready_at is None:
                raise ValueError("ROLLBACK_TARGET_NOT_RETAINED")
        else:
            if not task or task.status != "SUCCEEDED" or target.status != "BUILDING":
                raise ValueError("CANDIDATE_NOT_COMPLETE")
            if not active or active.id != task.payload["base_projection_id"]:
                raise ValueError("BASE_GRAPH_CHANGED")
        chunks = chunk_rows(session, tenant_id, target.generation_id)
        if (
            fresh["projection_id"] != target.id
            or not fresh["structural_passed"]
            or fresh["input_fingerprint"] != fingerprint(chunks)
        ):
            raise ValueError("REVIEW_STALE_OR_WRONG_SCOPE")
        active.status = "STALE"
        session.flush()  # 先解除唯一 READY 索引，再设置目标为 READY，提交后一起可见。
        target.status = "READY"
        if task:
            task.payload = {
                **task.payload,
                "activation": {
                    "previous_projection_id": active.id,
                    "snapshot_fingerprint": fresh["snapshot_fingerprint"],
                    "at_utc": datetime.now(UTC).isoformat(),
                    "rollback": rollback,
                },
            }
            task.stage = "COMPLETED"


async def run_candidate(
    engine, settings: Settings, tenant_id: int, projection_id: int, timeout_seconds: int
) -> str:
    context = projection_context(engine, tenant_id, projection_id)
    if not context["task_id"] or context["status"] != "BUILDING":
        raise ValueError("CANDIDATE_NOT_BUILDABLE")
    worker = GraphWorker(settings.model_copy(update={"graph_namespace": context["namespace"]}))
    try:
        await worker.store.prepare()
        return await process_task(worker, context["task_id"], timeout_seconds)
    finally:
        await worker.store.close()
        worker.engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "build", "run", "verify", "activate", "rollback")
    )
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--document-id", type=int, action="append")
    parser.add_argument("--projection-id", type=int)
    parser.add_argument("--expected-active-id", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--task-timeout", type=int, default=3600)
    args = parser.parse_args()
    for value in (
        args.tenant_id,
        args.projection_id,
        args.expected_active_id,
        args.task_timeout,
        *(args.document_id or []),
    ):
        if value is not None and value <= 0:
            parser.error("ID 和超时必须为正整数")
    if args.command in {"plan", "build"} and not args.document_id:
        parser.error("plan/build 需要明确 --document-id，可重复指定，最多 3 份")
    if args.command not in {"plan", "build"} and not args.projection_id:
        parser.error("该命令需要 --projection-id")
    if args.command in {"verify", "activate"} and not args.report:
        parser.error("verify/activate 需要 --report")
    if args.command == "rollback" and not args.expected_active_id:
        parser.error("rollback 需要 --expected-active-id")
    if args.command == "verify" and args.report.exists():
        parser.error("报告已存在，请使用新文件名，防止覆盖裁决")
    settings = get_settings()
    engine = make_engine(settings)
    try:
        if args.command in {"plan", "build"}:
            items = plan(engine, settings, args.tenant_id, args.document_id)
            if args.command == "plan":
                print(json.dumps(items, ensure_ascii=False, indent=2))
            else:
                for item in items:
                    projection_id, task_id = enqueue(engine, settings, item)
                    print(json.dumps({"projection_id": projection_id, "task_id": task_id}))
        elif args.command == "run":
            status = asyncio.run(
                run_candidate(
                    engine, settings, args.tenant_id, args.projection_id, args.task_timeout
                )
            )
            print(status)
            if status != "SUCCEEDED":
                raise SystemExit(1)
        else:
            fresh = asyncio.run(inspect(engine, settings, args.tenant_id, args.projection_id))
            if args.command == "verify":
                args.report.parent.mkdir(parents=True, exist_ok=True)
                with args.report.open("x", encoding="utf-8") as stream:
                    json.dump(fresh, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
                print(
                    f"结构校验: {fresh['structural_passed']}；事实抽样: {len(fresh['samples'])} 条"
                )
                if not fresh["structural_passed"]:
                    raise SystemExit(1)
            elif args.command == "activate":
                approve_review(json.loads(args.report.read_text(encoding="utf-8")), fresh)
                switch(engine, args.tenant_id, args.projection_id, fresh)
                print("已切换；旧版本保留，可回退。")
            else:
                if not fresh["structural_passed"]:
                    raise ValueError("ROLLBACK_GRAPH_INVALID")
                switch(
                    engine,
                    args.tenant_id,
                    args.projection_id,
                    fresh,
                    rollback=True,
                    expected_active_id=args.expected_active_id,
                )
                print("已回退；其他文档与图谱节点保留。")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
