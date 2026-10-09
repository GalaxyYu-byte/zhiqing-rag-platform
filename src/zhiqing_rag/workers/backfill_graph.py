"""为数据库中已有的已发布 Chunk 补录图谱任务并可选立即执行。"""

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass

from sqlalchemy import create_engine, exists, func, select
from sqlalchemy.orm import Session

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.documents import BackgroundTask, Document, DocumentChunk, IndexGeneration
from zhiqing_rag.modules.retrieval import GraphProjection
from zhiqing_rag.modules.schema import SCHEMA
from zhiqing_rag.workers.graph import GraphWorker
from zhiqing_rag.workers.graph_extraction import EXTRACTOR_VERSION
from zhiqing_rag.workers.publication import lock_work_knowledge_base, verify_access

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillCandidate:
    tenant_id: int
    document_id: int
    generation_id: int
    chunk_count: int


def make_engine(settings: Settings):
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5, "options": "-c lock_timeout=5000"},
        execution_options={"schema_translate_map": {SCHEMA: settings.db_schema}},
    )


def current_generation_conditions(
    tenant_id: int | None, kb_id: int | None, document_id: int | None
):
    conditions = [
        Document.active_generation_id == IndexGeneration.id,
        Document.tenant_id == IndexGeneration.tenant_id,
        Document.desired_revision_id == IndexGeneration.revision_id,
        Document.status == "PUBLISHED",
        Document.deleted_at.is_(None),
        IndexGeneration.status == "READY",
        IndexGeneration.chunk_count > 0,
    ]
    if tenant_id is not None:
        conditions.append(Document.tenant_id == tenant_id)
    if kb_id is not None:
        conditions.append(Document.kb_id == kb_id)
    if document_id is not None:
        conditions.append(Document.id == document_id)
    return conditions


def list_missing(
    engine,
    *,
    tenant_id: int | None = None,
    kb_id: int | None = None,
    document_id: int | None = None,
    limit: int | None = None,
) -> list[BackfillCandidate]:
    """仅选择当前活动且已发布、尚无 GRAPH 任务和投影的批次。"""
    task_exists = exists(
        select(BackgroundTask.id).where(
            BackgroundTask.tenant_id == IndexGeneration.tenant_id,
            BackgroundTask.generation_id == IndexGeneration.id,
            BackgroundTask.task_type == "GRAPH",
        )
    )
    projection_exists = exists(
        select(GraphProjection.id).where(
            GraphProjection.tenant_id == IndexGeneration.tenant_id,
            GraphProjection.generation_id == IndexGeneration.id,
            GraphProjection.projection_version == 1,
        )
    )
    statement = (
        select(
            IndexGeneration.tenant_id,
            IndexGeneration.document_id,
            IndexGeneration.id,
            IndexGeneration.chunk_count,
        )
        .join(
            Document,
            (Document.tenant_id == IndexGeneration.tenant_id)
            & (Document.id == IndexGeneration.document_id),
        )
        .where(
            *current_generation_conditions(tenant_id, kb_id, document_id),
            ~task_exists,
            ~projection_exists,
        )
        .order_by(IndexGeneration.tenant_id, IndexGeneration.id)
    )
    if limit is not None:
        statement = statement.limit(limit)
    with Session(engine) as session:
        return [BackfillCandidate(*row) for row in session.execute(statement).all()]


def list_pending(
    engine,
    *,
    tenant_id: int | None = None,
    kb_id: int | None = None,
    document_id: int | None = None,
    limit: int | None = None,
    statuses: tuple[str, ...] = ("PENDING", "RETRY_WAIT", "RUNNING"),
) -> list[int]:
    statement = (
        select(BackgroundTask.id)
        .join(
            IndexGeneration,
            (IndexGeneration.tenant_id == BackgroundTask.tenant_id)
            & (IndexGeneration.id == BackgroundTask.generation_id),
        )
        .join(
            Document,
            (Document.tenant_id == IndexGeneration.tenant_id)
            & (Document.id == IndexGeneration.document_id),
        )
        .where(
            *current_generation_conditions(tenant_id, kb_id, document_id),
            BackgroundTask.task_type == "GRAPH",
            BackgroundTask.payload["pipeline"].as_string() == "document-graph-v1",
            BackgroundTask.status.in_(statuses),
        )
        .order_by(BackgroundTask.tenant_id, BackgroundTask.id)
    )
    if limit is not None:
        statement = statement.limit(limit)
    with Session(engine) as session:
        return list(session.scalars(statement))


def requeue_failed(engine, settings: Settings, task_id: int) -> bool:
    """显式请求时恢复失败任务；保留历史尝试和可重放 Neo4j 稳定键。"""
    with Session(engine) as session, session.begin():
        locator = session.execute(
            select(BackgroundTask.tenant_id, BackgroundTask.document_id).where(
                BackgroundTask.id == task_id,
                BackgroundTask.task_type == "GRAPH",
                BackgroundTask.status == "FAILED",
            )
        ).first()
        if locator is None:
            return False
        lock_work_knowledge_base(session, locator.tenant_id, locator.document_id)
        task = session.scalar(
            select(BackgroundTask)
            .where(
                BackgroundTask.id == task_id,
                BackgroundTask.task_type == "GRAPH",
                BackgroundTask.status == "FAILED",
            )
            .with_for_update()
        )
        if task is None or task.attempt_count >= 100:
            return False
        document = session.scalar(
            select(Document)
            .where(
                Document.tenant_id == task.tenant_id,
                Document.id == task.document_id,
                Document.active_generation_id == task.generation_id,
                Document.desired_revision_id == task.revision_id,
                Document.status == "PUBLISHED",
                Document.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if document is None:
            return False
        verify_access(session, document, task.revision_id)
        projection = session.scalar(
            select(GraphProjection)
            .where(
                GraphProjection.tenant_id == task.tenant_id,
                GraphProjection.generation_id == task.generation_id,
                GraphProjection.projection_version == 1,
            )
            .with_for_update()
        )
        if projection is None or projection.graph_namespace != settings.graph_namespace:
            return False
        if projection.model_name != settings.graph_extraction_model:
            return False
        task.status = "PENDING"
        task.stage = "QUEUED"
        task.progress = 0
        task.error_code = None
        task.finished_at = None
        task.next_run_at = session.scalar(select(func.clock_timestamp()))
        task.max_attempts = min(100, task.attempt_count + settings.graph_max_attempts)
        task.lease_owner = task.lease_expires_at = None
        projection.status = "BUILDING"
        projection.ready_at = None
        projection.error_code = None
        projection.entity_count = 0
        projection.relation_count = 0
        return True


def enqueue_candidate(engine, settings: Settings, candidate: BackfillCandidate) -> int | None:
    """单批次单事务补录；并发运行由文档行锁和唯一幂等键保护。"""
    with Session(engine) as session, session.begin():
        lock_work_knowledge_base(session, candidate.tenant_id, candidate.document_id)
        document = session.scalar(
            select(Document)
            .where(
                Document.tenant_id == candidate.tenant_id,
                Document.id == candidate.document_id,
            )
            .with_for_update()
        )
        generation = session.scalar(
            select(IndexGeneration)
            .where(
                IndexGeneration.tenant_id == candidate.tenant_id,
                IndexGeneration.id == candidate.generation_id,
            )
            .with_for_update()
        )
        if (
            document is None
            or generation is None
            or not (
                document.active_generation_id == generation.id
                and document.desired_revision_id == generation.revision_id
                and document.status == "PUBLISHED"
                and document.deleted_at is None
                and generation.status == "READY"
                and generation.chunk_count > 0
            )
        ):
            return None
        verify_access(session, document, generation.revision_id)
        stored_count = session.scalar(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.tenant_id == candidate.tenant_id,
                DocumentChunk.generation_id == candidate.generation_id,
            )
        )
        if stored_count != generation.chunk_count:
            raise ValueError(f"批次 {generation.id} 的 Chunk 数量与索引记录不一致")
        existing_task = session.scalar(
            select(BackgroundTask.id).where(
                BackgroundTask.tenant_id == candidate.tenant_id,
                BackgroundTask.generation_id == generation.id,
                BackgroundTask.task_type == "GRAPH",
            )
        )
        existing_projection = session.scalar(
            select(GraphProjection.id).where(
                GraphProjection.tenant_id == candidate.tenant_id,
                GraphProjection.generation_id == generation.id,
                GraphProjection.projection_version == 1,
            )
        )
        if existing_task is not None or existing_projection is not None:
            if (existing_task is None) != (existing_projection is None):
                logger.warning("批次 %s 的图任务与投影不成对，跳过自动补录", generation.id)
            return None
        task = BackgroundTask(
            tenant_id=candidate.tenant_id,
            task_type="GRAPH",
            document_id=candidate.document_id,
            revision_id=generation.revision_id,
            generation_id=generation.id,
            idempotency_key=f"graph:{generation.id}:v1",
            stage="QUEUED",
            max_attempts=settings.graph_max_attempts,
            payload={"pipeline": "document-graph-v1"},
        )
        session.add(task)
        session.add(
            GraphProjection(
                tenant_id=candidate.tenant_id,
                document_id=candidate.document_id,
                revision_id=generation.revision_id,
                generation_id=generation.id,
                projection_version=1,
                graph_namespace=settings.graph_namespace,
                extractor_version=EXTRACTOR_VERSION,
                model_name=settings.graph_extraction_model,
                status="BUILDING",
            )
        )
        session.flush()
        return task.id


def task_state(engine, task_id: int) -> tuple[str, str | None] | None:
    with Session(engine) as session:
        return session.execute(
            select(BackgroundTask.status, BackgroundTask.error_code).where(
                BackgroundTask.id == task_id,
                BackgroundTask.task_type == "GRAPH",
            )
        ).one_or_none()


async def process_task(worker: GraphWorker, task_id: int, timeout_seconds: int) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        cooldown = worker.cooldown_until - time.monotonic()
        if cooldown > 0:
            await asyncio.sleep(min(cooldown, 5))
            continue
        state = await asyncio.to_thread(task_state, worker.engine, task_id)
        if state is None:
            return "MISSING"
        status, error_code = state
        if error_code == "MODEL_QUOTA_EXHAUSTED":
            return "MODEL_QUOTA_EXHAUSTED"
        if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return status
        lease = await asyncio.to_thread(worker.queue.claim, worker.worker_id, task_id)
        if lease is not None:
            await worker.process(lease)
        else:
            await asyncio.sleep(min(2, worker.settings.graph_poll_seconds))
    return "TIMED_OUT"


async def run_tasks(
    settings: Settings, task_ids: list[int], workers: int, timeout_seconds: int
) -> dict[str, int]:
    queue: asyncio.Queue[int] = asyncio.Queue()
    for task_id in task_ids:
        queue.put_nowait(task_id)
    results: dict[str, int] = {}
    quota_exhausted = asyncio.Event()

    async def lane():
        worker = GraphWorker(settings)
        try:
            await worker.store.prepare()
            while not queue.empty() and not quota_exhausted.is_set():
                task_id = queue.get_nowait()
                try:
                    status = await process_task(worker, task_id, timeout_seconds)
                    results[status] = results.get(status, 0) + 1
                    if status == "MODEL_QUOTA_EXHAUSTED":
                        quota_exhausted.set()
                    logger.info("图任务 %s：%s", task_id, status)
                finally:
                    queue.task_done()
        finally:
            await worker.store.close()
            worker.engine.dispose()

    await asyncio.gather(*(lane() for _ in range(min(workers, len(task_ids)))))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=int, help="仅处理指定租户")
    parser.add_argument("--kb-id", type=int, help="仅处理指定知识库")
    parser.add_argument("--document-id", type=int, help="仅处理指定文档")
    parser.add_argument("--limit", type=int, help="最多选择多少个文档批次")
    parser.add_argument("--dry-run", action="store_true", help="只预览数量，不写库、不调用模型")
    parser.add_argument("--run", action="store_true", help="建任务后立即调用 Graph Worker 处理")
    parser.add_argument("--retry-failed", action="store_true", help="重新排队已有的失败图任务")
    parser.add_argument("--workers", type=int, default=2, help="--run 时并行的 Graph Worker 数")
    parser.add_argument("--task-timeout", type=int, default=3600, help="单个图任务等待上限（秒）")
    args = parser.parse_args()
    for name in ("tenant_id", "kb_id", "document_id", "limit", "workers", "task_timeout"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须为正整数")
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    engine = make_engine(settings)
    try:
        filters = {
            "tenant_id": args.tenant_id,
            "kb_id": args.kb_id,
            "document_id": args.document_id,
        }
        pending = list_pending(engine, **filters, limit=args.limit)
        remaining = None if args.limit is None else max(0, args.limit - len(pending))
        failed = (
            list_pending(engine, **filters, limit=remaining, statuses=("FAILED",))
            if args.retry_failed and remaining != 0
            else []
        )
        remaining = None if remaining is None else max(0, remaining - len(failed))
        missing = list_missing(engine, **filters, limit=remaining) if remaining != 0 else []
        print(
            f"待处理图任务: {len(pending)}；待重试: {len(failed)}；待补录批次: {len(missing)}；"
            f"待补录 Chunk: {sum(item.chunk_count for item in missing)}"
        )
        if args.dry_run:
            return
        requeued = [task_id for task_id in failed if requeue_failed(engine, settings, task_id)]
        created = []
        for candidate in missing:
            task_id = enqueue_candidate(engine, settings, candidate)
            if task_id is not None:
                created.append(task_id)
        print(f"新建 Graph 任务: {len(created)}；重新排队: {len(requeued)}")
        if args.run:
            task_ids = list(dict.fromkeys([*pending, *requeued, *created]))
            if not task_ids:
                print("没有需要执行的 Graph 任务")
                return
            results = asyncio.run(run_tasks(settings, task_ids, args.workers, args.task_timeout))
            print(f"处理结果: {results}")
            if any(status != "SUCCEEDED" for status in results):
                raise SystemExit(1)
        else:
            print("任务已持久化；独立 Graph Worker 会按队列处理。")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
