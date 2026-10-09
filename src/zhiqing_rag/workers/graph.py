"""独立图谱 Worker：python -m zhiqing_rag.workers.graph。"""

import argparse
import asyncio
import contextlib
import logging
import random
import time
from uuid import uuid4

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)
from sqlalchemy import create_engine

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.documents.task_queue import LeaseLostError, ProcessingError
from zhiqing_rag.modules.schema import SCHEMA
from zhiqing_rag.workers.graph_cleanup import CleanupLease, GraphCleanupQueue
from zhiqing_rag.workers.graph_extraction import GraphExtractionError, extract_chunk
from zhiqing_rag.workers.graph_queue import GraphLease, GraphQueue
from zhiqing_rag.workers.graph_storage import GraphStore

logger = logging.getLogger(__name__)


class GraphWorker:
    def __init__(self, settings: Settings):
        if not settings.neo4j_enabled:
            raise ValueError("启用 Graph Worker 前请设置 NEO4J_ENABLED=true")
        if not settings.deepseek_api_key.get_secret_value():
            raise ValueError("启用 Graph Worker 前请配置 DEEPSEEK_API_KEY")
        self.settings = settings
        self.worker_id = f"graph-{uuid4().hex}"
        self.engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5, "options": "-c lock_timeout=5000"},
            execution_options={"schema_translate_map": {SCHEMA: settings.db_schema}},
        )
        self.queue = GraphQueue(self.engine, settings)
        self.cleanup = GraphCleanupQueue(self.engine, settings.graph_lease_seconds)
        self.store = GraphStore(settings)
        self.network_failures = 0
        self.cooldown_until = 0.0
        self.model_paused = False

    async def keep_alive(self, lease: GraphLease):
        while True:
            await asyncio.sleep(self.settings.graph_lease_seconds / 3)
            await asyncio.to_thread(self.queue.heartbeat, lease)

    async def pipeline(self, lease: GraphLease):
        scope, count = await asyncio.to_thread(self.queue.scope, lease)
        if count <= 0:
            raise ValueError("CHUNK_MISSING")
        start_index, entities, relations = await asyncio.to_thread(
            self.queue.checkpoint_state, lease
        )
        if start_index > count:
            raise ValueError("GRAPH_CHECKPOINT_INVALID")
        for index in range(start_index, count):
            # 每块前校验当前修订与租约，避免过期任务继续投影。
            await asyncio.to_thread(self.queue.scope, lease)
            chunk = await asyncio.to_thread(self.queue.chunk, lease, index)
            result = await asyncio.to_thread(self.queue.cached_extraction, lease, chunk)
            if result is None:
                result = await extract_chunk(self.settings, chunk["content"])
                await asyncio.to_thread(self.queue.save_extraction, lease, chunk, result)
            await asyncio.to_thread(self.queue.scope, lease)
            entity_keys, relation_keys = await self.store.write_chunk(scope, chunk, result)
            entities.update(entity_keys)
            relations.update(relation_keys)
            await asyncio.to_thread(
                self.queue.checkpoint, lease, index + 1, count, entities, relations
            )
        await asyncio.to_thread(self.queue.finish, lease, len(entities), len(relations))

    async def process_cleanup(self, lease: CleanupLease):
        try:
            if lease.event_type == "GRAPH_PURGE_DOCUMENT":
                await self.store.delete_document(
                    lease.tenant_id, lease.payload["document_id"], lease.payload["namespaces"]
                )
            else:
                await self.store.delete_generation(
                    lease.tenant_id,
                    lease.payload["document_id"],
                    lease.payload["generation_id"],
                    lease.payload["namespace"],
                )
            await asyncio.to_thread(self.cleanup.complete, lease)
        except Exception as error:
            logger.warning("图清理事件 %s 失败：%s", lease.event_id, type(error).__name__)
            with contextlib.suppress(ValueError):
                await asyncio.to_thread(self.cleanup.fail, lease, type(error).__name__)

    async def process(self, lease: GraphLease):
        heartbeat = asyncio.create_task(self.keep_alive(lease))
        pipeline = asyncio.create_task(self.pipeline(lease))
        try:
            done, _ = await asyncio.wait((heartbeat, pipeline), return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                await heartbeat
                raise LeaseLostError("心跳停止")
            await pipeline
            self.network_failures = 0
            self.cooldown_until = 0.0
            logger.info("图任务 %s 完成", lease.task_id)
        except LeaseLostError:
            logger.warning("图任务 %s 租约失效", lease.task_id)
        except Exception as error:
            permanent = isinstance(
                error, (AuthenticationError, PermissionDeniedError, BadRequestError)
            )
            code = "MODEL_CONFIGURATION_ERROR" if permanent else "GRAPH_PROCESSING_FAILED"
            if isinstance(error, APIStatusError) and error.status_code == 402:
                code, permanent = "MODEL_QUOTA_EXHAUSTED", True
                self.model_paused = True
            network_error = isinstance(error, (APIConnectionError, APITimeoutError)) or (
                isinstance(error, APIStatusError)
                and (error.status_code == 429 or error.status_code >= 500)
            )
            if network_error:
                self.network_failures = getattr(self, "network_failures", 0) + 1
                self.cooldown_until = (
                    time.monotonic()
                    + min(300, 15 * 2 ** min(self.network_failures - 1, 5))
                    + random.uniform(0, 5)
                )
                code = "MODEL_NETWORK_UNAVAILABLE"
            if isinstance(error, ValueError) and str(error) == "REVISION_UNAVAILABLE":
                code, permanent = "REVISION_UNAVAILABLE", True
            if isinstance(error, ValueError) and str(error) == "MODEL_CONFIG_CHANGED":
                code, permanent = "MODEL_CONFIG_CHANGED", True
            if isinstance(error, ValueError) and str(error) == "GRAPH_GENERATION_RETIRED":
                code, permanent = "GRAPH_GENERATION_RETIRED", True
            if isinstance(error, ProcessingError):
                code, permanent = error.code, True
            if isinstance(error, GraphExtractionError):
                code = f"GRAPH_{error.code}"
            detail = error.detail if isinstance(error, GraphExtractionError) else ""
            logger.warning(
                "图任务 %s 失败：%s (%s) %s", lease.task_id, code, type(error).__name__, detail
            )
            await asyncio.to_thread(self.queue.fail, lease, code, not permanent)
        finally:
            for task in (pipeline, heartbeat):
                task.cancel()
            for task in (pipeline, heartbeat):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def run(self, once: bool = False, task_id: int | None = None):
        try:
            prepared = False
            while True:
                if not prepared:
                    try:
                        await self.store.prepare()
                        prepared = True
                    except Exception as error:
                        if once:
                            raise
                        logger.warning("Neo4j 暂不可用：%s", type(error).__name__)
                        await asyncio.sleep(self.settings.graph_poll_seconds)
                        continue
                try:
                    cleanup_lease = await asyncio.to_thread(self.cleanup.claim, self.worker_id)
                except Exception as error:
                    if once:
                        raise
                    logger.warning("图清理队列不可用：%s", type(error).__name__)
                    cleanup_lease = None
                if cleanup_lease is not None:
                    await self.process_cleanup(cleanup_lease)
                if self.model_paused:
                    if once:
                        return
                    await asyncio.sleep(self.settings.graph_poll_seconds)
                    continue
                cooldown = self.cooldown_until - time.monotonic()
                if cooldown > 0:
                    if once:
                        return
                    await asyncio.sleep(min(cooldown, self.settings.graph_poll_seconds))
                    continue
                try:
                    lease = await asyncio.to_thread(self.queue.claim, self.worker_id, task_id)
                except Exception as error:
                    if once:
                        raise
                    logger.warning("图任务队列不可用：%s", type(error).__name__)
                    lease = None
                if lease is not None:
                    await self.process(lease)
                if once:
                    return
                await asyncio.sleep(self.settings.graph_poll_seconds)
        finally:
            await self.store.close()
            self.engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--task-id", type=int)
    args = parser.parse_args()
    if args.task_id is not None and (not args.once or args.task_id <= 0):
        parser.error("--task-id 必须为正整数，并与 --once 一起使用")
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(GraphWorker(settings).run(args.once, args.task_id))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
