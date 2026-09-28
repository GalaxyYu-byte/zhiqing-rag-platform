"""独立持久化索引 Worker：python -m zhiqing_rag.workers.ingestion。"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from minio.error import S3Error
from openai import AuthenticationError, BadRequestError, PermissionDeniedError
from sqlalchemy import create_engine

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.document_processing import EmbeddingStats, embed_chunks
from zhiqing_rag.document_processing.chunking import DocumentChunkDraft
from zhiqing_rag.document_processing.embedding import EmbeddingValidationError
from zhiqing_rag.infrastructure.object_storage import document_object_storage
from zhiqing_rag.modules.documents.task_queue import (
    IngestionQueue,
    LeaseLostError,
    ProcessingError,
    TaskLease,
)
from zhiqing_rag.modules.schema import SCHEMA
from zhiqing_rag.workers.publication import publish_chunks, verify_work_access

logger = logging.getLogger(__name__)


def download_source(settings: Settings, context, path: Path) -> None:
    if context["file_size"] > settings.upload_max_file_size:
        raise ProcessingError("FILE_TOO_LARGE")
    if context["bucket"] != settings.minio_bucket or not context["object_key"].startswith(
        settings.minio_object_prefix
    ):
        raise ProcessingError("OBJECT_SCOPE_MISMATCH")
    with document_object_storage(settings) as storage:
        response = storage.client.get_object(context["bucket"], context["object_key"])
        try:
            size = 0
            digest = hashlib.sha256()
            with path.open("wb") as target:
                for block in response.stream(1024 * 1024):
                    size += len(block)
                    if size > settings.upload_max_file_size:
                        raise ProcessingError("FILE_TOO_LARGE")
                    digest.update(block)
                    target.write(block)
            if size != context["file_size"] or digest.hexdigest() != context["file_sha256"]:
                raise ProcessingError("FILE_INTEGRITY_FAILED")
        finally:
            response.close()
            response.release_conn()


async def parse_chunks(settings: Settings, context, path: Path):
    result_path = path.parent / "parsed.json"
    config = context["chunk_config"]
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "zhiqing_rag.workers.parse_document",
        str(path),
        str(result_path),
        f"--filename={context['original_filename']}",
        "--size",
        str(config["max_tokens"]),
        "--overlap",
        str(config["overlap_tokens"]),
        "--max-file-size",
        str(settings.upload_max_file_size),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        if await process.wait() != 0 or not result_path.exists():
            raise ProcessingError("PARSE_FAILED")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "error_code" in result:
            raise ProcessingError(result["error_code"])
        return tuple(
            DocumentChunkDraft(**{**chunk, "source_refs": tuple(chunk["source_refs"])})
            for chunk in result["chunks"]
        )
    finally:
        if process.returncode is None:
            if os.name == "nt":
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
            else:
                process.kill()
            await process.wait()


def failure_code(error: Exception) -> tuple[str, bool]:
    if isinstance(error, ProcessingError):
        return error.code, False
    if isinstance(error, EmbeddingValidationError):
        return "INVALID_EMBEDDING", False
    if isinstance(error, (AuthenticationError, PermissionDeniedError, BadRequestError)):
        return "MODEL_CONFIGURATION_ERROR", False
    if isinstance(error, S3Error) and error.code in {"NoSuchKey", "NoSuchBucket"}:
        return "OBJECT_NOT_FOUND", False
    if isinstance(error, TimeoutError):
        return "PROCESSING_TIMEOUT", True
    return "PROCESSING_UNAVAILABLE", True


class IngestionWorker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.queue = IngestionQueue(settings)
        self.worker_id = f"ingestion-{uuid4().hex}"
        self.engine = create_engine(
            settings.database_url,
            pool_size=settings.ingestion_concurrency + 1,
            max_overflow=0,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5, "options": "-c lock_timeout=5000"},
            execution_options={"schema_translate_map": {SCHEMA: settings.db_schema}},
        )

    async def process(self, lease: TaskLease):
        started = time.monotonic()
        heartbeat = asyncio.create_task(self.keep_alive(lease))
        pipeline = asyncio.create_task(self.pipeline(lease))
        try:
            async with asyncio.timeout(self.settings.ingestion_timeout_seconds):
                done, _ = await asyncio.wait(
                    [pipeline, heartbeat], return_when=asyncio.FIRST_COMPLETED
                )
                if heartbeat in done:
                    await heartbeat
                    raise LeaseLostError("心跳停止")
                await pipeline
            logger.info("任务 %s 处理完成", lease.task_id)
        except LeaseLostError:
            logger.warning("任务 %s 租约失效，停止当前处理", lease.task_id)
        except Exception as error:
            code, retryable = failure_code(error)
            logger.warning("任务 %s 失败：%s (%s)", lease.task_id, code, type(error).__name__)
            await asyncio.to_thread(
                self.queue.fail,
                lease,
                code,
                retryable,
                {"duration_seconds": round(time.monotonic() - started, 2)},
            )
        finally:
            for task in (pipeline, heartbeat):
                task.cancel()
            for task in (pipeline, heartbeat):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def keep_alive(self, lease):
        while True:
            await asyncio.sleep(self.settings.ingestion_lease_seconds / 3)
            await asyncio.to_thread(self.queue.heartbeat, lease)

    async def pipeline(self, lease):
        context = await asyncio.to_thread(self.queue.context, lease)
        await asyncio.to_thread(verify_work_access, self.engine, lease)
        with TemporaryDirectory(prefix="zhiqing-ingestion-") as directory:
            path = Path(directory) / "source.tmp"
            # 下载在线程中执行；取消时等当前下载收尾，防止临时文件清理与写入竞争。
            download = asyncio.create_task(
                asyncio.to_thread(download_source, self.settings, context, path)
            )
            try:
                await asyncio.shield(download)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await download
                raise
            await asyncio.to_thread(self.queue.heartbeat, lease, "PARSING", 15)
            chunks = await parse_chunks(self.settings, context, path)
            await asyncio.to_thread(self.queue.heartbeat, lease, "EMBEDDING", 50)
            await asyncio.to_thread(verify_work_access, self.engine, lease)
            stats = EmbeddingStats()
            embedded = await embed_chunks(chunks, settings=self.settings, stats=stats)
            await asyncio.to_thread(self.queue.heartbeat, lease, "SAVING", 85)
            # 发布为单个数据库事务，旧租约或旧修订无法覆盖当前指针。
            await asyncio.to_thread(publish_chunks, self.engine, lease, embedded, asdict(stats))

    async def run(self, once: bool = False, task_id: int | None = None):
        active: set[asyncio.Task] = set()
        try:
            while True:
                for task in list(active):
                    if task.done():
                        active.remove(task)
                        if not task.cancelled() and task.exception() is not None:
                            logger.error("处理任务收尾失败：%s", type(task.exception()).__name__)
                while len(active) < self.settings.ingestion_concurrency:
                    try:
                        lease = await asyncio.to_thread(self.queue.claim, self.worker_id, task_id)
                    except Exception as error:
                        if once:
                            raise
                        logger.warning("任务队列暂不可用：%s", type(error).__name__)
                        break
                    if lease is None:
                        break
                    active.add(asyncio.create_task(self.process(lease)))
                    if once:
                        await asyncio.gather(*active)
                        return
                if once:
                    return
                await asyncio.sleep(self.settings.ingestion_poll_seconds)
        finally:
            for task in active:
                task.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            self.engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="只领取并处理一个任务，用于诊断")
    parser.add_argument("--task-id", type=int, help="与 --once 一起使用，只处理指定任务")
    args = parser.parse_args()
    if args.task_id is not None and (not args.once or args.task_id <= 0):
        parser.error("--task-id 必须为正整数，并与 --once 一起使用")
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(IngestionWorker(settings).run(args.once, args.task_id))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
