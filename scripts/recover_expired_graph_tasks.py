"""重新投递 Worker 异常退出后租约已过期的 Graph Task。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime

from sqlalchemy import select

from zq_rag_app.core.config import settings
from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.core.task_queue import close_task_queue, get_task_queue
from zq_rag_app.models.graph import GraphTask
from zq_rag_app.services.graph_task_service import GraphTaskStatus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="恢复租约已过期但仍处于 PROCESSING 的图任务",
    )
    parser.add_argument("--kb-id", type=int, required=True)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    now = datetime.now(UTC).replace(tzinfo=None)
    async with SessionLocal() as session:
        tasks = list(
            await session.scalars(
                select(GraphTask)
                .where(
                    GraphTask.kb_id == args.kb_id,
                    GraphTask.status == GraphTaskStatus.PROCESSING.value,
                    GraphTask.lease_expires_at.is_not(None),
                    GraphTask.lease_expires_at <= now,
                )
                .order_by(GraphTask.id)
            )
        )

    queue = await get_task_queue()
    recovered: list[int] = []
    already_exists: list[int] = []
    for task in tasks:
        lease_token = int(task.lease_expires_at.timestamp())
        job = await queue.enqueue_job(
            "execute_graph_index",
            task.id,
            _job_id=f"graph-recovery:{task.id}:{lease_token}",
            _queue_name=settings.graph_task_queue_name,
        )
        (recovered if job is not None else already_exists).append(task.id)

    print(
        json.dumps(
            {
                "kb_id": args.kb_id,
                "recovered": recovered,
                "job_already_exists": already_exists,
            },
            ensure_ascii=False,
        )
    )


async def _run_and_close() -> None:
    try:
        await main()
    finally:
        await close_task_queue()
        await close_database()


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop,
        ) as runner:
            runner.run(_run_and_close())
    else:
        asyncio.run(_run_and_close())
