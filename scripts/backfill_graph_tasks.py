"""为知识库中的历史正式文档创建并投递 Graph Task。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter

from sqlalchemy import select

from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.core.task_queue import close_task_queue, enqueue_graph_index
from zq_rag_app.models.document import Document
from zq_rag_app.services.graph_task_service import (
    GraphTaskStatus,
    create_graph_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为已完成文档批量创建并投递历史图谱回填任务",
    )
    parser.add_argument("--kb-id", type=int, required=True)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    async with SessionLocal() as session:
        doc_ids = list(
            await session.scalars(
                select(Document.id)
                .where(
                    Document.kb_id == args.kb_id,
                    Document.status == "DONE",
                    Document.is_deleted.is_(False),
                )
                .order_by(Document.id)
            )
        )

    counts: Counter[str] = Counter()
    task_ids: list[int] = []
    for doc_id in doc_ids:
        async with SessionLocal() as session:
            task, created = await create_graph_task(session, doc_id=int(doc_id))

        if task.status == GraphTaskStatus.DONE.value:
            counts["already_done"] += 1
            continue

        enqueued = await enqueue_graph_index(task.id)
        counts["created_or_restarted" if created else "already_pending"] += 1
        counts["enqueued" if enqueued else "job_already_exists"] += 1
        task_ids.append(task.id)

    print(
        json.dumps(
            {
                "kb_id": args.kb_id,
                "documents": len(doc_ids),
                "task_ids": task_ids,
                **counts,
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
