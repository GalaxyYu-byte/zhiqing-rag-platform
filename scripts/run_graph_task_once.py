"""选择一个真实文档并在当前进程执行一次 Graph Task。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from sqlalchemy import func, select

from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.core.neo4j import close_neo4j
from zq_rag_app.models.document import DocChunk, Document
from zq_rag_app.services.graph_task_service import (
    GraphTaskService,
    GraphTaskSnapshot,
    create_graph_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 PostgreSQL 真实 Chunk 执行一次图抽取任务",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--doc-id", type=int)
    target.add_argument("--kb-id", type=int)
    return parser.parse_args()


async def _choose_document(kb_id: int) -> tuple[int, int]:
    """选择当前版本 Chunk 最少的已完成文档，控制验证成本。"""

    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(Document.id, func.count(DocChunk.id).label("chunk_count"))
                .join(
                    DocChunk,
                    (DocChunk.doc_id == Document.id)
                    & (DocChunk.doc_version == Document.version),
                )
                .where(
                    Document.kb_id == kb_id,
                    Document.status == "DONE",
                    Document.is_deleted.is_(False),
                )
                .group_by(Document.id)
                .order_by(func.count(DocChunk.id), Document.id)
                .limit(1)
            )
        ).first()
    if row is None:
        raise RuntimeError(f"知识库 {kb_id} 没有可用于图抽取的文档")
    return int(row.id), int(row.chunk_count)


async def main() -> None:
    args = parse_args()
    if args.doc_id is not None:
        doc_id = args.doc_id
        expected_chunks = None
    else:
        doc_id, expected_chunks = await _choose_document(args.kb_id)

    async with SessionLocal() as session:
        task, created = await create_graph_task(session, doc_id=doc_id)

    print(
        json.dumps(
            {
                "doc_id": doc_id,
                "expected_chunks": expected_chunks,
                "task_id": task.id,
                "created": created,
            },
            ensure_ascii=False,
        )
    )
    await GraphTaskService().run(task.id)

    async with SessionLocal() as session:
        refreshed = await session.get(type(task), task.id)
        if refreshed is None:
            raise RuntimeError(f"图任务不存在: {task.id}")
        snapshot = GraphTaskSnapshot.from_model(refreshed)
    print(
        json.dumps(
            {
                field: getattr(snapshot, field)
                for field in snapshot.__dataclass_fields__
            },
            ensure_ascii=False,
            default=str,
        )
    )


async def _run_and_close() -> None:
    try:
        await main()
    finally:
        await close_neo4j()
        await close_database()


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop,
        ) as runner:
            runner.run(_run_and_close())
    else:
        asyncio.run(_run_and_close())
