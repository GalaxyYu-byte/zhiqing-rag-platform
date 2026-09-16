"""幂等初始化 Graph RAG 的 Neo4j 约束和索引。"""

from __future__ import annotations

import asyncio

from zq_rag_app.core.neo4j import close_neo4j
from zq_rag_app.graph.repository import GraphRepository


async def main() -> None:
    try:
        await GraphRepository().ensure_schema()
        print("Neo4j Graph RAG Schema 初始化完成")
    finally:
        await close_neo4j()


if __name__ == "__main__":
    asyncio.run(main())
