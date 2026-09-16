"""应用独立 Graph Task 的 PostgreSQL 幂等迁移。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from psycopg import AsyncConnection

from zq_rag_app.core.config import settings


MIGRATION_PATH = (
    Path(__file__).parents[1]
    / "zq_rag_app"
    / "schemas"
    / "migrations"
    / "004_graph_tasks.sql"
)


async def main() -> None:
    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    connection = await AsyncConnection.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_username,
        password=settings.db_password,
    )
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(sql)
        await connection.commit()
        print("PostgreSQL Graph Task 迁移完成")
    finally:
        await connection.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=asyncio.SelectorEventLoop,
        ) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
