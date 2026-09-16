"""应用文档级访问元数据迁移。"""

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
    / "005_document_access_metadata.sql"
)


async def main() -> None:
    connection = await AsyncConnection.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_username,
        password=settings.db_password,
    )
    try:
        async with connection.cursor() as cursor:
            await cursor.execute(MIGRATION_PATH.read_text(encoding="utf-8"))
        await connection.commit()
        print("PostgreSQL 文档访问元数据迁移完成")
    finally:
        await connection.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(main())
    else:
        asyncio.run(main())
