"""核对初始数据库目录，一致时才允许接管 Alembic 基线。"""

import argparse

from alembic.util import CommandError
from sqlalchemy import create_engine, inspect, pool, text
from sqlalchemy.exc import SQLAlchemyError

from alembic import command
from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.migration_catalog import (
    compare_baseline,
    load_baseline,
    read_catalog,
)
from zhiqing_rag.infrastructure.database.migrations import (
    acquire_migration_lock,
    get_alembic_config,
)

BASELINE_REVISION = "20261011_0001"


def check_baseline(connection, schema: str) -> tuple[str, ...]:
    connection.exec_driver_sql("SET LOCAL search_path TO public, pg_catalog")
    errors = list(compare_baseline(read_catalog(connection, schema), schema))
    extensions = set(connection.execute(text("SELECT extname FROM pg_extension")).scalars())
    errors.extend(f"缺少扩展：{name}" for name in ("vector", "pg_search") if name not in extensions)
    return tuple(errors)


def stamp_baseline(connection, settings) -> tuple[str, ...]:
    acquire_migration_lock(connection, settings.db_schema)
    errors = check_baseline(connection, settings.db_schema)
    if errors:
        return errors
    # 核对后锁住受管表，直到登记完成，防止并发 ALTER/DROP 使目录失效。
    tables = ", ".join(
        f'"{settings.db_schema}"."{table["name"]}"' for table in load_baseline()["tables"]
    )
    connection.exec_driver_sql(f"LOCK TABLE {tables} IN SHARE UPDATE EXCLUSIVE MODE")
    errors = check_baseline(connection, settings.db_schema)
    if errors:
        return errors
    if inspect(connection).has_table("alembic_version", schema=settings.db_schema):
        versions = tuple(
            connection.exec_driver_sql(
                f'SELECT version_num FROM "{settings.db_schema}".alembic_version'
            ).scalars()
        )
        if versions == (BASELINE_REVISION,):
            return ()
        if versions:
            return ("数据库已有其他迁移版本，不能覆盖版本记录。",)
    config = get_alembic_config(settings)
    config.attributes["connection"] = connection
    command.stamp(config, BASELINE_REVISION)
    return ()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="只读核对初始基线，不登记版本")
    mode.add_argument("--stamp-baseline", action="store_true", help="核对一致后登记初始版本")
    arguments = parser.parse_args(argv)
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        poolclass=pool.NullPool,
        connect_args={
            "connect_timeout": 5,
            "options": "-c statement_timeout=30000 -c lock_timeout=5000",
        },
    )
    try:
        with engine.begin() as connection:
            if arguments.check:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                errors = check_baseline(connection, settings.db_schema)
            else:
                errors = stamp_baseline(connection, settings)
            if errors:
                print("初始基线核对未通过，未登记 Alembic 版本：")
                for error in errors:
                    print(f"- {error}")
                return 1
        if arguments.check:
            print("初始基线一致：28 张表、约束、索引、函数、触发器及扩展均已核对。")
        else:
            print(f"Alembic 初始版本已登记：{BASELINE_REVISION}；业务表和数据未修改。")
        return 0
    except (SQLAlchemyError, CommandError):
        # SQLAlchemy 异常可能含连接信息；这里不输出原始错误或密码。
        print("数据库检查或基线登记失败，事务已回滚；请检查连接、权限及并发 DDL。")
        return 2
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
