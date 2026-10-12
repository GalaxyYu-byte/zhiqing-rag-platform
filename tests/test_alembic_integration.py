"""显式启用的 PostgreSQL 迁移验证；独立 schema 与所有 DDL 最终一起回滚。"""

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, inspect

from alembic import command
from scripts.migrate_database import BASELINE_REVISION, check_baseline
from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.migrations import get_alembic_config


@pytest.mark.skipif(
    os.getenv("RUN_ALEMBIC_DB_INTEGRATION") != "1",
    reason="设置 RUN_ALEMBIC_DB_INTEGRATION=1 才连接 PostgreSQL 验证迁移。",
)
def test_baseline_upgrade_catalog_current_head_and_downgrade_in_rolled_back_schema():
    schema = f"alembic_verify_{uuid4().hex}"
    settings = get_settings().model_copy(update={"db_schema": schema})
    engine = create_engine(
        settings.database_url,
        connect_args={
            "connect_timeout": 5,
            "options": "-c statement_timeout=60000 -c lock_timeout=5000",
        },
    )
    try:
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                assert not inspect(connection).has_schema(schema)
                config = get_alembic_config(settings)
                config.attributes["connection"] = connection
                command.upgrade(config, BASELINE_REVISION)
                assert check_baseline(connection, schema) == ()
                command.upgrade(config, "head")
                command.check(config)
                command.downgrade(config, "base")
                assert set(inspect(connection).get_table_names(schema=schema)) <= {
                    "alembic_version",
                }
            finally:
                transaction.rollback()
        with engine.connect() as connection:
            assert not inspect(connection).has_schema(schema)
    finally:
        engine.dispose()
