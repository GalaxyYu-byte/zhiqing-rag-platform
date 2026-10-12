"""通过项目 Settings 执行 PostgreSQL 迁移；离线模式不创建连接。"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import command, context
from zhiqing_rag.core.config import get_settings
from zhiqing_rag.infrastructure.database.migrations import (
    acquire_migration_lock,
    autogenerate_options,
)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

settings = config.attributes.get("settings")
if settings is None:
    settings = get_settings()
schema = settings.db_schema
config.attributes["target_schema"] = schema
options = autogenerate_options(schema)
options["version_table_schema"] = schema


def create_schema_requested() -> bool:
    # 有目标版本的迁移在版本表之前准备 schema；current/check 没有目标版本。
    # 使用 Alembic 的公共 API，也覆盖 programmatic command.upgrade/stamp。
    cli = getattr(config, "cmd_opts", None)
    cli_command = getattr(cli, "cmd", (None,))[0] if cli is not None else None
    try:
        target = context.get_revision_argument()
    except KeyError:
        target = None
    return (
        bool(config.attributes.get("create_schema"))
        or target is not None
        or cli_command
        in {
            command.upgrade,
            command.stamp,
        }
    )


def run_migrations_offline() -> None:
    context.configure(
        dialect_name="postgresql",
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        **options,
    )
    with context.begin_transaction():
        if create_schema_requested():
            context.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        context.run_migrations()


def run_on_connection(connection) -> None:
    context.configure(connection=connection, **options)
    with context.begin_transaction():
        acquire_migration_lock(connection, schema)
        if create_schema_requested():
            connection.exec_driver_sql(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        context.run_migrations()


def run_migrations_online() -> None:
    supplied = config.attributes.get("connection")
    if supplied is not None:
        run_on_connection(supplied)
        return
    engine = create_engine(
        settings.database_url,
        poolclass=pool.NullPool,
        connect_args={
            "connect_timeout": 5,
            "options": "-c statement_timeout=30000 -c lock_timeout=5000",
        },
    )
    try:
        with engine.connect() as connection:
            run_on_connection(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
