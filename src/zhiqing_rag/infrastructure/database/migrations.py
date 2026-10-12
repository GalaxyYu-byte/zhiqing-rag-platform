"""Alembic 元数据及配置；连接信息只从 Settings 读取。"""

from alembic.config import Config
from pgvector.sqlalchemy import Vector
from sqlalchemy import MetaData, text

from zhiqing_rag.core.config import PROJECT_ROOT, Settings
from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.modules import (  # noqa: F401
    chat,
    documents,
    evaluation,
    identity,
    knowledge,
    retrieval,
)


def build_metadata(schema: str) -> MetaData:
    """复制表及引用 schema；不更改应用全局 ORM 映射。"""
    metadata = MetaData()
    for table in Base.metadata.tables.values():
        table.to_metadata(metadata, schema=schema, referred_schema_fn=lambda *args: schema)
    document = metadata.tables[f"{schema}.document"]
    for constraint in document.foreign_key_constraints:
        if constraint.name in {"fk_document_desired_revision", "fk_document_active_generation"}:
            constraint.use_alter = True
    return metadata


def include_name(schema: str):
    """只比较当前业务 schema，排除版本记录表。"""

    def include(name, type_, parent_names):
        if type_ == "schema":
            return name == schema
        if type_ == "table":
            return name != "alembic_version" and parent_names.get("schema_name") == schema
        return True

    return include


def render_item(type_, item, autogen_context):
    if type_ == "type" and isinstance(item, Vector):
        autogen_context.imports.add("from pgvector.sqlalchemy import Vector")
        return f"Vector({item.dim})"
    return False


def get_alembic_config(settings: Settings | None = None) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    if settings is not None:
        config.attributes["settings"] = settings
    return config


def acquire_migration_lock(connection, schema: str) -> None:
    """同一数据库/schema 的 Alembic 操作共享事务锁，避免并发覆盖版本记录。"""
    connection.execute(
        text("""
        SELECT pg_advisory_xact_lock(hashtextextended(
            'zhiqing-rag:alembic:' || current_database() || ':' || :schema, 0))
    """),
        {"schema": schema},
    )


def autogenerate_options(schema: str) -> dict:
    return {
        "target_metadata": build_metadata(schema),
        "include_schemas": True,
        "include_name": include_name(schema),
        "compare_type": True,
        "compare_server_default": True,
        "render_item": render_item,
        # 现有 ORM 的 CHECK 未命名。1.19 的按名称比较会误删现库全部129个 CHECK。
        # CHECK、函数和触发器用手写迁移维护，首次接管通过冻结目录逐项核对。
        "autogenerate_plugins": [
            "alembic.autogenerate.*",
            "~alembic.autogenerate.checkconstraint_byname",
        ],
    }
