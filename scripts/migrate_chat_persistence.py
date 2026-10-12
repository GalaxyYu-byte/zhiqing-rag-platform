"""只新增首版聊天表和索引；已有数据与其他业务表保持不变。"""

import argparse
import re
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.schema import CreateIndex, CreateTable

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.modules.chat import AnswerCitation, ChatMessage, ChatRun, ChatSession
from zhiqing_rag.modules.schema import SCHEMA


def chat_tables():
    # 依赖顺序只涵盖聊天表，基础身份、文档表必须已存在。
    return tuple(model.__table__ for model in (ChatSession, ChatRun, ChatMessage, AnswerCitation))


def render_ddl(target_schema: str) -> tuple[str, ...]:
    """渲染目标 schema，不修改全局 ORM 元数据或连接数据库。"""
    dialect = postgresql.dialect()
    options = {"dialect": dialect, "schema_translate_map": {SCHEMA: target_schema},
               "render_schema_translate": True}
    statements = []
    for table in chat_tables():
        statements.append(str(CreateTable(table).compile(**options)) + ";")
        for index in sorted(table.indexes, key=lambda value: value.name):
            statements.append(str(CreateIndex(index).compile(**options)) + ";")
    return tuple(statements)


@dataclass(frozen=True)
class Readiness:
    missing_tables: tuple[str, ...]
    missing_indexes: tuple[str, ...]
    unusable_indexes: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not (self.missing_tables or self.missing_indexes or self.unusable_indexes)


def _same_predicate(actual: str | None, expected: str | None) -> bool:
    """兼容 pg_get_expr 将 IN 改写为 ANY，以及 varchar/text 隐式类型转换。"""
    if actual is None or expected is None:
        return actual is expected

    def states(value):
        cleaned = re.sub(r"::(?:character varying|varchar|text)(?:\[\])?", "", value,
                         flags=re.IGNORECASE)
        cleaned = re.sub(r'[\s()"]', "", cleaned).upper()
        shape = r"STATUS(?:IN'[^']*'(?:,'[^']*')*|=ANYARRAY\['[^']*'(?:,'[^']*')*\]|='[^']*')"
        if re.fullmatch(shape, cleaned) is None:
            return None
        return set(re.findall(r"'([^']*)'", value))

    actual_states = states(actual)
    return actual_states is not None and actual_states == states(expected)


def check_readiness(connection, target_schema: str) -> Readiness:
    """只查询系统目录，检查四表及显式索引是否有效且归属于正确的表。"""
    tables = chat_tables()
    names = [table.name for table in tables]
    parameters = {"schema": target_schema, "tables": names}
    found_tables = set(connection.execute(text("""
        SELECT table_class.relname::text AS table_name
        FROM pg_catalog.pg_class table_class
        JOIN pg_catalog.pg_namespace namespace ON namespace.oid = table_class.relnamespace
        WHERE namespace.nspname = :schema AND table_class.relkind IN ('r', 'p')
          AND table_class.relname::text = ANY(CAST(:tables AS text[]))
    """), parameters).scalars())
    found_indexes = {
        (row["table_name"], row["index_name"]): row
        for row in connection.execute(text("""
            SELECT table_class.relname::text AS table_name,
                   index_class.relname::text AS index_name,
                   index_state.indisvalid AS valid, index_state.indisready AS ready,
                   index_state.indisunique AS is_unique,
                   pg_catalog.pg_get_expr(index_state.indpred, index_state.indrelid) AS predicate,
                   ARRAY(
                     SELECT attribute.attname::text
                     FROM unnest(index_state.indkey) WITH ORDINALITY AS key(attnum, position)
                     JOIN pg_catalog.pg_attribute attribute
                       ON attribute.attrelid = index_state.indrelid
                      AND attribute.attnum = key.attnum
                     ORDER BY key.position
                   ) AS columns
            FROM pg_catalog.pg_index index_state
            JOIN pg_catalog.pg_class index_class ON index_class.oid = index_state.indexrelid
            JOIN pg_catalog.pg_class table_class ON table_class.oid = index_state.indrelid
            JOIN pg_catalog.pg_namespace namespace ON namespace.oid = table_class.relnamespace
            WHERE namespace.nspname = :schema
              AND table_class.relname::text = ANY(CAST(:tables AS text[]))
        """), parameters).mappings()
    }
    missing_indexes = []
    unusable_indexes = []
    for table in tables:
        for index in sorted(table.indexes, key=lambda value: value.name):
            row = found_indexes.get((table.name, index.name))
            if row is None:
                missing_indexes.append(index.name)
                continue
            columns = [getattr(expression, "name", None)
                       or str(expression).split()[0] for expression in index.expressions]
            predicate = index.dialect_options["postgresql"].get("where")
            expected_predicate = str(predicate) if predicate is not None else None
            if not (row["valid"] and row["ready"] and row["is_unique"] == bool(index.unique)
                    and list(row["columns"]) == columns
                    and _same_predicate(row["predicate"], expected_predicate)):
                unusable_indexes.append(index.name)
    return Readiness(tuple(name for name in names if name not in found_tables),
                     tuple(missing_indexes), tuple(unusable_indexes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="仅显示目标 schema 建表和索引 SQL")
    modes.add_argument("--check", action="store_true", help="只读检查四表与必需索引，不执行迁移")
    arguments = parser.parse_args(argv)
    settings = get_settings()
    if arguments.dry_run:
        for statement in render_ddl(settings.db_schema):
            print(statement)
        return 0
    engine = create_engine(settings.database_url, connect_args={
        "connect_timeout": 5,
        "options": "-c statement_timeout=15000 -c lock_timeout=5000",
    }).execution_options(schema_translate_map={SCHEMA: settings.db_schema})
    try:
        if arguments.check:
            with engine.connect() as connection:
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                readiness = check_readiness(connection, settings.db_schema)
            if readiness.ready:
                print("准备状态：已就绪（4 张聊天表及 5 个必需索引）。")
                return 0
            print("准备状态：未就绪。")
            for label, values in (("缺少表", readiness.missing_tables),
                                  ("缺少索引", readiness.missing_indexes),
                                  ("索引无效或定义不符", readiness.unusable_indexes)):
                if values:
                    print(f"{label}：{', '.join(values)}")
            return 1
        with engine.begin() as connection:
            for table in chat_tables():
                table.create(connection, checkfirst=True)
                for index in sorted(table.indexes, key=lambda value: value.name):
                    index.create(connection, checkfirst=True)
        print("chat_session、chat_run、chat_message、answer_citation 表和索引已就绪。")
        return 0
    except SQLAlchemyError:
        print("准备状态：无法确认，数据库检查或迁移未完成。")
        return 2
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
