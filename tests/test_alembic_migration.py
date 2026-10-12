"""冻结基线、离线 SQL 与 schema 隔离；不读取 .env 或连接数据库。"""

import ast
import json
import re
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
import sqlalchemy
from pgvector.sqlalchemy import Vector
from sqlalchemy import MetaData
from sqlalchemy.exc import SQLAlchemyError

from alembic import command
from scripts import migrate_database as adoption
from zhiqing_rag.core import config as app_config
from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database import migration_catalog, migrations
from zhiqing_rag.infrastructure.database.base import Base
from zhiqing_rag.infrastructure.database.migration_catalog import compare_baseline, load_baseline
from zhiqing_rag.modules.schema import SCHEMA

ROOT = Path(__file__).resolve().parents[1]
BASELINE_REVISION = "20261011_0001"
TABLES = {
    "tenant",
    "user_account",
    "department",
    "tenant_member",
    "auth_session",
    "audit_event",
    "knowledge_base",
    "knowledge_base_grant",
    "document",
    "document_access_grant",
    "document_revision",
    "embedding_profile",
    "index_generation",
    "document_chunk",
    "background_task",
    "task_attempt",
    "outbox_event",
    "chat_session",
    "chat_run",
    "chat_message",
    "answer_citation",
    "conversation_summary",
    "graph_projection",
    "answer_feedback",
    "evaluation_dataset",
    "evaluation_case",
    "evaluation_run",
    "evaluation_result",
}


@pytest.fixture
def offline_config(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("离线迁移不允许读取运行配置或连接数据库")

    monkeypatch.setattr(app_config, "get_settings", denied)
    monkeypatch.setattr(sqlalchemy, "create_engine", denied)
    monkeypatch.setattr(psycopg, "connect", denied)
    settings = Settings(
        _env_file=None,
        db_schema="alembic_test",
        db_password="sql_render_private_value",
    )
    config = migrations.get_alembic_config(settings=settings)
    config.output_buffer = StringIO()
    return config


def render_upgrade(config):
    command.upgrade(config, "head", sql=True)
    return config.output_buffer.getvalue()


def created_tables(sql, schema):
    return list(
        re.finditer(
            rf'\bCREATE\s+TABLE\s+"?{schema}"?\."?([a-z][a-z0-9_]*)"?',
            sql,
            re.IGNORECASE,
        )
    )


def test_offline_upgrade_creates_complete_baseline_without_connecting(offline_config):
    sql = render_upgrade(offline_config)
    tables = {match.group(1) for match in created_tables(sql, "alembic_test")}
    assert tables - {"alembic_version"} == TABLES
    assert BASELINE_REVISION in sql
    assert "sql_render_private_value" not in sql
    schema_creation = re.search(
        r'CREATE\s+SCHEMA\s+IF\s+NOT\s+EXISTS\s+"?alembic_test"?',
        sql,
        re.IGNORECASE,
    )
    version_creation = next(
        match
        for match in created_tables(sql, "alembic_test")
        if match.group(1) == "alembic_version"
    )
    assert schema_creation is not None and schema_creation.start() < version_creation.start()
    assert re.search(r"CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS\s+vector", sql, re.IGNORECASE)
    assert re.search(r"CREATE\s+EXTENSION\s+IF\s+NOT\s+EXISTS\s+pg_search", sql, re.IGNORECASE)


def test_offline_upgrade_adds_foreign_keys_after_all_business_tables(offline_config):
    sql = render_upgrade(offline_config)
    tables = [
        match
        for match in created_tables(sql, "alembic_test")
        if match.group(1) != "alembic_version"
    ]
    first_foreign_key = re.search(r"\bFOREIGN\s+KEY\b", sql, re.IGNORECASE)
    assert first_foreign_key is not None
    assert first_foreign_key.start() > max(match.start() for match in tables)
    for name in ("fk_document_desired_revision", "fk_document_active_generation"):
        assert name in sql
    assert re.search(r"REFERENCES\s+alembic_test\.document_revision", sql, re.IGNORECASE)
    assert re.search(r"REFERENCES\s+alembic_test\.index_generation", sql, re.IGNORECASE)


def test_custom_schema_applies_to_functions_triggers_and_search_indexes(offline_config):
    sql = render_upgrade(offline_config)
    assert "zhiqing_rag." not in sql
    assert '"zhiqing_rag".' not in sql
    assert len(re.findall(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\b", sql, re.IGNORECASE)) == 10
    assert len(re.findall(r"\bCREATE\s+TRIGGER\b", sql, re.IGNORECASE)) == 30
    assert re.search(r"FUNCTION\s+alembic_test\.", sql, re.IGNORECASE)
    assert re.search(r"ON\s+alembic_test\.document_revision", sql, re.IGNORECASE)
    assert re.search(r"USING\s+hnsw\b", sql, re.IGNORECASE)
    assert re.search(r"USING\s+bm25\b", sql, re.IGNORECASE)


def test_offline_downgrade_keeps_schema_extensions_and_has_no_cascade(offline_config):
    command.downgrade(offline_config, f"{BASELINE_REVISION}:base", sql=True)
    sql = offline_config.output_buffer.getvalue()
    tables = set(
        re.findall(
            r'\bDROP\s+TABLE\s+"?alembic_test"?\."?([a-z][a-z0-9_]*)"?',
            sql,
            re.IGNORECASE,
        )
    )
    assert tables - {"alembic_version"} == TABLES
    assert not re.search(r"\bDROP\s+(?:SCHEMA|EXTENSION)\b|\bCASCADE\b", sql, re.IGNORECASE)
    assert "zhiqing_rag." not in sql


def test_frozen_revision_does_not_follow_current_orm_metadata(offline_config, monkeypatch):
    # 将当前 ORM 视图换成空集合，旧基线仍须渲染同样的数据库对象。
    monkeypatch.setattr(migrations, "build_metadata", lambda schema: MetaData(schema=schema))
    sql = render_upgrade(offline_config)
    assert {match.group(1) for match in created_tables(sql, "alembic_test")} - {
        "alembic_version",
    } == TABLES
    (revision_file,) = (ROOT / "alembic" / "versions").glob(f"{BASELINE_REVISION}_*.py")
    tree = ast.parse(revision_file.read_text(encoding="utf-8"))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports.extend(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert all(not name.startswith("zhiqing_rag.modules") for name in imports if name)


def test_metadata_clone_translates_foreign_keys_without_mutating_registered_models():
    default = migrations.build_metadata(SCHEMA)
    custom = migrations.build_metadata("alembic_test")
    assert {table.name for table in custom.tables.values()} == TABLES
    assert all(table.schema == "alembic_test" for table in custom.tables.values())
    assert all(table.schema == SCHEMA for table in Base.metadata.tables.values())
    assert all(table.schema == SCHEMA for table in default.tables.values())
    for table in custom.tables.values():
        for foreign_key in table.foreign_keys:
            assert foreign_key.column.table.schema == "alembic_test"


def test_autogenerate_compares_only_target_schema_and_excludes_version_table():
    include = migrations.include_name("alembic_test")
    assert include("alembic_test", "schema", {})
    assert not include(None, "schema", {})
    assert not include("public", "schema", {})
    assert not include("zhiqing_rag", "schema", {})
    assert include("document", "table", {"schema_name": "alembic_test"})
    assert not include("alembic_version", "table", {"schema_name": "alembic_test"})
    assert not include("document", "table", {"schema_name": "public"})
    assert not include("document", "table", {"schema_name": None})


def test_autogenerated_vector_type_has_usable_import():
    context = SimpleNamespace(imports=set())
    assert migrations.render_item("type", Vector(1024), context) == "Vector(1024)"
    assert context.imports == {"from pgvector.sqlalchemy import Vector"}
    assert migrations.render_item("type", sqlalchemy.Integer(), context) is False


def test_catalog_accepts_complete_frozen_baseline_and_custom_schema():
    original = load_baseline()
    assert compare_baseline(deepcopy(original), SCHEMA) == ()
    custom = json.loads(re.sub(r"\bzhiqing_rag\b", "alembic_test", json.dumps(original)))
    assert compare_baseline(custom, "alembic_test") == ()
    assert custom != original


@pytest.fixture
def replayed_catalog():
    # 真实隔离 schema 重放产生的 25 个 CHECK 和 2 个索引展示差异。
    fixture_path = ROOT / "tests" / "fixtures" / "alembic_replayed_definitions.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    schema = fixture["schema"]
    actual = json.loads(re.sub(r"\bzhiqing_rag\b", schema, json.dumps(load_baseline())))
    for category in ("constraints", "indexes"):
        objects = {(row["table_name"], row["name"]): row for row in actual[category]}
        for row in fixture[category]:
            objects[row["table_name"], row["name"]]["definition"] = row["definition"]
    return actual, schema


def test_catalog_accepts_real_postgres_replay_text_array_casts(replayed_catalog):
    actual, schema = replayed_catalog
    assert compare_baseline(actual, schema) == ()


@pytest.mark.parametrize(
    "category,name",
    [
        ("constraints", "chat_run_status_check"),
        ("indexes", "uq_chat_one_active_run"),
    ],
)
@pytest.mark.parametrize(
    "change",
    [
        "allowed_value",
        "value_order",
        "numeric_cast",
        "varchar_limit",
    ],
)
def test_text_array_normalization_preserves_values_order_and_other_casts(
    replayed_catalog,
    category,
    name,
    change,
):
    actual, schema = replayed_catalog
    row = next(row for row in actual[category] if row["name"] == name)
    definition = row["definition"]
    if change == "allowed_value":
        row["definition"] = definition.replace("'PENDING'", "'NEW_STATE'", 1)
    elif change == "value_order":
        row["definition"] = (
            definition.replace("'PENDING'", "'__FIRST__'")
            .replace("'RUNNING'", "'PENDING'")
            .replace("'__FIRST__'", "'RUNNING'")
        )
    elif change == "numeric_cast":
        row["definition"] = definition.replace(
            "'PENDING'::character varying", "'PENDING'::numeric", 1
        )
    else:
        row["definition"] = definition.replace(
            "'PENDING'::character varying", "'PENDING'::character varying(3)", 1
        )
    assert row["definition"] != definition
    errors = compare_baseline(actual, schema)
    assert len(errors) == 1 and name in errors[0]


@pytest.mark.parametrize("category", ["functions", "constraints"])
def test_text_array_normalization_preserves_function_code_and_quoted_sql_literals(
    monkeypatch,
    category,
):
    expected = load_baseline()
    row = expected[category][0]
    if category == "functions":
        row["definition"] = (
            "CREATE OR REPLACE FUNCTION zhiqing_rag.cast_guard() RETURNS boolean "
            "LANGUAGE sql AS $$ SELECT 'PENDING'::text = ANY "
            "(ARRAY['PENDING'::character varying]::text[]) $$"
        )
    else:
        row["definition"] = "CHECK (note <> 'ANY (ARRAY[''PENDING''::character varying]::text[])')"
    actual = deepcopy(expected)
    changed = actual[category][0]
    changed["definition"] = changed["definition"].replace(
        "::character varying]::text[]",
        "::character varying::text]",
    )
    monkeypatch.setattr(migration_catalog, "load_baseline", lambda: expected)
    errors = compare_baseline(actual, SCHEMA)
    assert len(errors) == 1 and row["name"] in errors[0]


@pytest.mark.parametrize(
    "change",
    [
        "invalid_index",
        "unready_index",
        "changed_index",
        "disabled_trigger",
        "changed_function",
        "missing_column",
        "missing_check",
        "extra_table",
        "changed_sequence",
    ],
)
def test_catalog_rejects_incomplete_or_changed_definitions_without_dumping_sql(change):
    actual = deepcopy(load_baseline())
    if change in {"invalid_index", "unready_index", "changed_index"}:
        row = next(row for row in actual["indexes"] if row["name"] == "ix_chunk_bm25")
        if change == "invalid_index":
            row["valid"] = False
        elif change == "unready_index":
            row["ready"] = False
        else:
            row["definition"] = row["definition"].replace("key_field=id", "key_field=content")
    elif change == "disabled_trigger":
        row = actual["triggers"][0]
        row["enabled"] = "D"
    elif change == "changed_function":
        row = actual["functions"][0]
        row["definition"] += "\n-- private_sql_body_value"
    elif change == "missing_column":
        row = actual["columns"].pop()
    elif change == "missing_check":
        row = next(row for row in actual["constraints"] if row["type"] == "c")
        actual["constraints"].remove(row)
    elif change == "extra_table":
        row = {"name": "untracked_table", "comment": None}
        actual["tables"].append(row)
    else:
        row = actual["sequences"][0]
        row["cache_size"] += 1
    errors = compare_baseline(actual, SCHEMA)
    assert errors
    assert any(row["name"] in error for error in errors)
    assert "private_sql_body_value" not in "\n".join(errors)


class AdoptionConnection:
    def __init__(self, versions=()):
        self.calls = []
        self.events = []
        self.versions = versions

    def exec_driver_sql(self, statement):
        self.calls.append(statement)
        self.events.append(("sql", statement))
        if statement.startswith("SELECT version_num"):
            return SimpleNamespace(scalars=lambda: self.versions)
        assert statement.startswith("LOCK TABLE") or statement == "SET TRANSACTION READ ONLY"


@pytest.fixture
def adoption_setup(monkeypatch):
    settings = Settings(_env_file=None, db_schema="alembic_test")
    stamps = []
    monkeypatch.setattr(
        adoption,
        "acquire_migration_lock",
        lambda conn, schema: conn.events.append(("migration_lock", schema)),
    )
    monkeypatch.setattr(
        adoption.command, "stamp", lambda config, revision: stamps.append((config, revision))
    )
    return settings, stamps


def test_adoption_mismatch_refuses_table_lock_and_stamp(adoption_setup, monkeypatch):
    settings, stamps = adoption_setup
    connection = AdoptionConnection()
    errors = ("缺少表：document",)
    monkeypatch.setattr(adoption, "check_baseline", lambda *args: errors)
    assert adoption.stamp_baseline(connection, settings) == errors
    assert connection.calls == []
    assert connection.events == [("migration_lock", "alembic_test")]
    assert stamps == []


def test_adoption_rechecks_after_lock_and_refuses_concurrent_changes(adoption_setup, monkeypatch):
    settings, stamps = adoption_setup
    connection = AdoptionConnection()
    checks = iter([(), ("定义不符：immutable_citation",)])
    monkeypatch.setattr(adoption, "check_baseline", lambda *args: next(checks))
    assert adoption.stamp_baseline(connection, settings) == ("定义不符：immutable_citation",)
    assert len(connection.calls) == 1 and connection.calls[0].startswith("LOCK TABLE")
    assert connection.calls[0].endswith("IN SHARE UPDATE EXCLUSIVE MODE")
    assert all(f'"alembic_test"."{name}"' in connection.calls[0] for name in TABLES)
    assert stamps == []


@pytest.mark.parametrize("versions", [("another_revision",), (BASELINE_REVISION, "branch_head")])
def test_adoption_refuses_to_overwrite_existing_versions(adoption_setup, monkeypatch, versions):
    settings, stamps = adoption_setup
    connection = AdoptionConnection(versions)
    monkeypatch.setattr(adoption, "check_baseline", lambda *args: ())
    monkeypatch.setattr(
        adoption,
        "inspect",
        lambda conn: SimpleNamespace(
            has_table=lambda name, schema: True,
        ),
    )
    assert adoption.stamp_baseline(connection, settings)
    assert stamps == []


def test_adoption_registers_fixed_baseline_using_same_transaction(adoption_setup, monkeypatch):
    settings, stamps = adoption_setup
    connection = AdoptionConnection()

    def check(connection, schema):
        connection.events.append(("check", schema))
        return ()

    monkeypatch.setattr(adoption, "check_baseline", check)
    monkeypatch.setattr(
        adoption,
        "inspect",
        lambda conn: SimpleNamespace(
            has_table=lambda name, schema: False,
        ),
    )
    assert adoption.stamp_baseline(connection, settings) == ()
    config, revision = stamps[0]
    assert revision == BASELINE_REVISION
    assert config.attributes["connection"] is connection
    assert config.attributes["settings"] is settings
    assert connection.events[:2] == [
        ("migration_lock", "alembic_test"),
        ("check", "alembic_test"),
    ]
    assert connection.events[2][1].endswith("IN SHARE UPDATE EXCLUSIVE MODE")
    assert connection.events[3] == ("check", "alembic_test")


def test_adoption_of_already_registered_baseline_is_idempotent(adoption_setup, monkeypatch):
    settings, stamps = adoption_setup
    connection = AdoptionConnection((BASELINE_REVISION,))
    monkeypatch.setattr(adoption, "check_baseline", lambda *args: ())
    monkeypatch.setattr(
        adoption,
        "inspect",
        lambda conn: SimpleNamespace(
            has_table=lambda name, schema: True,
        ),
    )
    assert adoption.stamp_baseline(connection, settings) == ()
    assert stamps == []


def test_check_cli_uses_read_only_transaction_and_sanitizes_connection_failure(
    adoption_setup,
    monkeypatch,
    capsys,
):
    settings, _ = adoption_setup
    connection = AdoptionConnection()

    class Engine:
        disposed = False
        error = None

        @contextmanager
        def begin(self):
            if self.error is not None:
                raise self.error
            yield connection

        def dispose(self):
            self.disposed = True

    engine = Engine()
    monkeypatch.setattr(adoption, "get_settings", lambda: settings)
    monkeypatch.setattr(adoption, "create_engine", lambda *args, **kwargs: engine)
    monkeypatch.setattr(adoption, "check_baseline", lambda *args: ())
    assert adoption.main(["--check"]) == 0
    assert connection.calls == ["SET TRANSACTION READ ONLY"]
    assert engine.disposed
    capsys.readouterr()
    engine.error = SQLAlchemyError("password=private_password_value server=private_host_value")
    assert adoption.main(["--check"]) == 2
    output = capsys.readouterr().out
    assert "private_password_value" not in output and "private_host_value" not in output
