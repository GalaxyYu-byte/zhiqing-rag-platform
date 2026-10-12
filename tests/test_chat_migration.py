"""聊天迁移 DDL 与只读准备检查；替身不连接任何数据库。"""

from contextlib import contextmanager
from copy import deepcopy

import pytest
from sqlalchemy.exc import SQLAlchemyError

from scripts import migrate_chat_persistence as migration
from zhiqing_rag.core.config import Settings


def indexes():
    rows = [
        ("chat_session", "ix_chat_owner", ["tenant_id", "owner_member_id", "last_active_at"],
         False, None),
        ("chat_run", "ix_chat_run_history", ["tenant_id", "session_id", "created_at"],
         False, None),
        ("chat_run", "ix_chat_run_lease", ["lease_expires_at"], False, "status = 'RUNNING'"),
        ("chat_run", "uq_chat_one_active_run", ["tenant_id", "session_id"], True,
         "status IN ('PENDING', 'RUNNING')"),
        ("answer_citation", "ix_citation_document", ["tenant_id", "document_id", "revision_id"],
         False, None),
    ]
    return [{"table_name": table, "index_name": name, "columns": columns, "is_unique": unique,
             "predicate": predicate, "valid": True, "ready": True}
            for table, name, columns, unique, predicate in rows]


class Result:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return self.rows

    def mappings(self):
        return self.rows


class Connection:
    def __init__(self):
        self.tables = ["chat_session", "chat_run", "chat_message", "answer_citation"]
        self.indexes = indexes()
        self.calls = []

    def execute(self, statement, parameters):
        rendered = str(statement)
        assert rendered.strip().startswith("SELECT")
        self.calls.append((rendered, deepcopy(parameters)))
        return Result(self.indexes if "pg_catalog.pg_index" in rendered else self.tables)

    def exec_driver_sql(self, statement):
        assert statement == "SET TRANSACTION READ ONLY"
        self.calls.append((statement, None))


class Engine:
    def __init__(self, connection):
        self.connection = connection
        self.options = {}
        self.disposed = False
        self.error = None

    def execution_options(self, **options):
        self.options = options
        return self

    @contextmanager
    def connect(self):
        if self.error is not None:
            raise self.error
        yield self.connection

    def begin(self):
        raise AssertionError("只读检查不能打开迁移写事务")

    def dispose(self):
        self.disposed = True


@pytest.fixture
def fake_engine(monkeypatch):
    connection = Connection()
    engine = Engine(connection)
    settings = Settings(_env_file=None, db_schema="chat_custom")
    monkeypatch.setattr(migration, "get_settings", lambda: settings)
    monkeypatch.setattr(migration, "create_engine", lambda *args, **kwargs: engine)
    return engine, connection


def test_dry_run_translates_tables_foreign_keys_and_indexes_without_mutating_orm():
    statements = migration.render_ddl("chat_custom")
    assert len(statements) == 9
    combined = "\n".join(statements)
    assert "CREATE TABLE chat_custom.chat_session" in combined
    assert "REFERENCES chat_custom.tenant_member" in combined
    assert "REFERENCES chat_custom.document_revision" in combined
    assert "ON chat_custom.chat_run" in combined
    assert "zhiqing_rag." not in combined and "__[SCHEMA_" not in combined
    assert all(table.schema == "zhiqing_rag" for table in migration.chat_tables())


def test_catalog_check_finds_all_tables_and_valid_required_indexes():
    connection = Connection()
    assert migration.check_readiness(connection, "chat_custom").ready
    assert len(connection.calls) == 2
    for _, parameters in connection.calls:
        assert parameters["schema"] == "chat_custom"
        assert parameters["tables"] == connection.tables


def test_missing_tables_and_indexes_are_reported_separately():
    connection = Connection()
    connection.tables.remove("chat_message")
    connection.indexes = [row for row in connection.indexes
                          if row["index_name"] != "uq_chat_one_active_run"]
    status = migration.check_readiness(connection, "chat_custom")
    assert not status.ready
    assert status.missing_tables == ("chat_message",)
    assert status.missing_indexes == ("uq_chat_one_active_run",)
    assert status.unusable_indexes == ()


@pytest.mark.parametrize("changes", [
    {"valid": False}, {"ready": False}, {"is_unique": False}, {"predicate": None},
    {"columns": ["session_id", "tenant_id"]}, {"table_name": "another_table"},
    {"predicate": "status = 'RUNNING'"}, {"predicate": "status <> 'PENDING'"},
    {"predicate": "status IN ('PENDING', 'RUNNING') OR true"},
])
def test_invalid_or_mismatched_active_run_index_is_not_ready(changes):
    connection = Connection()
    active = next(row for row in connection.indexes
                  if row["index_name"] == "uq_chat_one_active_run")
    active.update(changes)
    status = migration.check_readiness(connection, "chat_custom")
    assert not status.ready
    assert "uq_chat_one_active_run" in status.unusable_indexes + status.missing_indexes


def test_postgres_in_to_any_and_varchar_casts_preserve_expected_active_predicate():
    actual = "((status)::text = ANY ((ARRAY['PENDING'::varchar, 'RUNNING'::varchar])::text[]))"
    assert migration._same_predicate(actual, "status IN ('PENDING','RUNNING')")
    assert not migration._same_predicate("status = 'running'", "status = 'RUNNING'")


def test_check_mode_is_read_only_and_disposes_engine(fake_engine, capsys):
    engine, connection = fake_engine
    assert migration.main(["--check"]) == 0
    assert connection.calls[0] == ("SET TRANSACTION READ ONLY", None)
    assert engine.options == {"schema_translate_map": {"zhiqing_rag": "chat_custom"}}
    assert engine.disposed
    assert "已就绪" in capsys.readouterr().out


def test_check_mode_has_nonzero_exit_code_for_missing_objects(fake_engine, capsys):
    engine, connection = fake_engine
    connection.tables = []
    connection.indexes = []
    assert migration.main(["--check"]) == 1
    output = capsys.readouterr().out
    assert "未就绪" in output and "缺少表" in output and "缺少索引" in output
    assert engine.disposed


def test_check_connection_error_does_not_expose_connection_data(fake_engine, capsys):
    engine, _ = fake_engine
    engine.error = SQLAlchemyError("password=secret-value server=private-address")
    assert migration.main(["--check"]) == 2
    output = capsys.readouterr().out
    assert "无法确认" in output
    assert "secret-value" not in output and "private-address" not in output
    assert engine.disposed


def test_dry_run_does_not_construct_database_engine(monkeypatch, capsys):
    monkeypatch.setattr(migration, "get_settings",
                        lambda: Settings(_env_file=None, db_schema="chat_custom"))

    def denied(*args, **kwargs):
        raise AssertionError("dry-run 不允许建立数据库连接")

    monkeypatch.setattr(migration, "create_engine", denied)
    assert migration.main(["--dry-run"]) == 0
    assert "chat_custom.chat_run" in capsys.readouterr().out


def test_check_and_dry_run_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        migration.main(["--check", "--dry-run"])
    assert error.value.code == 2
