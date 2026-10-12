"""Freeze the existing 28-table PostgreSQL schema, including its business triggers.

Revision ID: 20261011_0001
Revises:
Create Date: 2026-10-11
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from alembic import context, op

revision = "20261011_0001"
down_revision = None
branch_labels = None
depends_on = None

_SOURCE_SCHEMA = "zhiqing_rag"
_BASELINE_PATH = Path(__file__).resolve().parents[1] / "baseline_20261011.json"


def _baseline() -> dict:
    # The revision uses frozen catalog definitions, never the current ORM models.
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def _target_schema() -> str:
    schema = context.config.attributes.get("target_schema", _SOURCE_SCHEMA)
    if not isinstance(schema, str) or re.fullmatch(r"[a-z][a-z0-9_]*", schema) is None:
        raise ValueError("Migration schema must match ^[a-z][a-z0-9_]*$.")
    return schema


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _qualified(schema: str, name: str) -> str:
    return f"{_identifier(schema)}.{_identifier(name)}"


def _translate(statement: str, schema: str) -> str:
    # Also maps qualified references inside PL/pgSQL function bodies.
    return re.sub(r"\bzhiqing_rag\b", schema, statement)


def _column_ddl(column: dict, schema: str, sequences: dict) -> str:
    parts = [_identifier(column["name"]), _translate(column["type"], schema)]
    identity = column["identity"]
    if identity:
        mode = {"a": "ALWAYS", "d": "BY DEFAULT"}[identity]
        sequence_name = f"{column['table_name']}_{column['name']}_seq"
        sequence = sequences[sequence_name]
        cycle = "CYCLE" if sequence["cycle"] else "NO CYCLE"
        options = (
            f"SEQUENCE NAME {_qualified(schema, sequence_name)} "
            f"START WITH {sequence['start_value']} "
            f"INCREMENT BY {sequence['increment_by']} "
            f"MINVALUE {sequence['min_value']} MAXVALUE {sequence['max_value']} "
            f"CACHE {sequence['cache_size']} {cycle}"
        )
        parts.append(f"GENERATED {mode} AS IDENTITY ({options})")
    elif column["default"] is not None:
        parts.append("DEFAULT " + _translate(column["default"], schema))
    if column["not_null"]:
        parts.append("NOT NULL")
    return " ".join(parts)


def upgrade() -> None:
    baseline = _baseline()
    schema = _target_schema()
    columns = defaultdict(list)
    constraints = defaultdict(list)
    sequences = {sequence["name"]: sequence for sequence in baseline["sequences"]}
    for column in baseline["columns"]:
        columns[column["table_name"]].append(column)
    for constraint in baseline["constraints"]:
        constraints[constraint["table_name"]].append(constraint)

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {_identifier(schema)}")
    for table in baseline["tables"]:
        table_name = table["name"]
        definitions = [_column_ddl(column, schema, sequences) for column in columns[table_name]]
        definitions.extend(
            f"CONSTRAINT {_identifier(constraint['name'])} "
            + _translate(constraint["definition"], schema)
            for constraint in constraints[table_name]
            if constraint["type"] != "f"
        )
        op.execute(
            f"CREATE TABLE {_qualified(schema, table_name)} (\n    "
            + ",\n    ".join(definitions)
            + "\n)"
        )
        if table["comment"] is not None:
            op.execute(
                f"COMMENT ON TABLE {_qualified(schema, table_name)} IS {_literal(table['comment'])}"
            )
        for column in columns[table_name]:
            if column["comment"] is not None:
                op.execute(
                    f"COMMENT ON COLUMN {_qualified(schema, table_name)}."
                    f"{_identifier(column['name'])} IS {_literal(column['comment'])}"
                )

    # All referenced tables and unique keys exist before adding the cyclic foreign keys.
    for constraint in baseline["constraints"]:
        if constraint["type"] == "f":
            op.execute(
                f"ALTER TABLE {_qualified(schema, constraint['table_name'])} "
                f"ADD CONSTRAINT {_identifier(constraint['name'])} "
                + _translate(constraint["definition"], schema)
            )
    for index in baseline["indexes"]:
        if index["constraint_name"] is None:
            op.execute(_translate(index["definition"], schema))
    for function in baseline["functions"]:
        op.execute(_translate(function["definition"], schema))
    trigger_states = {
        "O": "ENABLE TRIGGER",
        "D": "DISABLE TRIGGER",
        "R": "ENABLE REPLICA TRIGGER",
        "A": "ENABLE ALWAYS TRIGGER",
    }
    for trigger in baseline["triggers"]:
        op.execute(_translate(trigger["definition"], schema))
        op.execute(
            f"ALTER TABLE {_qualified(schema, trigger['table_name'])} "
            f"{trigger_states[trigger['enabled']]} {_identifier(trigger['name'])}"
        )


def downgrade() -> None:
    baseline = _baseline()
    schema = _target_schema()
    for trigger in reversed(baseline["triggers"]):
        op.execute(
            f"DROP TRIGGER {_identifier(trigger['name'])} "
            f"ON {_qualified(schema, trigger['table_name'])}"
        )
    for function in reversed(baseline["functions"]):
        # All ten frozen functions are zero-argument trigger functions.
        op.execute(f"DROP FUNCTION {_qualified(schema, function['name'])}()")
    for constraint in reversed(baseline["constraints"]):
        if constraint["type"] == "f":
            op.execute(
                f"ALTER TABLE {_qualified(schema, constraint['table_name'])} "
                f"DROP CONSTRAINT {_identifier(constraint['name'])}"
            )
    for table in reversed(baseline["tables"]):
        op.execute(f"DROP TABLE {_qualified(schema, table['name'])}")
    # Keep the schema and shared extensions; dependencies outside this revision must block drops.
