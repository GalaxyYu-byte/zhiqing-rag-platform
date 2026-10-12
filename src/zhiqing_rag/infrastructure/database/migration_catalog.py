"""只读核对冻结数据库基线，覆盖 ORM 未声明的函数、触发器及索引状态。"""

import json
import re
from collections import defaultdict

from sqlalchemy import text

from zhiqing_rag.core.config import PROJECT_ROOT

_SOURCE_SCHEMA = "zhiqing_rag"
_CATALOG_QUERIES = {
    "tables": """
        SELECT c.relname AS name, pg_catalog.obj_description(c.oid, 'pg_class') AS comment
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relkind IN ('r', 'p', 'f')
          AND c.relname <> 'alembic_version'
        ORDER BY c.relname
    """,
    "columns": """
        SELECT c.relname AS table_name, a.attname AS name,
               pg_catalog.format_type(a.atttypid, a.atttypmod) AS type,
               a.attnotnull AS not_null,
               pg_catalog.pg_get_expr(d.adbin, d.adrelid) AS default,
               a.attidentity AS identity,
               pg_catalog.col_description(c.oid, a.attnum) AS comment
        FROM pg_catalog.pg_class c
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid
        LEFT JOIN pg_catalog.pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
        WHERE n.nspname = :schema AND c.relkind IN ('r', 'p', 'f')
          AND c.relname <> 'alembic_version' AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY c.relname, a.attnum
    """,
    "constraints": """
        SELECT c.relname AS table_name, con.conname AS name, con.contype AS type,
               pg_catalog.pg_get_constraintdef(con.oid, true) AS definition
        FROM pg_catalog.pg_constraint con
        JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname <> 'alembic_version'
        ORDER BY c.relname, con.conname
    """,
    "indexes": """
        SELECT t.relname AS table_name, c.relname AS name,
               pg_catalog.pg_get_indexdef(c.oid) AS definition,
               i.indisvalid AS valid, i.indisready AS ready,
               con.conname AS constraint_name
        FROM pg_catalog.pg_index i
        JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
        JOIN pg_catalog.pg_class t ON t.oid = i.indrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
        LEFT JOIN pg_catalog.pg_constraint con
          ON con.conindid = c.oid AND con.contype IN ('p', 'u', 'x')
        WHERE n.nspname = :schema AND t.relname <> 'alembic_version'
        ORDER BY t.relname, c.relname
    """,
    "functions": """
        SELECT p.proname AS name, pg_catalog.pg_get_functiondef(p.oid) AS definition
        FROM pg_catalog.pg_proc p
        JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = :schema AND p.prokind = 'f'
        ORDER BY p.proname, p.oid
    """,
    "triggers": """
        SELECT c.relname AS table_name, t.tgname AS name,
               pg_catalog.pg_get_triggerdef(t.oid, true) AS definition,
               t.tgenabled AS enabled
        FROM pg_catalog.pg_trigger t
        JOIN pg_catalog.pg_class c ON c.oid = t.tgrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND NOT t.tgisinternal
          AND c.relname <> 'alembic_version'
        ORDER BY c.relname, t.tgname
    """,
    "views": """
        SELECT viewname AS name, definition FROM pg_catalog.pg_views
        WHERE schemaname = :schema
        UNION ALL
        SELECT matviewname AS name, definition FROM pg_catalog.pg_matviews
        WHERE schemaname = :schema
        ORDER BY name
    """,
    "sequences": """
        SELECT sequencename AS name, start_value, increment_by,
               min_value, max_value, cache_size, cycle
        FROM pg_catalog.pg_sequences
        WHERE schemaname = :schema
        ORDER BY sequencename
    """,
}
_CATEGORY_NAMES = {
    "tables": "表",
    "columns": "字段",
    "constraints": "约束",
    "indexes": "索引",
    "functions": "函数",
    "triggers": "触发器",
    "views": "视图",
    "sequences": "序列",
}
_SQL_QUOTED = re.compile(
    r"'(?:''|\\[\s\S]|[^'])*'"
    r'|"(?:""|[^\"])*"'
    r"|(?P<dollar>\$(?:[a-zA-Z_][a-zA-Z_0-9]*)?\$)[\s\S]*?(?P=dollar)"
)
_BLANK_LINES = re.compile(r"\n(?:[ \t]*\n)+")
_TEXT_LITERAL = r"'(?:''|[^'])*'"
_TEXT_TYPE = r"(?:character\s+varying|varchar|text)"
_TEXT_ARRAY_ITEM = (
    rf"(?:\(\s*{_TEXT_LITERAL}\s*::\s*{_TEXT_TYPE}\s*\)\s*::\s*text"
    rf"|{_TEXT_LITERAL}\s*::\s*{_TEXT_TYPE}(?:\s*::\s*text)?)"
)
_TEXT_ARRAY_COMPARISON = re.compile(
    rf"\b(?P<quantifier>ANY|ALL)\s*\(\s*(?P<opening>\()?\s*ARRAY\s*\[\s*"
    rf"(?P<items>{_TEXT_ARRAY_ITEM}(?:\s*,\s*{_TEXT_ARRAY_ITEM})*)\s*\]"
    r"\s*(?(opening)\))\s*(?P<cast>::\s*text\s*\[\])?\s*\)",
    re.IGNORECASE,
)


def _validate_schema(schema: str) -> None:
    if not isinstance(schema, str) or re.fullmatch(r"[a-z][a-z0-9_]*", schema) is None:
        raise ValueError("数据库 schema 必须匹配 ^[a-z][a-z0-9_]*$。")


def read_catalog(connection, schema: str) -> dict:
    """仅查询系统目录；调用方先设 search_path 为 public, pg_catalog 并开启只读事务。"""
    _validate_schema(schema)
    return {
        category: [
            dict(row)
            for row in connection.execute(
                text(query),
                {"schema": schema},
            ).mappings()
        ]
        for category, query in _CATALOG_QUERIES.items()
    }


def load_baseline() -> dict:
    """读取与初始迁移一起冻结的目录，不加载当前 ORM 或数据库配置。"""
    baseline_path = PROJECT_ROOT / "alembic" / "baseline_20261011.json"
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def _collapse_fragment(statement: str) -> str:
    """规范 SQL 代码的重复空行，保留单引号、标识符及 dollar 字符串中的内容。"""
    parts = []
    position = 0
    for match in _SQL_QUOTED.finditer(statement):
        parts.append(_BLANK_LINES.sub("\n", statement[position : match.start()]))
        parts.append(match.group())
        position = match.end()
    parts.append(_BLANK_LINES.sub("\n", statement[position:]))
    return "".join(parts)


def _normalize_sql(statement: str) -> str:
    # 函数最外层 dollar 引号包裹的是程序代码；其中嵌套的 SQL 字符串仍须保留。
    if statement.lstrip().upper().startswith("CREATE OR REPLACE FUNCTION"):
        match = re.search(r"\bAS\s+(\$(?:[a-zA-Z_][a-zA-Z_0-9]*)?\$)", statement)
        if match is not None:
            closing = statement.rfind(match.group(1))
            if closing >= match.end():
                return (
                    _collapse_fragment(statement[: match.end()])
                    + _collapse_fragment(statement[match.end() : closing])
                    + _collapse_fragment(statement[closing:])
                )
    return _collapse_fragment(statement)


def _normalize_text_arrays(statement: str) -> str:
    """兼容重放 DDL 后文本数组的 cast 展示；保留字面量、顺序和非文本类型。"""
    # 不用于函数正文，也不把标识符或字符串里类似 ARRAY 的内容解释成 SQL。
    if not re.match(r"(?:CHECK\b|CREATE\s+(?:UNIQUE\s+)?INDEX\b)", statement):
        return statement
    protected = [(match.start(), match.end()) for match in _SQL_QUOTED.finditer(statement)]

    def normalize(match):
        if any(start <= match.start() < end for start, end in protected):
            return match.group()
        items = list(re.finditer(_TEXT_ARRAY_ITEM, match["items"], re.IGNORECASE))
        if not match["cast"] and not all(
            re.search(r"::\s*text\s*$", item.group(), re.IGNORECASE) for item in items
        ):
            return match.group()
        values = [re.search(_TEXT_LITERAL, item.group()).group() for item in items]
        rendered = ", ".join(f"{value}::text" for value in values)
        return f"{match['quantifier'].upper()} (ARRAY[{rendered}])"

    return _TEXT_ARRAY_COMPARISON.sub(normalize, statement)


def _normalize(value, schema: str, field: str = ""):
    if isinstance(value, str):
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(rf"\b{re.escape(schema)}\b", _SOURCE_SCHEMA, normalized)
        if field in {"definition", "default"}:
            normalized = _normalize_sql(normalized)
        return _normalize_text_arrays(normalized) if field == "definition" else normalized
    if isinstance(value, list):
        return [_normalize(item, schema, field) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item, schema, key) for key, item in value.items()}
    return value


def _objects(rows: list) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["table_name"], row["name"]) if "table_name" in row else (row["name"],)
        grouped[key].append(row)
    # 重载函数或重复对象也会参与比较，不能因相同名称被字典覆盖而漏检。
    return {
        key: sorted(values, key=lambda value: json.dumps(value, sort_keys=True))
        for key, values in grouped.items()
    }


def _display_name(key: tuple) -> str:
    return ".".join(re.sub(r"[\x00-\x1f\x7f]", " ", part) for part in key)


def compare_baseline(actual: dict, schema: str) -> tuple[str, ...]:
    """逐对象核对全部冻结目录，只报告对象名称，不输出函数正文或业务数据。"""
    _validate_schema(schema)
    baseline = _normalize(load_baseline(), _SOURCE_SCHEMA)
    normalized = _normalize(actual, schema)
    differences = []
    for category, label in _CATEGORY_NAMES.items():
        expected_objects = _objects(baseline[category])
        actual_objects = _objects(normalized.get(category, []))
        for key in sorted(expected_objects.keys() - actual_objects.keys()):
            differences.append(f"缺少{label}：{_display_name(key)}")
        for key in sorted(actual_objects.keys() - expected_objects.keys()):
            differences.append(f"多余{label}：{_display_name(key)}")
        for key in sorted(expected_objects.keys() & actual_objects.keys()):
            if expected_objects[key] != actual_objects[key]:
                differences.append(f"{label}定义不符：{_display_name(key)}")
    return tuple(differences)
