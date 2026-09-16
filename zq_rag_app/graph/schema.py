"""Neo4j 图 Schema 资源加载。"""

from __future__ import annotations

from pathlib import Path


SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "neo4j_constraints.cypher"


def load_schema_statements(path: Path = SCHEMA_PATH) -> tuple[str, ...]:
    """读取 Cypher 文件并返回可逐条执行的 DDL。"""

    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]
    return tuple(
        statement.strip()
        for statement in "\n".join(lines).split(";")
        if statement.strip()
    )
