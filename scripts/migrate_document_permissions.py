"""幂等新增文档显式共享表及其索引，不更改已有授权或文档密级。"""

import argparse

from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.modules import documents as _documents  # noqa: F401
from zhiqing_rag.modules.documents.document_access_grant import DocumentAccessGrant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="仅显示新增表和索引 SQL")
    arguments = parser.parse_args()
    table = DocumentAccessGrant.__table__
    if arguments.dry_run:
        dialect = postgresql.dialect()
        print(str(CreateTable(table).compile(dialect=dialect)) + ";")
        for index in sorted(table.indexes, key=lambda value: value.name):
            print(str(CreateIndex(index).compile(dialect=dialect)) + ";")
        return
    engine = create_engine(get_settings().database_url)
    try:
        with engine.begin() as connection:
            table.create(connection, checkfirst=True)
            # 表存在时 create(checkfirst) 不补索引；显式逐项确保幂等可恢复。
            for index in sorted(table.indexes, key=lambda value: value.name):
                index.create(connection, checkfirst=True)
        print("document_access_grant 表及索引已就绪，已有文档和授权未修改。")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
