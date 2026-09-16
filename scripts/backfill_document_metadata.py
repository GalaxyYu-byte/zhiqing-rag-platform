"""按 qa-docs/manifest.csv 回填已入库文档的部门、密级和业务状态。"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from pathlib import Path

from sqlalchemy import select, update

from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.models.document import Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回填文档访问元数据")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("qa-docs/manifest.csv")
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    by_name = {row["file_name"].strip(): row for row in manifest}
    async with SessionLocal() as session:
        existing_rows = await session.execute(
            select(Document.file_name).where(
                Document.kb_id == args.kb_id,
                Document.is_deleted.is_(False),
            )
        )
        existing_names = {str(row[0]) for row in existing_rows.all()}
        matched_names = sorted(existing_names & by_name.keys())
        for file_name in matched_names:
            row = by_name[file_name]
            await session.execute(
                update(Document)
                .where(
                    Document.kb_id == args.kb_id,
                    Document.file_name == file_name,
                    Document.is_deleted.is_(False),
                )
                .values(
                    document_code=row["document_id"].strip(),
                    department_id=row["department"].strip(),
                    confidentiality=row["confidentiality"].strip(),
                    business_status=row["status"].strip(),
                )
            )
        await session.commit()
    print(
        f"文档元数据回填完成: matched_files={len(matched_names)}, "
        f"unmatched_db_files={len(existing_names - by_name.keys())}"
    )
    unmatched_names = sorted(existing_names - by_name.keys())
    if unmatched_names:
        print("未匹配数据库文件: " + "、".join(unmatched_names))


async def _run_and_close() -> None:
    try:
        await main()
    finally:
        await close_database()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(_run_and_close())
    else:
        asyncio.run(_run_and_close())
