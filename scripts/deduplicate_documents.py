"""按当前版本 Chunk 正文哈希审计并软删除完全重复的文档。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import defaultdict

from sqlalchemy import select, update

from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.models.document import DocChunk, Document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计完全重复文档")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="软删除每组中除最小 doc_id 外的副本；默认只审计",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    async with SessionLocal() as session:
        rows = await session.execute(
            select(
                Document.id,
                Document.file_name,
                Document.document_code,
                DocChunk.chunk_index,
                DocChunk.content,
            )
            .join(
                DocChunk,
                (DocChunk.doc_id == Document.id)
                & (DocChunk.doc_version == Document.version),
            )
            .where(
                Document.kb_id == args.kb_id,
                Document.is_deleted.is_(False),
                Document.status == "DONE",
            )
            .order_by(Document.id, DocChunk.chunk_index)
        )
        documents: dict[int, dict[str, object]] = {}
        for row in rows.all():
            item = documents.setdefault(
                int(row.id),
                {
                    "doc_id": int(row.id),
                    "file_name": str(row.file_name),
                    "document_code": row.document_code,
                    "chunks": [],
                },
            )
            item["chunks"].append(str(row.content))

        groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for item in documents.values():
            content = "\n\x1e\n".join(item.pop("chunks"))
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            item["content_sha256"] = digest
            groups[(str(item["file_name"]), digest)].append(item)
        duplicates = [
            sorted(group, key=lambda item: int(item["doc_id"]))
            for group in groups.values()
            if len(group) > 1
        ]
        duplicate_ids = [
            int(item["doc_id"])
            for group in duplicates
            for item in group[1:]
        ]
        if args.apply and duplicate_ids:
            await session.execute(
                update(Document)
                .where(
                    Document.kb_id == args.kb_id,
                    Document.id.in_(duplicate_ids),
                    Document.is_deleted.is_(False),
                )
                .values(is_deleted=True)
            )
            await session.commit()
        payload = {
            "kb_id": args.kb_id,
            "mode": "apply" if args.apply else "audit",
            "duplicate_groups": duplicates,
            "soft_deleted_doc_ids": duplicate_ids if args.apply else [],
            "recoverable": True,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))


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
