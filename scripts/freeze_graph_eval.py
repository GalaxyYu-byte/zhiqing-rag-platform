"""Freeze published graph-building inputs from PostgreSQL and verify the offline snapshot."""

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database.connection import database_connection

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "graph-build-input-v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(json_bytes(row) for row in rows)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def selected_documents(selection: dict) -> list[dict]:
    if selection.get("schema_version") != SCHEMA_VERSION or not selection.get("documents"):
        raise ValueError("评估文档清单缺少版本或文档")
    documents = selection["documents"]
    ids = [item["document_id"] for item in documents]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("文档 ID 必须唯一且升序")
    for item in documents:
        path = ROOT / item["source_path"]
        if not path.is_file() or not path.resolve().is_relative_to((ROOT / "dox").resolve()):
            raise ValueError(f"来源文件缺失或不在 dox 目录: {item['source_path']}")
        if digest(path.read_bytes()) != item["source_sha256"]:
            raise ValueError(f"来源文件哈希不匹配: {item['source_path']}")
    return documents


def read_snapshot(settings: Settings, selection: dict) -> tuple[list[dict], list[dict]]:
    chosen = selected_documents(selection)
    documents = []
    chunks = []
    schema = settings.db_schema  # Settings enforces [a-z][a-z0-9_]*.
    with database_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            for item in chosen:
                cursor.execute(
                    f"""SELECT d.tenant_id, d.kb_id, d.id AS document_id, d.title,
                               d.desired_revision_id, d.active_generation_id,
                               g.revision_id, g.status AS generation_status, g.chunk_count,
                               g.parser_version, g.cleaner_version, g.chunker_version,
                               g.chunk_config, r.original_filename, r.file_sha256
                        FROM {schema}.document d
                        JOIN {schema}.index_generation g
                          ON g.id = d.active_generation_id AND g.tenant_id = d.tenant_id
                        JOIN {schema}.document_revision r
                          ON r.id = g.revision_id AND r.tenant_id = d.tenant_id
                        WHERE d.id = %s AND d.tenant_id = %s AND d.kb_id = %s
                          AND d.status = 'PUBLISHED' AND d.deleted_at IS NULL""",
                    (item["document_id"], item["tenant_id"], item["kb_id"]),
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError(f"文档不再处于已发布状态: {item['document_id']}")
                if (
                    row["generation_status"] != "READY"
                    or row["desired_revision_id"] != row["revision_id"]
                    or row["revision_id"] != item["revision_id"]
                    or row["active_generation_id"] != item["generation_id"]
                    or row["file_sha256"] != item["source_sha256"]
                    or row["chunk_count"] != item["expected_chunks"]
                ):
                    raise ValueError(f"修订、批次、来源或块数发生变化: {item['document_id']}")
                documents.append(
                    {
                        "tenant_id": row["tenant_id"],
                        "kb_id": row["kb_id"],
                        "document_id": row["document_id"],
                        "title": row["title"],
                        "revision_id": row["revision_id"],
                        "generation_id": row["active_generation_id"],
                        "source_path": item["source_path"],
                        "source_filename": row["original_filename"],
                        "source_sha256": row["file_sha256"],
                        "format": item["format"],
                        "selection_reason": item["selection_reason"],
                        "parser_version": row["parser_version"],
                        "cleaner_version": row["cleaner_version"],
                        "chunker_version": row["chunker_version"],
                        "chunk_config": row["chunk_config"],
                        "chunk_count": row["chunk_count"],
                    }
                )
                cursor.execute(
                    f"""SELECT id, tenant_id, kb_id, document_id, revision_id,
                               generation_id, chunk_index, content, content_sha256,
                               page_number, section_title, start_offset, end_offset, metadata
                        FROM {schema}.document_chunk
                        WHERE tenant_id = %s AND generation_id = %s
                        ORDER BY chunk_index""",
                    (item["tenant_id"], item["generation_id"]),
                )
                for chunk in cursor.fetchall():
                    if (
                        chunk["document_id"] != item["document_id"]
                        or chunk["revision_id"] != item["revision_id"]
                        or chunk["kb_id"] != item["kb_id"]
                    ):
                        raise ValueError(f"Chunk 归属不匹配: {chunk['id']}")
                    chunks.append(
                        {
                            key: chunk[key]
                            for key in (
                                "id",
                                "tenant_id",
                                "kb_id",
                                "document_id",
                                "revision_id",
                                "generation_id",
                                "chunk_index",
                                "content",
                                "content_sha256",
                                "page_number",
                                "section_title",
                                "start_offset",
                                "end_offset",
                            )
                        }
                        | {"source_refs": chunk["metadata"].get("source_refs", [])}
                    )
    validate_rows(documents, chunks)
    return documents, chunks


def validate_rows(documents: list[dict], chunks: list[dict]) -> None:
    if not documents or len({row["document_id"] for row in documents}) != len(documents):
        raise ValueError("文档为空或 ID 重复")
    if len({row["id"] for row in chunks}) != len(chunks):
        raise ValueError("Chunk ID 重复")
    by_generation = {(row["tenant_id"], row["generation_id"]): row for row in documents}
    indices: dict[tuple[int, int], list[int]] = {key: [] for key in by_generation}
    for chunk in chunks:
        key = (chunk["tenant_id"], chunk["generation_id"])
        document = by_generation.get(key)
        if document is None or any(
            chunk[field] != document[field] for field in ("kb_id", "document_id", "revision_id")
        ):
            raise ValueError(f"Chunk 归属不在冻结文档中: {chunk['id']}")
        if (
            not chunk["content"]
            or digest(chunk["content"].encode("utf-8")) != chunk["content_sha256"]
        ):
            raise ValueError(f"Chunk 正文哈希不匹配: {chunk['id']}")
        indices[key].append(chunk["chunk_index"])
    for key, document in by_generation.items():
        if sorted(indices[key]) != list(range(document["chunk_count"])):
            raise ValueError(f"Chunk 不完整或序号不连续: {key}")


def freeze(selection_path: Path, output: Path, settings: Settings) -> None:
    selection_data = selection_path.read_bytes()
    selection = json.loads(selection_data)
    documents, chunks = read_snapshot(settings, selection)
    files = {
        "selection.json": selection_data,
        "documents.jsonl": jsonl_bytes(documents),
        "chunks.jsonl": jsonl_bytes(chunks),
    }
    for document in documents:
        name = f"sources/{document['document_id']}{Path(document['source_path']).suffix.lower()}"
        files[name] = (ROOT / document["source_path"]).read_bytes()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": selection["dataset_id"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "selection_sha256": digest(selection_data),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "formats": dict(sorted(Counter(row["format"] for row in documents).items())),
        "files": {name: digest(data) for name, data in files.items()},
        "annotation_status": "unlabeled",
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    verify(output)
    print(f"已冻结 {len(documents)} 份文档、{len(chunks)} 个 Chunk: {output}")


def verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("冻结集版本不匹配")
    for name, expected in manifest["files"].items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError(f"冻结文件哈希不匹配: {name}")
    selection_data = (output / "selection.json").read_bytes()
    if digest(selection_data) != manifest["selection_sha256"]:
        raise ValueError("选择清单哈希不匹配")
    selection = json.loads(selection_data)
    documents = load_jsonl(output / "documents.jsonl")
    chunks = load_jsonl(output / "chunks.jsonl")
    validate_rows(documents, chunks)
    if selection["dataset_id"] != manifest["dataset_id"] or len(selection["documents"]) != len(
        documents
    ):
        raise ValueError("选择清单与冻结文档不匹配")
    for chosen, document in zip(selection["documents"], documents, strict=True):
        if (
            any(
                chosen[key] != document[key]
                for key in (
                    "tenant_id",
                    "kb_id",
                    "document_id",
                    "revision_id",
                    "generation_id",
                    "source_path",
                    "source_sha256",
                    "format",
                    "selection_reason",
                )
            )
            or chosen["expected_chunks"] != document["chunk_count"]
        ):
            raise ValueError(f"选择清单与冻结文档不匹配: {document['document_id']}")
        source_name = (
            f"sources/{document['document_id']}{Path(document['source_path']).suffix.lower()}"
        )
        if digest((output / source_name).read_bytes()) != document["source_sha256"]:
            raise ValueError(f"来源文件与文档哈希不匹配: {document['document_id']}")
    if (len(documents), len(chunks)) != (manifest["document_count"], manifest["chunk_count"]):
        raise ValueError("冻结集计数不匹配")
    if dict(sorted(Counter(row["format"] for row in documents).items())) != manifest["formats"]:
        raise ValueError("文件格式分布不匹配")
    print(f"校验通过：{len(documents)} 份文档、{len(chunks)} 个 Chunk")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--selection", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.selection, args.output, Settings())
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
