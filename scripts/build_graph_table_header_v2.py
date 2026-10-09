"""Rechunk frozen XLSX sources into an isolated table-header v2 pilot."""

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from experiment_graph_table_extraction import extract_tables, read_jsonl

from zhiqing_rag.document_processing import ChunkConfig, process_document
from zhiqing_rag.workers.graph_extraction import Extraction, normalize_name, validate_evidence

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "eval/graph_build/v1"
GOLD = ROOT / "eval/graph_build/gold-v1/cases.jsonl"
OUTPUT = ROOT / "eval/graph_build/v2-table-header-pilot"
TARGET_DOCUMENT_IDS = {83, 89}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rows_to_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def build() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"试验版本已存在: {OUTPUT}")
    documents = [
        row
        for row in read_jsonl(V1 / "documents.jsonl")
        if row["document_id"] in TARGET_DOCUMENT_IDS
    ]
    chunks = []
    proposals = []
    for document in documents:
        document_id = document["document_id"]
        source = V1 / "sources" / f"{document_id}.xlsx"
        if digest(source.read_bytes()) != document["source_sha256"]:
            raise ValueError(f"v1 来源文件哈希不一致: {document_id}")
        config = ChunkConfig(**document["chunk_config"])
        processed = process_document(
            source,
            original_filename=document["source_filename"],
            chunk_config=config,
        )
        for draft in processed.chunks:
            chunk = {
                "local_chunk_id": f"v2-{document_id}-{draft.chunk_index:02d}",
                "document_id": document_id,
                "chunk_index": draft.chunk_index,
                "content": draft.content,
                "content_sha256": draft.content_sha256,
                "token_count": draft.token_count,
                "metadata": draft.metadata,
            }
            if digest(chunk["content"].encode("utf-8")) != chunk["content_sha256"]:
                raise ValueError(f"新 Chunk 哈希错误: {chunk['local_chunk_id']}")
            chunks.append(chunk)
            if draft.metadata.get("table_preamble"):
                continue
            header = (
                "办理岗位应做" if document_id == 83 else "资料类型 | 主要读者"
            )
            if header not in draft.content or draft.metadata.get("table_header_row") != 3:
                raise ValueError(f"数据 Chunk 缺少真实列头: {chunk['local_chunk_id']}")
            extraction, _, header_in_chunk = extract_tables(chunk, None)
            if not header_in_chunk:
                raise ValueError(f"表格抽取依赖跨 Chunk 表头: {chunk['local_chunk_id']}")
            by_id = {entity.id: entity for entity in extraction.entities}
            for relation in extraction.relations:
                if (
                    by_id[relation.source_id].name not in relation.evidence_quote
                    or by_id[relation.target_id].name not in relation.evidence_quote
                ):
                    raise ValueError(f"关系证据未同时包含两端: {chunk['local_chunk_id']}")
            if extraction.relations:
                proposals.append(
                    {
                        "local_chunk_id": chunk["local_chunk_id"],
                        "document_id": document_id,
                        "extraction": extraction.model_dump(mode="json"),
                    }
                )
    gold_cases = [case for case in read_jsonl(GOLD) if case["document_id"] in TARGET_DOCUMENT_IDS]
    proposed = Counter()
    for row in proposals:
        by_id = {entity["id"]: entity for entity in row["extraction"]["entities"]}
        for relation in row["extraction"]["relations"]:
            source = by_id[relation["source_id"]]
            target = by_id[relation["target_id"]]
            proposed[
                (
                    row["document_id"],
                    source["type"],
                    normalize_name(source["canonical_name"]),
                    relation["type"],
                    target["type"],
                    normalize_name(target["canonical_name"]),
                )
            ] += 1
    outcomes = []
    for case in gold_cases:
        by_id = {entity["id"]: entity for entity in case["entities"]}
        for index, relation in enumerate(case["relations"], start=1):
            source = by_id[relation["source_id"]]
            target = by_id[relation["target_id"]]
            signature = (
                case["document_id"],
                source["type"],
                normalize_name(source["name"]),
                relation["type"],
                target["type"],
                normalize_name(target["name"]),
            )
            matched = proposed[signature] > 0
            if matched:
                proposed[signature] -= 1
            outcomes.append(
                {
                    "gold_relation_id": f"{case['case_id']}:r{index}",
                    "document_id": case["document_id"],
                    "matched": matched,
                }
            )
    if not all(row["matched"] for row in outcomes):
        raise ValueError("v2 表格候选未覆盖全部目标 Gold 关系")
    metadata = {
        "schema_version": "graph-build-table-header-pilot-v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": digest((V1 / "manifest.json").read_bytes()),
        "gold_cases_sha256": digest(GOLD.read_bytes()),
        "chunker_version": "chunking-v2",
        "document_ids": sorted(TARGET_DOCUMENT_IDS),
        "chunk_count": len(chunks),
        "data_chunk_count": sum(not row["metadata"].get("table_preamble") for row in chunks),
        "proposal_relation_count": sum(len(row["extraction"]["relations"]) for row in proposals),
        "gold_relation_matches": sum(row["matched"] for row in outcomes),
        "gold_relation_count": len(outcomes),
        "mode": "local_rechunk_and_offline_evidence_validation_no_graph_writes",
    }
    files = {
        "chunks.jsonl": rows_to_bytes(chunks),
        "proposals.jsonl": rows_to_bytes(proposals),
        "gold_relation_outcomes.jsonl": rows_to_bytes(outcomes),
    }
    metadata["files"] = {name: digest(data) for name, data in files.items()}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (OUTPUT / name).write_bytes(data)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


def verify() -> None:
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_manifest_sha256"] != digest((V1 / "manifest.json").read_bytes()):
        raise ValueError("v1 来源版本已变化")
    for name, expected in manifest["files"].items():
        if digest((OUTPUT / name).read_bytes()) != expected:
            raise ValueError(f"v2 试验文件哈希不一致: {name}")
    chunks = read_jsonl(OUTPUT / "chunks.jsonl")
    proposals = read_jsonl(OUTPUT / "proposals.jsonl")
    outcomes = read_jsonl(OUTPUT / "gold_relation_outcomes.jsonl")
    if len(chunks) != manifest["chunk_count"] or len(proposals) == 0:
        raise ValueError("v2 试验集计数不一致")
    if sum(row["matched"] for row in outcomes) != manifest["gold_relation_matches"]:
        raise ValueError("v2 Gold 对照计数不一致")
    documents = {
        row["document_id"]: row
        for row in read_jsonl(V1 / "documents.jsonl")
        if row["document_id"] in TARGET_DOCUMENT_IDS
    }
    for document_id, document in documents.items():
        source = V1 / "sources" / f"{document_id}.xlsx"
        if digest(source.read_bytes()) != document["source_sha256"]:
            raise ValueError(f"v1 来源文件哈希不一致: {document_id}")
        current = process_document(
            source,
            original_filename=document["source_filename"],
            chunk_config=ChunkConfig(**document["chunk_config"]),
        ).chunks
        saved = sorted(
            (row for row in chunks if row["document_id"] == document_id),
            key=lambda row: row["chunk_index"],
        )
        if len(current) != len(saved) or any(
            draft.content != row["content"]
            or draft.content_sha256 != row["content_sha256"]
            or draft.token_count != row["token_count"]
            or draft.metadata != row["metadata"]
            for draft, row in zip(current, saved, strict=True)
        ):
            raise ValueError(f"v2 重分块无法复现: {document_id}")
    chunks_by_id = {row["local_chunk_id"]: row for row in chunks}
    for proposal in proposals:
        chunk = chunks_by_id[proposal["local_chunk_id"]]
        extraction = Extraction.model_validate(proposal["extraction"])
        validate_evidence(extraction, chunk["content"])
        if chunk["metadata"].get("table_header_row") != 3:
            raise ValueError(f"v2 数据 Chunk 缺失表头元数据: {proposal['local_chunk_id']}")
        by_id = {entity.id: entity for entity in extraction.entities}
        for relation in extraction.relations:
            if (
                by_id[relation.source_id].name not in relation.evidence_quote
                or by_id[relation.target_id].name not in relation.evidence_quote
            ):
                raise ValueError(f"v2 关系证据不完整: {proposal['local_chunk_id']}")
    print(f"v2 试验集校验通过：{len(chunks)} Chunk，{len(outcomes)} Gold 关系")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    arguments = parser.parse_args()
    if arguments.command == "build":
        build()
        verify()
    else:
        verify()
