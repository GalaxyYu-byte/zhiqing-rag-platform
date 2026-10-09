"""Check rechunked headers and row-local evidence on unseen XLSX documents."""

import hashlib
import json
from pathlib import Path

from zhiqing_rag.document_processing import ChunkConfig, process_document
from zhiqing_rag.workers.graph_extraction import Extraction, validate_evidence

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "eval/graph_build/v1"
OUTPUT = ROOT / "eval/graph_build/v2-table-header-pilot/unseen-validation"
CONFIG = ChunkConfig(max_tokens=512, overlap_tokens=48)
RELATION_SCHEMAS = {
    "采购节点与交接说明.xlsx": ("主责角色", "Actor", "环节", "Step", "RESPONSIBLE_FOR"),
    "项目交付物核对清单.xlsx": ("文件或成果", "Material", "检查要点", "Rule", "GOVERNED_BY"),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells(line: str) -> list[str]:
    if not (line.startswith("| ") and line.endswith(" |")):
        return []
    return [cell.strip() for cell in line[1:-1].split("|")]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"留出样本结果已存在: {OUTPUT}")
    used_source_hashes = {
        digest(V1 / "sources" / name) for name in ("83.xlsx", "89.xlsx")
    }
    unseen = [
        path
        for path in sorted((ROOT / "dox").rglob("*.xlsx"))
        if digest(path) not in used_source_hashes
    ]
    if len(unseen) != 6:
        raise ValueError(f"预期 6 份未参与规则设计的 xlsx，实际 {len(unseen)}")
    checks = []
    facts = []
    for source in unseen:
        processed = process_document(source, chunk_config=CONFIG)
        for element in processed.cleaned.elements:
            if not element.table_rows:
                continue
            expected_header = list(element.table_rows[2])
            break
        else:
            raise ValueError(f"没有表格: {source}")
        data_chunks = [
            chunk for chunk in processed.chunks if not chunk.metadata.get("table_preamble")
        ]
        for chunk in data_chunks:
            table_lines = [line for line in chunk.content.splitlines() if line.startswith("| ")]
            if not table_lines or cells(table_lines[0]) != expected_header:
                raise ValueError(f"数据 Chunk 缺少原始列头: {source.name}#{chunk.chunk_index}")
            if chunk.metadata.get("table_header_row") != 3 or chunk.token_count > 512:
                raise ValueError(f"表格续块元数据或预算错误: {source.name}#{chunk.chunk_index}")
            checks.append(
                {
                    "source": str(source.relative_to(ROOT)),
                    "source_sha256": digest(source),
                    "chunk_index": chunk.chunk_index,
                    "content_sha256": chunk.content_sha256,
                    "token_count": chunk.token_count,
                    "header": expected_header,
                    "header_in_current_chunk": True,
                }
            )
            schema = RELATION_SCHEMAS.get(source.name)
            if schema is None:
                continue
            source_column, source_type, target_column, target_type, relation_type = schema
            header = cells(table_lines[0])
            for line in table_lines[2:]:
                values = cells(line)
                if len(values) != len(header):
                    raise ValueError(f"表格行列数不一致: {source.name}#{chunk.chunk_index}")
                row = dict(zip(header, values, strict=True))
                source_name, target_name = row[source_column], row[target_column]
                if not source_name or not target_name:
                    continue
                extraction = Extraction.model_validate(
                    {
                        "entities": [
                            {
                                "id": "e1",
                                "type": source_type,
                                "name": source_name,
                                "canonical_name": source_name,
                                "evidence_quote": line,
                            },
                            {
                                "id": "e2",
                                "type": target_type,
                                "name": target_name,
                                "canonical_name": target_name,
                                "evidence_quote": line,
                            },
                        ],
                        "relations": [
                            {
                                "type": relation_type,
                                "source_id": "e1",
                                "target_id": "e2",
                                "evidence_quote": line,
                                "condition_text": None,
                            }
                        ],
                    }
                )
                validate_evidence(extraction, chunk.content)
                facts.append(
                    {
                        "source": str(source.relative_to(ROOT)),
                        "chunk_index": chunk.chunk_index,
                        "relation_type": relation_type,
                        "source_entity": source_name,
                        "target_entity": target_name,
                        "evidence_quote": line,
                        "header_in_current_chunk": True,
                        "same_row_evidence": source_name in line and target_name in line,
                    }
                )
    if not checks or not facts or not all(row["same_row_evidence"] for row in facts):
        raise ValueError("留出样本缺少有效的当前 Chunk 行内证据")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    for name, rows in (("chunk_checks.jsonl", checks), ("fact_candidates.jsonl", facts)):
        (OUTPUT / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    summary = {
        "unseen_document_count": len(unseen),
        "unseen_data_chunk_count": len(checks),
        "same_chunk_header_count": sum(row["header_in_current_chunk"] for row in checks),
        "row_relation_candidates": len(facts),
        "same_row_evidence_count": sum(row["same_row_evidence"] for row in facts),
        "mode": "local_unseen_xlsx_no_model_or_graph_writes",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
