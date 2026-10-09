"""Check whether frozen table Gold relations are explicit in their row columns."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "eval/graph_build/gold-v1/cases.jsonl"
OUTPUT = ROOT / "eval/graph_build/runs/2026-09-29-live/recall-loss/table-alignment"
TARGET_COLUMNS = {
    83: ("办理岗位应做", "办理岗位应做"),
    89: ("资料类型", "保管关注点"),
}


def cells(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def main() -> None:
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line
    ]
    cases = sorted(
        (case for case in cases if case["document_id"] in TARGET_COLUMNS),
        key=lambda case: (case["document_id"], case["case_id"]),
    )
    headers: dict[int, list[str]] = {}
    rows = []
    for case in cases:
        document_id = case["document_id"]
        relation_columns = TARGET_COLUMNS[document_id]
        table_rows = []
        for line in case["content"].splitlines():
            row = cells(line)
            if not row:
                continue
            if all(cell and set(cell) <= {"-", ":"} for cell in row):
                continue
            if all(column in row for column in relation_columns):
                headers[document_id] = row
                continue
            header = headers.get(document_id)
            if header is None or len(row) != len(header):
                continue
            if sum(bool(cell) for cell in row) < 2:
                continue
            table_rows.append(dict(zip(header, row, strict=True)))
        source_column, target_column = relation_columns
        entities = {entity["id"]: entity for entity in case["entities"]}
        for relation in case["relations"]:
            source = entities[relation["source_id"]]["name"]
            target = entities[relation["target_id"]]["name"]
            matching_rows = [
                index
                for index, row in enumerate(table_rows, start=1)
                if source in row[source_column] and target in row[target_column]
            ]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "document_id": document_id,
                    "relation_type": relation["type"],
                    "source": source,
                    "target": target,
                    "source_column": source_column,
                    "target_column": target_column,
                    "matching_row_numbers": matching_rows,
                    "matching_row_cells": [table_rows[index - 1] for index in matching_rows],
                }
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "relations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts = Counter(row["document_id"] for row in rows)
    aligned = Counter(row["document_id"] for row in rows if row["matching_row_numbers"])
    report = [
        "# 冻结表格的行列证据对齐检查",
        "",
        "只读取 Gold 和冻结 Chunk，按 Markdown 表格行列定位关系端点，不调用模型或数据库。",
        "这证明原文中是否有可定位的结构证据，不代表预测已命中或语义裁决已完成。",
        "",
        "| 文档 | Gold 关系 | 同一行指定列可定位 |",
        "| --- | ---: | ---: |",
    ]
    for document_id in sorted(counts):
        report.append(f"| {document_id} | {counts[document_id]} | {aligned[document_id]} |")
    report += [
        "",
        "文档 83 使用‘办理岗位应做’列内的执行人与动作；文档 89 使用‘资料类型’和‘保管关注点’两列。",
        "对齐结果在 `relations.jsonl`，其中保留匹配的整行单元格。",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"aligned {sum(aligned.values())}/{len(rows)} Gold relations")


if __name__ == "__main__":
    main()
