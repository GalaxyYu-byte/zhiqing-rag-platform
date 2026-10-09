"""Build an offline review queue from the first responses of four zero-output cases."""

import json
from pathlib import Path

from zhiqing_rag.workers.graph_extraction import normalize_name, validated_endpoint_subset

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/graph_build/runs/2026-09-29-live/zero-chunk-investigation/replay-20260930"
GOLD = ROOT / "eval/graph_build/gold-v1/cases.jsonl"
TARGETS = {"gb-35-00", "gb-35-01", "gb-45-00", "gb-73-03"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    gold = {case["case_id"]: case for case in read_jsonl(GOLD)}
    rows = [row for row in read_jsonl(BASE / "replay.jsonl") if row["case_id"] in TARGETS]
    queue = []
    markdown = [
        "# 首次模型响应的候选事实审核队列",
        "",
        "仅纳入通过现有结构与逐字证据校验的首次响应记录。Gold 精确匹配只供定位，未匹配不等于错误。",
        "这里的支持性裁决尚未完成；任何判断都不得写作人工已验证 Gold。",
        "",
    ]
    for row in rows:
        case = gold[row["case_id"]]
        extraction = validated_endpoint_subset(row["attempts"][0]["raw_output"], case["content"])
        by_id = {entity.id: entity for entity in extraction.entities}
        gold_by_id = {entity["id"]: entity for entity in case["entities"]}
        gold_entities = {
            (entity["type"], normalize_name(entity["name"])) for entity in case["entities"]
        }
        gold_relations = {
            (
                gold_by_id[relation["source_id"]]["type"],
                normalize_name(gold_by_id[relation["source_id"]]["name"]),
                relation["type"],
                gold_by_id[relation["target_id"]]["type"],
                normalize_name(gold_by_id[relation["target_id"]]["name"]),
            )
            for relation in case["relations"]
        }
        markdown += [
            f"## {row['case_id']}",
            "",
            "### 完整 Chunk",
            "",
            "```text",
            case["content"],
            "```",
            "",
            "### Gold",
            "",
            "实体：" + "；".join(f"{e['type']} {e['name']}" for e in case["entities"]),
            "",
            "关系：" + "；".join(
                f"{gold_by_id[r['source_id']]['name']} --{r['type']}--> "
                f"{gold_by_id[r['target_id']]['name']}"
                for r in case["relations"]
            ),
            "",
            "### 候选实体",
            "",
        ]
        for index, entity in enumerate(extraction.entities, start=1):
            key = f"{row['case_id']}-E{index:02d}"
            exact = (entity.type, normalize_name(entity.canonical_name)) in gold_entities
            queue.append(
                {
                    "key": key,
                    "case_id": row["case_id"],
                    "kind": "entity",
                    "type": entity.type,
                    "name": entity.name,
                    "canonical_name": entity.canonical_name,
                    "evidence_quote": entity.evidence_quote,
                    "gold_exact_match": exact,
                    "review_decision": None,
                    "review_note": None,
                }
            )
            markdown.append(
                f"- {key} [{entity.type}] {entity.canonical_name} "
                f"(Gold 精确命中: {'是' if exact else '否'})｜证据：{entity.evidence_quote}"
            )
        markdown += ["", "### 候选关系", ""]
        for index, relation in enumerate(extraction.relations, start=1):
            key = f"{row['case_id']}-R{index:02d}"
            source = by_id[relation.source_id]
            target = by_id[relation.target_id]
            exact = (
                source.type,
                normalize_name(source.canonical_name),
                relation.type,
                target.type,
                normalize_name(target.canonical_name),
            ) in gold_relations
            queue.append(
                {
                    "key": key,
                    "case_id": row["case_id"],
                    "kind": "relation",
                    "type": relation.type,
                    "source": source.canonical_name,
                    "source_type": source.type,
                    "target": target.canonical_name,
                    "target_type": target.type,
                    "evidence_quote": relation.evidence_quote,
                    "condition_text": relation.condition_text,
                    "gold_exact_match": exact,
                    "review_decision": None,
                    "review_note": None,
                }
            )
            markdown.append(
                f"- {key} {source.canonical_name} --{relation.type}--> {target.canonical_name} "
                f"(Gold 精确命中: {'是' if exact else '否'})｜证据：{relation.evidence_quote}"
                + (f"｜条件：{relation.condition_text}" if relation.condition_text else "")
            )
        markdown.append("")
    (BASE / "review_queue.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in queue),
        encoding="utf-8",
    )
    (BASE / "review_queue.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"wrote {len(queue)} candidates: {sum(x['kind'] == 'entity' for x in queue)} entities, "
          f"{sum(x['kind'] == 'relation' for x in queue)} relations")


if __name__ == "__main__":
    main()
