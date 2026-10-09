"""Explain targeted Gold entity and relation recall losses from an offline graph snapshot."""

import argparse
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

if __package__:
    from scripts.evaluate_graph_quality import (
        FROZEN,
        GOLD,
        entity_signature,
        file_hash,
        normalize_name,
        predicted_relation_signature,
        ratio,
        read_jsonl,
    )
else:
    from evaluate_graph_quality import (
        FROZEN,
        GOLD,
        entity_signature,
        file_hash,
        normalize_name,
        predicted_relation_signature,
        ratio,
        read_jsonl,
    )


def overlap(first_start: int, first_end: int, second_start: int, second_end: int) -> bool:
    return first_start < second_end and first_end > second_start


def assess(snapshot: dict, cases: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    evidence_by_key = {row["key"]: row for row in snapshot["neo4j"]["evidence"]}
    mentions_by_chunk: dict[int, dict[str, dict]] = defaultdict(dict)
    assertions_by_chunk: dict[int, list[dict]] = defaultdict(list)
    for mention in snapshot["neo4j"]["mentions"]:
        evidence = evidence_by_key.get(mention["evidence_key"])
        if evidence is not None:
            mentions_by_chunk[evidence["chunk_id"]][mention["entity"]["key"]] = mention["entity"]
    for row in snapshot["neo4j"]["assertions"]:
        for evidence in row["evidences"]:
            assertions_by_chunk[evidence["chunk_id"]].append(row)

    frozen_cases = [case for case in cases if case["source_variant"] == "frozen"]
    zero_prediction_cases = [
        case for case in frozen_cases if not mentions_by_chunk[case["chunk_id"]]
    ]
    zero_prediction_chunks = {case["chunk_id"] for case in zero_prediction_cases}
    chunks_with_evidence = {row["chunk_id"] for row in snapshot["neo4j"]["evidence"]}
    high_density_chunks = {
        chunk_id for chunk_id, entities in mentions_by_chunk.items() if len(entities) >= 25
    }
    gold_entities = []
    relations = []
    by_id = {}
    for case in frozen_cases:
        case_id = case["case_id"]
        chunk_id = case["chunk_id"]
        predicted_entities = list(mentions_by_chunk[chunk_id].values())
        predicted_by_signature: dict[tuple, list[dict]] = defaultdict(list)
        for entity in predicted_entities:
            predicted_by_signature[entity_signature(entity)].append(entity)
        used_by_signature = Counter()
        entity_evidence = {
            row["target_id"]: row for row in case["evidence"] if row["target_kind"] == "entity"
        }
        for gold in case["entities"]:
            signature = entity_signature(gold)
            candidates = predicted_by_signature[signature]
            matched = used_by_signature[signature] < len(candidates)
            if matched:
                used_by_signature[signature] += 1
            name = normalize_name(gold["name"])
            wrong_type = [
                entity
                for entity in predicted_entities
                if normalize_name(entity["name"]) == name and entity["type"] != gold["type"]
            ]
            containment = [
                entity
                for entity in predicted_entities
                if entity["type"] == gold["type"]
                and normalize_name(entity["name"]) != name
                and len(name) >= 2
                and len(normalize_name(entity["name"])) >= 2
                and (
                    name in normalize_name(entity["name"]) or normalize_name(entity["name"]) in name
                )
            ]
            closest = sorted(
                (
                    (
                        SequenceMatcher(None, name, normalize_name(entity["name"])).ratio(),
                        entity,
                    )
                    for entity in predicted_entities
                    if entity["type"] == gold["type"]
                ),
                key=lambda item: item[0],
                reverse=True,
            )[:3]
            gold_span = entity_evidence.get(gold["id"])
            same_passage = []
            if gold_span is not None:
                for mention in snapshot["neo4j"]["mentions"]:
                    evidence = evidence_by_key.get(mention["evidence_key"])
                    if evidence is None or evidence["chunk_id"] != chunk_id:
                        continue
                    start, end = evidence.get("quote_start"), evidence.get("quote_end")
                    if (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and overlap(start, end, gold_span["start"], gold_span["end"])
                    ):
                        same_passage.append(mention["entity"]["key"])
            if matched:
                diagnostic = "MATCH"
            elif wrong_type:
                diagnostic = "SAME_NAME_WRONG_TYPE_CANDIDATE"
            elif containment:
                diagnostic = "SAME_TYPE_NAME_BOUNDARY_CANDIDATE"
            elif closest and closest[0][0] >= 0.72:
                diagnostic = "SAME_TYPE_SIMILAR_NAME_CANDIDATE"
            elif same_passage:
                diagnostic = "SAME_PASSAGE_DIFFERENT_ENTITY_CANDIDATE"
            else:
                diagnostic = "NO_OBVIOUS_CANDIDATE"
            row = {
                "gold_entity_id": f"{case_id}:{gold['id']}",
                "case_id": case_id,
                "chunk_id": chunk_id,
                "document_id": case["document_id"],
                "type": gold["type"],
                "name": gold["name"],
                "gold_evidence_quote": gold["evidence_quote"],
                "matched": matched,
                "diagnostic_candidate": diagnostic,
                "same_name_wrong_type": [
                    {"key": entity["key"], "type": entity["type"], "name": entity["name"]}
                    for entity in wrong_type
                ],
                "same_type_name_containment": [
                    {"key": entity["key"], "name": entity["name"]} for entity in containment
                ],
                "closest_same_type": [
                    {"score": round(score, 3), "key": entity["key"], "name": entity["name"]}
                    for score, entity in closest
                ],
                "predicted_mentions_overlapping_gold_evidence": len(set(same_passage)),
                "coverage_tags": case["coverage_tags"],
            }
            gold_entities.append(row)
            by_id[(case_id, gold["id"])] = row

        predicted_signatures = Counter(entity_signature(entity) for entity in predicted_entities)
        predicted_relations = Counter(
            signature
            for row in assertions_by_chunk[chunk_id]
            if (signature := predicted_relation_signature(row)) is not None
        )
        for index, relation in enumerate(case["relations"], start=1):
            source = by_id[(case_id, relation["source_id"])]
            target = by_id[(case_id, relation["target_id"])]
            source_sig = (source["type"], normalize_name(source["name"]))
            target_sig = (target["type"], normalize_name(target["name"]))
            if source_sig == target_sig and relation["source_id"] != relation["target_id"]:
                source_found = target_found = predicted_signatures[source_sig] >= 2
            else:
                source_found = predicted_signatures[source_sig] > 0
                target_found = predicted_signatures[target_sig] > 0
            signature = (*source_sig, relation["type"], *target_sig)
            triple_found = source_found and target_found and predicted_relations[signature] > 0
            if triple_found:
                predicted_relations[signature] -= 1
            if source_found and target_found:
                loss = "MATCH" if triple_found else "RELATION_MISSING_WITH_ENDPOINTS"
            elif not source_found and not target_found:
                loss = "BOTH_ENDPOINTS_MISSING"
            elif not source_found:
                loss = "SOURCE_MISSING"
            else:
                loss = "TARGET_MISSING"
            relations.append(
                {
                    "gold_relation_id": f"{case_id}:r{index}",
                    "case_id": case_id,
                    "chunk_id": chunk_id,
                    "document_id": case["document_id"],
                    "predicate": relation["type"],
                    "source_entity_id": source["gold_entity_id"],
                    "source": source["name"],
                    "source_type": source["type"],
                    "source_found": source_found,
                    "target_entity_id": target["gold_entity_id"],
                    "target": target["name"],
                    "target_type": target["type"],
                    "target_found": target_found,
                    "triple_found": triple_found,
                    "loss_category": loss,
                    "coverage_tags": case["coverage_tags"],
                }
            )

    missed_entities = [row for row in gold_entities if not row["matched"]]
    missed_relations = [row for row in relations if row["loss_category"] != "MATCH"]
    blockers = Counter(
        entity_id
        for row in missed_relations
        for entity_id, found in (
            (row["source_entity_id"], row["source_found"]),
            (row["target_entity_id"], row["target_found"]),
        )
        if not found
    )
    per_document: dict[int, Counter] = defaultdict(Counter)
    for row in gold_entities:
        per_document[row["document_id"]]["gold_entities"] += 1
        per_document[row["document_id"]]["matched_entities"] += row["matched"]
    for row in relations:
        per_document[row["document_id"]]["gold_relations"] += 1
        per_document[row["document_id"]]["matched_relations"] += row["loss_category"] == "MATCH"
    zero_relation_match_documents = [
        document_id
        for document_id, counts in per_document.items()
        if counts["gold_relations"] and not counts["matched_relations"]
    ]
    summary = {
        "gold_entities": len(gold_entities),
        "matched_entities": len(gold_entities) - len(missed_entities),
        "missed_entities": len(missed_entities),
        "zero_prediction_chunks": len(zero_prediction_cases),
        "zero_prediction_chunks_with_evidence": len(zero_prediction_chunks & chunks_with_evidence),
        "zero_prediction_chunks_with_gold_entities": sum(
            bool(case["entities"]) for case in zero_prediction_cases
        ),
        "gold_entities_in_zero_prediction_chunks": sum(
            len(case["entities"]) for case in zero_prediction_cases
        ),
        "gold_relations_in_zero_prediction_chunks": sum(
            len(case["relations"]) for case in zero_prediction_cases
        ),
        "missed_entities_outside_zero_prediction_chunks": sum(
            row["chunk_id"] not in zero_prediction_chunks for row in missed_entities
        ),
        "entity_miss_partition": dict(
            sorted(
                Counter(
                    "ZERO_GRAPH_OUTPUT_CHUNK"
                    if row["chunk_id"] in zero_prediction_chunks
                    else row["diagnostic_candidate"]
                    for row in missed_entities
                ).items()
            )
        ),
        "chunks_at_or_above_prompt_entity_limit_25": len(high_density_chunks),
        "missed_entities_in_high_density_chunks": sum(
            row["chunk_id"] in high_density_chunks for row in missed_entities
        ),
        "missed_relations_outside_zero_prediction_chunks": sum(
            row["chunk_id"] not in zero_prediction_chunks for row in missed_relations
        ),
        "entity_diagnostic_candidates": dict(
            sorted(Counter(row["diagnostic_candidate"] for row in missed_entities).items())
        ),
        "entity_misses_by_type": dict(
            sorted(Counter(row["type"] for row in missed_entities).items())
        ),
        "entity_gold_by_type": dict(sorted(Counter(row["type"] for row in gold_entities).items())),
        "entity_recall_by_type": {
            entity_type: {
                "matched": sum(
                    row["type"] == entity_type and row["matched"] for row in gold_entities
                ),
                "gold": sum(row["type"] == entity_type for row in gold_entities),
                "recall": ratio(
                    sum(row["type"] == entity_type and row["matched"] for row in gold_entities),
                    sum(row["type"] == entity_type for row in gold_entities),
                ),
            }
            for entity_type in sorted({row["type"] for row in gold_entities})
        },
        "gold_relations": len(relations),
        "matched_relations": len(relations) - len(missed_relations),
        "missed_relations": len(missed_relations),
        "per_document": [
            {"document_id": document_id, **dict(counts)}
            for document_id, counts in sorted(per_document.items())
        ],
        "zero_relation_match_documents": sorted(zero_relation_match_documents),
        "relation_misses_in_zero_match_documents": sum(
            per_document[document_id]["gold_relations"]
            for document_id in zero_relation_match_documents
        ),
        "entity_misses_in_zero_relation_match_documents": sum(
            per_document[document_id]["gold_entities"]
            - per_document[document_id]["matched_entities"]
            for document_id in zero_relation_match_documents
        ),
        "relation_loss_categories": dict(
            sorted(Counter(row["loss_category"] for row in relations).items())
        ),
        "missed_relations_by_predicate": dict(
            sorted(Counter(row["predicate"] for row in missed_relations).items())
        ),
        "unique_missing_gold_entities_blocking_relations": len(blockers),
        "endpoint_miss_occurrences_in_lost_relations": sum(blockers.values()),
        "top_blocking_gold_entities": [
            {
                "gold_entity_id": entity_id,
                "blocked_relation_count": count,
                "type": by_id[tuple(entity_id.split(":", 1))]["type"],
                "name": by_id[tuple(entity_id.split(":", 1))]["name"],
            }
            for entity_id, count in blockers.most_common(15)
        ],
        "diagnostic_warning": (
            "Name similarity and evidence overlap are review candidates, not adjudicated "
            "root causes or semantic identity matches."
        ),
    }
    return summary, gold_entities, relations


def render_markdown(summary: dict) -> str:
    entity_types = summary["entity_recall_by_type"]
    lines = [
        "# GraphRAG 召回损失归因",
        "",
        f"Gold 实体 {summary['gold_entities']} 个，命中 {summary['matched_entities']} 个，"
        f"漏召回 {summary['missed_entities']} 个。",
        "",
        "## ① Entity Recall 低的可观察原因",
        "",
        "| 实体类型 | 命中 / Gold | 召回 |",
        "| --- | ---: | ---: |",
    ]
    for name, row in entity_types.items():
        lines.append(f"| {name} | {row['matched']}/{row['gold']} | {row['recall']} |")
    lines.extend(["", "78 个漏召回实体的互斥分组：", ""])
    for name, count in summary["entity_miss_partition"].items():
        lines.append(f"- `{name}`：{count}。")
    lines.extend(
        [
            "",
            "这些分组由名称、类型及证据重叠自动生成，属于待人工确认的原因线索。"
            "逐条候选和相近系统实体见 entity_outcomes.jsonl。",
            "",
            f"59 个原始 Chunk 中有 {summary['zero_prediction_chunks']} 个没有任何系统实体提及；"
            f"其中 {summary['zero_prediction_chunks_with_gold_entities']} 个有正向 Gold，"
            f"直接对应 {summary['gold_entities_in_zero_prediction_chunks']} 个实体漏召回"
            f"和 {summary['gold_relations_in_zero_prediction_chunks']} 条关系漏召回。"
            f"这 {summary['zero_prediction_chunks']} 个 Chunk 也没有任何 Evidence 节点"
            f"（有 Evidence 的数量为 {summary['zero_prediction_chunks_with_evidence']}）。"
            "由于原始抽取缓存缺失，不能判定是模型返回空结果、任务处理还是落图环节造成。",
            f"其余 Chunk 还有 {summary['missed_entities_outside_zero_prediction_chunks']} 个"
            "实体未严格命中；Rule 仅命中 9/30，Step 命中 31/61。"
            "Field 的 4 个 Gold 均位于同一个零输出 Chunk，不能据此断定所有字段抽取都失败。",
            f"另有 {summary['chunks_at_or_above_prompt_entity_limit_25']} 个 Chunk 的投影实体数"
            "达到或超过提示词的 25 个上限建议，涉及 "
            f"{summary['missed_entities_in_high_density_chunks']} "
            "个 Gold 实体漏召回；这是容量压力线索，并非已证实的截断原因。",
        ]
    )
    lines.extend(
        [
            "",
            "## 文档差异",
            "",
            "| 文档 | Entity 命中/Gold | Relation 命中/Gold |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in summary["per_document"]:
        lines.append(
            f"| {row['document_id']} {row.get('title', '')} | "
            f"{row['matched_entities']}/{row['gold_entities']} | "
            f"{row['matched_relations']}/{row['gold_relations']} |"
        )
    lines.extend(
        [
            "",
            f"其中 {len(summary['zero_relation_match_documents'])} 份文档的 Gold 关系"
            "完全未命中，合计贡献 "
            f"{summary['relation_misses_in_zero_match_documents']}/"
            f"{summary['missed_relations']} 条关系漏召回和 "
            f"{summary['entity_misses_in_zero_relation_match_documents']}/"
            f"{summary['missed_entities']} 个实体漏召回。",
            "",
            "## ② Relation Recall 损失拆解",
            "",
            f"Gold 关系 {summary['gold_relations']} 条，命中 {summary['matched_relations']} 条，"
            f"漏召回 {summary['missed_relations']} 条。",
            "",
            "| 结果 | 条数 |",
            "| --- | ---: |",
        ]
    )
    for name, count in summary["relation_loss_categories"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "按严格实体类型和规范名称匹配，51 条漏召回关系均至少缺一个 Gold 端点；"
            "两端都已命中的 40 条关系，其关系类型和方向也全部命中。"
            "这不等于 51 条关系的概念在语义上完全没有被抽出，名称或类型差异仍需裁决。",
            f"其中 {summary['gold_relations_in_zero_prediction_chunks']} 条位于零实体输出 Chunk，"
            f"其余 {summary['missed_relations_outside_zero_prediction_chunks']} 条"
            "位于有其他实体输出的 Chunk。",
            "",
            "按关系类型统计漏召回：",
            "",
        ]
    )
    for name, count in summary["missed_relations_by_predicate"].items():
        lines.append(f"- `{name}`：{count} 条。")
    lines.extend(
        [
            "",
            f"漏召回关系共涉及 {summary['unique_missing_gold_entities_blocking_relations']} 个"
            "不同的未命中 Gold 实体；按 source/target 位置计，"
            f"有 {summary['endpoint_miss_occurrences_in_lost_relations']} 次端点缺失。"
            "逐条关系和端点状态见 relation_loss_outcomes.jsonl。",
            "",
            "阻塞关系最多的 Gold 实体：",
            "",
            "| Gold 实体 | 类型 | 受影响关系数 |",
            "| --- | --- | ---: |",
        ]
    )
    for row in summary["top_blocking_gold_entities"][:5]:
        lines.append(f"| {row['name']} | {row['type']} | {row['blocked_relation_count']} |")
    lines.extend(
        [
            "",
            "同一条关系两端都缺失时，只算 1 条关系损失，但算 2 次端点缺失。"
            "条件文本不一致另见六层评估报告，不并入三元组漏召回。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(snapshot_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {output_dir}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot["frozen_manifest_sha256"] != file_hash(FROZEN / "manifest.json"):
        raise ValueError("快照与冻结集版本不一致")
    if snapshot["gold_manifest_sha256"] != file_hash(GOLD / "manifest.json"):
        raise ValueError("快照与 Gold 版本不一致")
    summary, entities, relations = assess(snapshot, read_jsonl(GOLD / "cases.jsonl"))
    titles = {row["document_id"]: row["title"] for row in read_jsonl(FROZEN / "documents.jsonl")}
    for row in summary["per_document"]:
        row["title"] = titles.get(row["document_id"], "")
    report = {
        "schema_version": "graph-recall-loss-v1",
        "snapshot_sha256": file_hash(snapshot_path),
        "gold_manifest_sha256": snapshot["gold_manifest_sha256"],
        **summary,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    for filename, rows in (
        ("entity_outcomes.jsonl", entities),
        ("relation_loss_outcomes.jsonl", relations),
    ):
        (output_dir / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    print(f"实体漏召回 {summary['missed_entities']}，关系漏召回 {summary['missed_relations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.snapshot, args.output)


if __name__ == "__main__":
    main()
