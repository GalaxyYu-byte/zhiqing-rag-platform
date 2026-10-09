"""Offline pilot for column-aware graph extraction from frozen Markdown tables."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from zhiqing_rag.workers.graph_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    normalize_name,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "eval/graph_build/v1/chunks.jsonl"
GOLD = ROOT / "eval/graph_build/gold-v1/cases.jsonl"
BASELINE_ENTITIES = (
    ROOT / "eval/graph_build/runs/2026-09-29-live/recall-loss/entity_outcomes.jsonl"
)
BASELINE_RELATIONS = (
    ROOT / "eval/graph_build/runs/2026-09-29-live/recall-loss/relation_loss_outcomes.jsonl"
)
OUTPUT = ROOT / "eval/graph_build/runs/2026-09-29-live/recall-loss/table-experiment-v1"

# Pilot vocabulary for unspaced role/action cells. It is not a general Chinese role parser.
ROLE_BASES = (
    "用人部门",
    "接收人",
    "管理员",
    "经办人",
    "主管",
    "行政",
    "人事",
    "员工",
)
ROLE = "(?:" + "|".join(re.escape(role) for role in ROLE_BASES) + ")"
ROLE_ACTION = re.compile(rf"^(?P<actor>{ROLE}(?:与{ROLE})*)(?P<action>.+)$")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def split_role_action(value: str) -> tuple[str, str, str | None] | None:
    condition = None
    action_cell = value
    if "后由" in action_cell:
        prefix, action_cell = action_cell.rsplit("后由", 1)
        condition = prefix + "后"
    elif action_cell.startswith("由"):
        action_cell = action_cell[1:]
    match = ROLE_ACTION.fullmatch(action_cell)
    if not match:
        return None
    return match["actor"], match["action"], condition


def extract_tables(
    chunk: dict, active_header: list[str] | None
) -> tuple[Extraction, list[str] | None, bool]:
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []
    ids: dict[tuple[str, str], str] = {}

    def entity(kind: str, name: str, quote: str) -> str:
        key = (kind, normalize_name(name))
        if key not in ids:
            local_id = f"e{len(entities) + 1}"
            ids[key] = local_id
            entities.append(
                ExtractedEntity(
                    id=local_id,
                    type=kind,
                    name=name,
                    canonical_name=name,
                    evidence_quote=quote,
                )
            )
        return ids[key]

    header = active_header
    header_in_chunk = False
    for line in chunk["content"].splitlines():
        cells = table_cells(line)
        if not cells or all(cell and set(cell) <= {"-", ":"} for cell in cells):
            continue
        if "办理岗位应做" in cells and "事项" in cells:
            header = cells
            header_in_chunk = True
            continue
        if "资料类型" in cells and "保管关注点" in cells:
            header = cells
            header_in_chunk = True
            continue
        if header is None or len(cells) != len(header):
            continue
        row = dict(zip(header, cells, strict=True))
        if "办理岗位应做" in header:
            parsed = split_role_action(row["办理岗位应做"])
            if parsed is None:
                continue
            actor, action, condition = parsed
            source_id = entity("Actor", actor, row["办理岗位应做"])
            target_id = entity("Step", action, row["办理岗位应做"])
            relations.append(
                ExtractedRelation(
                    type="RESPONSIBLE_FOR",
                    source_id=source_id,
                    target_id=target_id,
                    evidence_quote=line,
                    condition_text=condition,
                )
            )
        elif "资料类型" in header and "保管关注点" in header:
            material, rule = row["资料类型"], row["保管关注点"]
            if not material or not rule:
                continue
            source_id = entity("Material", material, line)
            target_id = entity("Rule", rule, line)
            relations.append(
                ExtractedRelation(
                    type="GOVERNED_BY",
                    source_id=source_id,
                    target_id=target_id,
                    evidence_quote=line,
                )
            )
    extraction = Extraction(entities=entities, relations=relations)
    validate_evidence(extraction, chunk["content"])
    return extraction, header, header_in_chunk


def score(proposals: list[dict]) -> tuple[list[dict], dict]:
    gold = {
        case["chunk_id"]: case
        for case in read_jsonl(GOLD)
        if case["source_variant"] == "frozen"
    }
    baseline_entities = {
        row["gold_entity_id"]: row["matched"] for row in read_jsonl(BASELINE_ENTITIES)
    }
    baseline_relations = {
        row["gold_relation_id"]: row["triple_found"] for row in read_jsonl(BASELINE_RELATIONS)
    }
    outcomes = []
    counts = Counter()
    for proposal in proposals:
        case = gold[proposal["chunk_id"]]
        extraction = Extraction.model_validate(proposal["extraction"])
        by_id = {item.id: item for item in extraction.entities}
        predicted_entities = Counter(
            (item.type, normalize_name(item.canonical_name)) for item in extraction.entities
        )
        predicted_relations = Counter(
            (
                by_id[item.source_id].type,
                normalize_name(by_id[item.source_id].canonical_name),
                item.type,
                by_id[item.target_id].type,
                normalize_name(by_id[item.target_id].canonical_name),
            )
            for item in extraction.relations
        )
        gold_by_id = {item["id"]: item for item in case["entities"]}
        for item in case["entities"]:
            key = f"{case['case_id']}:{item['id']}"
            signature = (item["type"], normalize_name(item["name"]))
            proposed = predicted_entities[signature] > 0
            if proposed:
                predicted_entities[signature] -= 1
            baseline = baseline_entities[key]
            outcomes.append(
                {
                    "kind": "entity",
                    "gold_id": key,
                    "baseline_match": baseline,
                    "pilot_match": proposed,
                    "new_strict_hit": proposed and not baseline,
                }
            )
            counts["gold_entities"] += 1
            counts["baseline_entities"] += baseline
            counts["pilot_entities"] += proposed
            counts["new_entities"] += proposed and not baseline
        for index, item in enumerate(case["relations"], start=1):
            source = gold_by_id[item["source_id"]]
            target = gold_by_id[item["target_id"]]
            signature = (
                source["type"],
                normalize_name(source["name"]),
                item["type"],
                target["type"],
                normalize_name(target["name"]),
            )
            proposed = predicted_relations[signature] > 0
            if proposed:
                predicted_relations[signature] -= 1
            key = f"{case['case_id']}:r{index}"
            baseline = baseline_relations[key]
            outcomes.append(
                {
                    "kind": "relation",
                    "gold_id": key,
                    "baseline_match": baseline,
                    "pilot_match": proposed,
                    "new_strict_hit": proposed and not baseline,
                }
            )
            counts["gold_relations"] += 1
            counts["baseline_relations"] += baseline
            counts["pilot_relations"] += proposed
            counts["new_relations"] += proposed and not baseline
    return outcomes, dict(counts)


def main() -> None:
    chunks = sorted(read_jsonl(FROZEN), key=lambda row: (row["document_id"], row["chunk_index"]))
    headers: dict[int, list[str]] = {}
    proposals = []
    for chunk in chunks:
        extraction, header, header_in_chunk = extract_tables(
            chunk, headers.get(chunk["document_id"])
        )
        if header:
            headers[chunk["document_id"]] = header
        if extraction.relations:
            proposals.append(
                {
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "content_sha256": chunk["content_sha256"],
                    "header_in_current_chunk": header_in_chunk,
                    "extraction": extraction.model_dump(mode="json"),
                }
            )
    outcomes, counts = score(proposals)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("proposals.jsonl", proposals), ("gold_outcomes.jsonl", outcomes)):
        (OUTPUT / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    metadata = {
        "mode": "offline_table_column_pilot_no_graph_writes",
        "frozen_chunks_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "gold_cases_sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(),
        "chunks_scanned": len(chunks),
        "chunks_with_proposals": len(proposals),
        "chunks_using_carried_header": sum(
            not row["header_in_current_chunk"] for row in proposals
        ),
        "proposed_entities": sum(len(row["extraction"]["entities"]) for row in proposals),
        "proposed_relations": sum(len(row["extraction"]["relations"]) for row in proposals),
        "counts": counts,
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
