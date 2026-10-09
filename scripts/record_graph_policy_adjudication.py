"""Record five user-approved annotation policies and apply them to a new evaluation draft."""

import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.replay_graph_table_holdout import FROZEN
from scripts.replay_graph_zero_chunks import compare_to_gold
from scripts.verify_graph_table_unseen import cells
from zhiqing_rag.workers.graph_extraction import Extraction, validate_evidence

RUN = PILOT / "name-feedback-regression-20260930"
OUTPUT = PILOT / "human-policy-adjudication-v1"
POLICIES = [
    {"id": 1, "rule": "资料实体用资料类别名称，文件或成果内容描述不直接作为别名合并。"},
    {"id": 2, "rule": "检查组是分类，不能仅据此标为 Actor。"},
    {"id": 3, "rule": "申请人应说明中的说明内容按 Field 处理，不强制建为 Step；相关关系重标。"},
    {"id": 4, "rule": "状态表行排列不足以证明直接 TRANSITIONS_TO，判 INSUFFICIENT。"},
    {"id": 5, "rule": "暂停/退回条件须保留触发后暂停或退回的含义；裸 Rule 候选待裁决。"},
]


def table_rows(case):
    lines = case["content"].splitlines()
    header = cells(lines[0])
    rows = []
    for line in lines[2:]:
        values = cells(line)
        if len(values) == len(header):
            rows.append({"quote": line, "values": dict(zip(header, values, strict=True))})
    return rows


def revise_material_subjects(case):
    """Apply the same subject policy to every annotated row, including former hits."""
    revised = deepcopy(case)
    if not case["case_id"].startswith("holdout-03-"):
        return revised, []
    by_id = {e["id"]: e for e in revised["entities"]}
    rows = {r["quote"]: r["values"] for r in table_rows(case)}
    changes = []
    for relation in revised["relations"]:
        source = by_id[relation["source_id"]]
        row = rows[relation["evidence_quote"]]
        if source["name"] != row["文件或成果"]:
            raise ValueError("原 Gold 主体不符合已展示的旧标注模式")
        replacement = row["资料类别"]
        if not replacement:
            raise ValueError("新资料实体名称为空")
        changes.append({"case_id": case["case_id"], "entity_id": source["id"],
                        "policy_id": 1, "old_name": source["name"], "new_name": replacement,
                        "evidence_quote": relation["evidence_quote"],
                        "application_reviewer": "Codex", "item_human_review": "PENDING"})
        source.update(name=replacement, canonical_name=replacement)
    revised["gold_scope"] = "selected_targets_under_user_approved_material_subject_policy"
    revised["review_status"] = "human_policy_approved_codex_application_item_review_pending"
    return revised, changes


def apply_prediction_policies(case, extraction):
    """Derived flags are distinct from human review of each prediction."""
    flags = []
    rows = table_rows(case)
    groups = {r["values"].get("检查组") for r in rows} - {None, ""}
    fields = {r["values"].get("申请人应说明") for r in rows} - {None, ""}
    pause_conditions = {r["values"].get("暂停或退回条件") for r in rows} - {None, ""}
    field_ids = set()
    category_ids = set()
    for entity in extraction.entities:
        if entity.name in groups and entity.type == "Actor":
            category_ids.add(entity.id)
            flags.append({"policy_id": 2, "kind": "entity", "entity_id": entity.id,
                          "name": entity.name, "observed_type": entity.type,
                          "expected_representation": "table_category_metadata",
                          "derived_verdict": "INVALID_TYPE"})
        if entity.name in fields and entity.type != "Field":
            field_ids.add(entity.id)
            flags.append({"policy_id": 3, "kind": "entity", "entity_id": entity.id,
                          "name": entity.name, "observed_type": entity.type,
                          "expected_type": "Field", "derived_verdict": "INVALID_TYPE"})
    by_id = {e.id: e for e in extraction.entities}
    for index, relation in enumerate(extraction.relations, 1):
        common = {"kind": "relation", "relation_index": index, "type": relation.type,
                  "source": by_id[relation.source_id].name,
                  "target": by_id[relation.target_id].name}
        if relation.source_id in category_ids or relation.target_id in category_ids:
            flags.append({**common, "policy_id": 2, "derived_verdict": "NEEDS_REANNOTATION",
                          "reason": "分类实体不能承担 Actor 关系端点。"})
        if relation.source_id in field_ids or relation.target_id in field_ids:
            flags.append({**common, "policy_id": 3, "derived_verdict": "NEEDS_REANNOTATION",
                          "reason": "改为 Field 后原 Step 端点关系需重新标注。"})
        if case["case_id"].startswith("holdout-02-") and relation.type == "TRANSITIONS_TO":
            flags.append({**common, "policy_id": 4, "derived_verdict": "INSUFFICIENT",
                          "reason": "原状态表未给出这两个状态间的直接转移。"})
        if (relation.type == "GOVERNED_BY" and common["target"] in pause_conditions):
            flags.append({**common, "policy_id": 5, "derived_verdict": "REVIEW_REQUIRED",
                          "required_effect": "暂停或退回",
                          "condition_text": relation.condition_text,
                          "reason": "需核查触发后的暂停/退回含义；该具体候选尚未人工裁决。"})
    for flag in flags:
        flag.update(case_id=case["case_id"], application_reviewer="Codex",
                    human_policy_status="APPROVED", item_human_review="PENDING")
    return flags


def build():
    if OUTPUT.exists():
        raise FileExistsError("裁决版本已存在；使用 verify，不覆盖")
    original = read_jsonl(FROZEN / "cases.jsonl")
    predictions = {r["case_id"]: r for r in read_jsonl(RUN / "replay.jsonl")}
    cases, changes, flags = [], [], []
    old_counts, new_counts = Counter(), Counter()
    document_counts = {}
    for case in original:
        revised, revisions = revise_material_subjects(case)
        cases.append(revised)
        changes.extend(revisions)
        extraction = Extraction.model_validate(predictions[case["case_id"]]["extraction"])
        validate_evidence(extraction, case["content"])
        flags.extend(apply_prediction_policies(case, extraction))
        if case["relations"]:
            before, after = compare_to_gold(extraction, case), compare_to_gold(extraction, revised)
            old_counts.update(before)
            new_counts.update(after)
            doc = document_counts.setdefault(case["source"], {"old": Counter(), "new": Counter()})
            doc["old"].update(before)
            doc["new"].update(after)
    if (len(changes) != 15 or new_counts["gold_entities"] != 57
            or new_counts["gold_relations"] != 30):
        raise ValueError("裁决作用范围或目标数量不符")
    summary = {
        "old_policy_counts": dict(old_counts), "approved_policy_counts": dict(new_counts),
        "documents": {key: {mode: dict(value) for mode, value in doc.items()}
                      for key, doc in document_counts.items()},
        "approved_policy_entity_recall": new_counts["exact_entity_matches"] / 57,
        "approved_policy_triple_recall": new_counts["exact_directed_triple_matches"] / 30,
        "derived_flag_counts": dict(Counter(f"policy_{f['policy_id']}:{f['kind']}" for f in flags)),
        "human_policy_status": "APPROVED", "full_item_human_review": "PENDING",
        "final_precision": None, "final_f1": None, "evidence_precision": None,
        "scope": "same saved predictions; changed annotation policy; selected Gold covers 2 docs",
    }
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / "human_decisions.json", {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "reviewer": "human_user_in_current_chat",
        "approval_basis": "direct_user_reply_to_five_displayed_annotation_policies",
        "verbatim_reply": "1 同意  2 同意 3 同意  4 同意  5 同意",
        "policies": [{**p, "status": "APPROVED"} for p in POLICIES],
        "scope": "five policies and displayed examples; does not certify every prediction",
    })
    write_jsonl(OUTPUT / "cases.jsonl", cases)
    write_jsonl(OUTPUT / "gold_changes.jsonl", changes)
    write_jsonl(OUTPUT / "policy_application_flags.jsonl", flags)
    write_json(OUTPUT / "summary.json", summary)
    write_json(OUTPUT / "manifest.json", {
        "schema_version": "graph-build-human-policy-adjudication-v1",
        "parent_cases_sha256": sha(FROZEN / "cases.jsonl"),
        "prediction_sha256": sha(RUN / "replay.jsonl"),
        "files": {name: sha(OUTPUT / name) for name in
                  ("human_decisions.json", "cases.jsonl", "gold_changes.jsonl",
                   "policy_application_flags.jsonl", "summary.json")},
        "human_policy_status": "APPROVED", "full_item_human_review": "PENDING",
    })
    verify()
    print(json.dumps(summary, ensure_ascii=False))


def verify():
    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if sha(FROZEN / "cases.jsonl") != manifest["parent_cases_sha256"]:
        raise ValueError("父级 Gold 草稿变化")
    if sha(RUN / "replay.jsonl") != manifest["prediction_sha256"]:
        raise ValueError("绑定的模型预测变化")
    for name, expected in manifest["files"].items():
        if sha(OUTPUT / name) != expected:
            raise ValueError(f"裁决记录哈希不符: {name}")
    originals = read_jsonl(FROZEN / "cases.jsonl")
    saved = read_jsonl(OUTPUT / "cases.jsonl")
    if saved != [revise_material_subjects(case)[0] for case in originals]:
        raise ValueError("Gold 修订无法重现")
    for case in saved:
        for evidence in case["evidence"]:
            if case["content"][evidence["start"]:evidence["end"]] != evidence["quote"]:
                raise ValueError("证据偏移失效")
        validate_evidence(Extraction.model_validate({
            "entities": case["entities"],
            "relations": [{k: v for k, v in relation.items() if k != "id"}
                          for relation in case["relations"]],
        }), case["content"])
    print("人工口径版本校验通过：5项批准，15行主体一致修订；逐条人工审核仍待完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.command == "build" else verify()
