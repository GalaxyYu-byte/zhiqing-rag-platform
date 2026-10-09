"""Record Codex's explicit preliminary decisions for the fixed holdout replay."""

from collections import Counter

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.replay_graph_table_holdout import FROZEN
from zhiqing_rag.workers.graph_extraction import normalize_name

RUN = PILOT / "holdout-replay-20260930"
EXPECTED_REPLAY_SHA256 = "560cfba67cf6de00ecf6295ed843545709e7e2b56e3fc65d44a25afef1d820fa"


def decision(case_id, index):
    """Explicit decisions after reading every prediction and the original table."""
    if case_id == "holdout-01-02" and index == 33:
        return (
            "INSUFFICIENT",
            "未确认权限的申请人动作是补充说明；目标说明使用目的来自共享链接检查行。",
        )
    if case_id == "holdout-02-01":
        if index <= 7:
            return "INSUFFICIENT", "状态表按行列出处理要求，未规定这两个状态之间的直接转移。"
        return (
            "INSUFFICIENT",
            "需要留下的内容不证明由该动作产生；原始描述等可能来自外部，不能将保留等同生成。",
        )
    if case_id == "holdout-02-02":
        if index <= 6:
            return (
                "INSUFFICIENT",
                "状态行排列不证明直接转移或回访到重新打开的循环；没有明确转移条件。",
            )
        return "SUPPORT", "原文明示本文状态名称不是线上系统固定枚举，限制适用于该状态名称的解释。"
    if case_id == "holdout-03-02":
        if index % 3 == 2:
            return (
                "INSUFFICIENT",
                "检查要点是对象约束，被误建为 Step；不是该 Step 对材料执行操作的证据。",
            )
        if index % 3 == 0:
            return (
                "INSUFFICIENT",
                "发现差异后是条件分支，未规定检查要点之后必然执行该动作；条件和规则类型丢失。",
            )
        return "SUPPORT", "接收方用途列明确连接本行文件或成果及其使用目的；Step 边界仍需业务审核。"
    if case_id == "holdout-04-02":
        if index <= 5:
            return (
                "INSUFFICIENT",
                "沟通/执行/观察/归档是检查组分类，不是责任角色，不能建立 Actor 负责关系。",
            )
        if index == 7:
            return (
                "INSUFFICIENT",
                "没有一行代表实际发布已执行是清单范围声明，不证明发布记录对象受该规则约束。",
            )
    if case_id in ("holdout-06-01", "holdout-06-02") and index % 4 == 0:
        return (
            "INSUFFICIENT",
            "暂停或退回条件被写成裸 Rule，未表达触发后需暂停/退回；"
            "与条件相同的 condition_text 也未恢复该约束。",
        )
    if case_id.startswith("holdout-01"):
        return "SUPPORT", "本 Chunk 角色列与本行的职责动作明确对应；HAS_STEP 依据同一工作环节行。"
    if case_id == "holdout-03-01":
        return (
            "SUPPORT",
            "文件或成果和检查要点列给出本行对象与约束；Material 的业务建模边界待人工确定。",
        )
    if case_id.startswith("holdout-06"):
        return (
            "SUPPORT",
            "本行主责角色/环节/输入/形成的结果列明确对应；结果为流程定义，未断言实际已发生。",
        )
    if case_id == "holdout-04-02" and index == 6:
        return "SUPPORT", "归档行明确要求发布记录可追溯到最终版本。"
    raise ValueError(f"未裁决的预测: {case_id}:{index}")


def context_rows(content, source, target, quote):
    lines = content.splitlines()
    rows = [line for line in lines[2:] if quote in line or (source in line and target in line)]
    if not rows:
        rows = [line for line in lines[2:] if source in line or target in line]
    for name in (source, target):
        if name not in lines[0] and not any(name in row for row in rows):
            rows.extend(line for line in lines[2:] if name in line and line not in rows)
    return {"header": lines[0], "rows": rows}


def main():
    if sha(RUN / "replay.jsonl") != EXPECTED_REPLAY_SHA256:
        raise ValueError("预测内容与本次初审版本不符，需重新逐条裁决")
    cases = {c["case_id"]: c for c in read_jsonl(FROZEN / "cases.jsonl")}
    relations = []
    failures = []
    entity_flags = []
    counts = Counter()
    normalization_hints = []
    for row in read_jsonl(RUN / "replay.jsonl"):
        case_id = row["case_id"]
        case = cases[case_id]
        content = case["content"]
        if row["status"] != "VALID_EXTRACTION":
            for attempt in row["attempts"]:
                import json

                try:
                    raw = json.loads(attempt.get("raw_output") or "")
                except ValueError:
                    continue
                mismatches = []
                for kind in ("entities", "relations"):
                    for index, record in enumerate(raw.get(kind, []), 1):
                        for field in ("name", "canonical_name", "evidence_quote", "condition_text"):
                            if record.get(field) and record[field] not in content:
                                mismatches.append(
                                    {
                                        "kind": kind,
                                        "index": index,
                                        "field": field,
                                        "value": record[field],
                                    }
                                )
                failures.append(
                    {
                        "case_id": case_id,
                        "attempt": attempt["attempt"],
                        "final_error_code": row.get("error_code"),
                        "name_or_evidence_mismatches": mismatches,
                        "empty_response": raw == {"entities": [], "relations": []},
                    }
                )
            continue
        extraction = row["extraction"]
        entities = {e["id"]: e for e in extraction["entities"]}
        for index, relation in enumerate(extraction["relations"], 1):
            source = entities[relation["source_id"]]
            target = entities[relation["target_id"]]
            verdict, reason = decision(case_id, index)
            counts[verdict] += 1
            relations.append(
                {
                    "case_id": case_id,
                    "relation_index": index,
                    "source": source["name"],
                    "source_type": source["type"],
                    "type": relation["type"],
                    "target": target["name"],
                    "target_type": target["type"],
                    "condition_text": relation["condition_text"],
                    "evidence_quote": relation["evidence_quote"],
                    "context": context_rows(
                        content, source["name"], target["name"], relation["evidence_quote"]
                    ),
                    "evidence_verdict": verdict,
                    "reason": reason,
                    "reviewer": "Codex",
                    "independent_human_review": "PENDING",
                }
            )
        duplicates = Counter(
            (e["type"], normalize_name(e["canonical_name"])) for e in extraction["entities"]
        )
        for (entity_type, name), count in duplicates.items():
            if count > 1:
                normalization_hints.append(
                    {
                        "case_id": case_id,
                        "type": entity_type,
                        "normalized_name": name,
                        "records": count,
                    }
                )
        for entity in extraction["entities"]:
            reason = None
            if case_id == "holdout-04-02" and entity["type"] == "Actor":
                reason = "检查组分类被错误建成 Actor。"
            if case_id == "holdout-03-02" and entity["id"] in ("e4", "e9", "e14", "e19", "e24"):
                reason = "检查要点应与 Rule 边界比较，被建成 Step。"
            if reason:
                entity_flags.append(
                    {
                        "case_id": case_id,
                        "entity": entity,
                        "issue": "ENTITY_TYPE_MISMATCH",
                        "reason": reason,
                        "reviewer": "Codex",
                        "independent_human_review": "PENDING",
                    }
                )
    if len(relations) != 169 or counts != Counter(SUPPORT=117, INSUFFICIENT=52):
        raise ValueError("固定初审对应的预测发生变化，不能套用旧裁决")
    write_jsonl(RUN / "codex_evidence_review.jsonl", relations)
    write_jsonl(RUN / "failure_diagnostics.jsonl", failures)
    write_jsonl(RUN / "entity_type_review.jsonl", entity_flags)
    write_jsonl(RUN / "normalization_candidates.jsonl", normalization_hints)
    write_json(
        RUN / "codex_review_manifest.json",
        {
            "replay_sha256": sha(RUN / "replay.jsonl"),
            "frozen_cases_sha256": sha(FROZEN / "cases.jsonl"),
            "review_sha256": sha(RUN / "codex_evidence_review.jsonl"),
            "reviewer": "Codex",
            "independent_human_review": "PENDING",
            "relation_counts": dict(counts),
            "entity_type_flags": len(entity_flags),
            "within_chunk_exact_duplicate_groups": len(normalization_hints),
            "formal_evidence_precision": None,
            "formal_normalization_error_rates": None,
        },
    )
    print(f"Codex 逐条初审：{dict(counts)}；{len(entity_flags)} 个实体类型待裁决")


if __name__ == "__main__":
    main()
