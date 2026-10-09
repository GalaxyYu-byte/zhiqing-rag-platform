"""Prepare source-derived Gold targets and item review for the saved thirteen-chunk run.

This is a post-run Codex annotation draft, not an independent human test set.
No model calls, database writes, or production schema changes are made here.
"""

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.record_graph_policy_adjudication import OUTPUT as POLICY
from scripts.record_graph_policy_adjudication import RUN, table_rows
from zhiqing_rag.workers.graph_extraction import Extraction, normalize_name, validate_evidence

OUTPUT = PILOT / "six-document-review-draft-v1"
EXPECTED_REPLAY_SHA256 = "0fd226554fc70360bcad98fcd66b024127cd092ae432fc75e71310cec4b4dde3"
# Explicit source modelling decisions. These remain draft policies awaiting human review.
RELEASE_OBJECT_TYPES = [
    "Material",
    "Field",
    "Field",
    "Resource",
    "Resource",
    "Field",
    "Material",
    "Resource",
    "Field",
    "Resource",
    "Rule",
    "Actor",
    "Actor",
    "Field",
    "Field",
    "Material",
]
RELEASE_ACTION_TYPES = [
    "Rule",
    "Step",
    "Rule",
    "Step",
    "Step",
    "Step",
    "Step",
    "Step",
    "Step",
    "Step",
    "Step",
    "Step",
    "Step",
    "Rule",
    "Step",
    "Step",
]


class GoldBuilder:
    def __init__(self, case):
        self.case = deepcopy(case)
        self.case.update(entities=[], relations=[], evidence=[], assertions=[], row_annotations=[])
        self.keys = {}

    def entity(self, name, kind, quote):
        key = (kind, normalize_name(name))
        if key not in self.keys:
            entity_id = f"e{len(self.keys) + 1}"
            self.keys[key] = entity_id
            self.case["entities"].append(
                {
                    "id": entity_id,
                    "type": kind,
                    "name": name,
                    "canonical_name": name,
                    "evidence_quote": quote,
                }
            )
        return self.keys[key]

    def relation(self, source, kind, target, quote, **semantics):
        rid = f"r{len(self.case['relations']) + 1}"
        self.case["relations"].append(
            {
                "id": rid,
                "source_id": source,
                "target_id": target,
                "type": kind,
                "evidence_quote": quote,
                "condition_text": None,
            }
        )
        self.case["assertions"].append(
            {
                "relation_id": rid,
                "modality": "normative",
                "actual_execution_asserted": False,
                **semantics,
            }
        )

    def finish(self):
        case = self.case
        case.update(
            gold_scope="all_data_rows_under_explicit_column_policy_not_all_possible_facts",
            review_status="codex_post_run_draft_item_human_review_pending",
        )
        for kind, records in (("entity", case["entities"]), ("relation", case["relations"])):
            for item in records:
                start = case["content"].index(item["evidence_quote"])
                case["evidence"].append(
                    {
                        "target_kind": kind,
                        "target_id": item["id"],
                        "start": start,
                        "end": start + len(item["evidence_quote"]),
                        "quote": item["evidence_quote"],
                    }
                )
        return case


def expand_gold(cases):
    """Column policy reads only frozen text; predictions are never an input."""
    expanded = []
    release_index = 0
    for case in cases:
        builder = GoldBuilder(case)
        doc = int(case["case_id"].split("-")[1])
        for row in table_rows(case):
            values, quote = row["values"], row["quote"]
            if not any(list(values.values())[1:]):
                builder.case["row_annotations"].append(
                    {
                        "quote": quote,
                        "scope": "disclaimer_not_negative_gold",
                        "reason": "范围说明保留为上下文，不声称不存在可建模的 Rule。",
                    }
                )
                continue
            builder.case["row_annotations"].append({"quote": quote, "scope": "data_row"})
            if doc == 1:
                process = builder.entity(values["工作环节"], "Process", quote)
                for role in ("申请人", "直属负责人", "资源负责人", "系统管理员"):
                    actor = builder.entity(role, "Actor", case["content"].splitlines()[0])
                    step = builder.entity(values[role], "Step", quote)
                    builder.relation(
                        actor,
                        "RESPONSIBLE_FOR",
                        step,
                        quote,
                        context=values["工作环节"],
                        negation_embedded_in_name=values[role].startswith(("不", "立即停止")),
                    )
                    builder.relation(process, "HAS_STEP", step, quote)
                rule = builder.entity(values["完成证据"], "Rule", quote)
                builder.relation(process, "GOVERNED_BY", rule, quote)
            elif doc == 2:
                state = builder.entity(values["工作状态"], "State", quote)
                entry = builder.entity(values["进入条件"], "Rule", quote)
                exit_rule = builder.entity(values["离开该状态前核对"], "Rule", quote)
                builder.entity(values["经办动作"], "Step", quote)
                builder.entity(values["需要留下的内容"], "Material", quote)
                builder.relation(state, "GOVERNED_BY", entry, quote, rule_role="entry_condition")
                builder.relation(state, "GOVERNED_BY", exit_rule, quote, rule_role="exit_check")
                # A record to retain does not establish which step created it.
            elif doc == 3:
                material = builder.entity(values["资料类别"], "Material", quote)
                rule = builder.entity(values["检查要点"], "Rule", quote)
                builder.entity(values["发现差异后"], "Step", quote)
                builder.relation(material, "GOVERNED_BY", rule, quote)
                builder.case["row_annotations"][-1].update(
                    content_description=values["文件或成果"],
                    usage=values["接收方用途"],
                    condition_context="发现差异后",
                    approved_policy=1,
                )
            elif doc == 4:
                obj_type = RELEASE_OBJECT_TYPES[release_index]
                action_type = RELEASE_ACTION_TYPES[release_index]
                release_index += 1
                obj = builder.entity(values["核对事项"], obj_type, quote)
                action = builder.entity(values["经办说明"], action_type, quote)
                passed = builder.entity(values["通过依据"], "Rule", quote)
                builder.entity(values["不通过时"], "Step", quote)
                if obj_type not in ("Actor", "Rule"):
                    builder.relation(obj, "GOVERNED_BY", passed, quote)
                    if action_type == "Rule":
                        builder.relation(obj, "GOVERNED_BY", action, quote)
                builder.case["row_annotations"][-1].update(
                    category=values["检查组"],
                    condition_context="不通过时",
                    approved_policy=2,
                    type_policy_status="DRAFT",
                )
            elif doc == 5:
                builder.entity("申请人", "Actor", case["content"].splitlines()[0])
                builder.entity(values["申请人应说明"], "Field", quote)
                builder.entity(values["需要对应的依据"], "Material", quote)
                builder.entity(values["审核关注点"], "Rule", quote)
                builder.entity(values["退回时说明"], "Step", quote)
                builder.case["row_annotations"][-1].update(
                    category=values["费用情形"],
                    approved_policy=3,
                    relation_gap="Actor→Field / Field→Material 无现有对应关系；不强建边。",
                )
            elif doc == 6:
                actor = builder.entity(values["主责角色"], "Actor", quote)
                step = builder.entity(values["环节"], "Step", quote)
                inputs = builder.entity(values["输入"], "Material", quote)
                result = builder.entity(values["形成的结果"], "Material", quote)
                rule = builder.entity(values["暂停或退回条件"], "Rule", quote)
                builder.relation(actor, "RESPONSIBLE_FOR", step, quote)
                builder.relation(step, "REQUIRES_MATERIAL", inputs, quote)
                builder.relation(step, "PRODUCES_MATERIAL", result, quote)
                builder.relation(
                    step,
                    "GOVERNED_BY",
                    rule,
                    quote,
                    required_effect="暂停或退回",
                    trigger=values["暂停或退回条件"],
                    approved_policy=5,
                    semantic_review="PENDING",
                )
        expanded.append(builder.finish())
    if release_index != 16:
        raise ValueError("发布表行数与显式类型口径不一致")
    return expanded


def evidence_decision(case_id, index):
    """Explicit decisions for the bound saved output, not keyword-based adjudication."""
    if case_id == "holdout-01-01" or (case_id == "holdout-01-02" and index <= 32):
        return (
            "SUPPORT",
            "角色列与本行职责明确对应；工作环节包含同一行职责动作。",
            "row_responsibility",
        )
    if case_id == "holdout-01-02":
        if index >= 37:
            return (
                "SUPPORT",
                "本 Chunk 的其他行支持该角色与动作，但模型引用来自未确认权限行，引用错位。",
                "quote_mismatch_with_chunk_support",
            )
        return (
            "INSUFFICIENT",
            "引用未确认权限行，却挂了其他行的动作；该引用不支持当前 Fact。",
            "cross_row_binding",
        )
    if case_id == "holdout-02-01":
        if index <= 7:
            return (
                "INSUFFICIENT",
                "只有状态行排列，没有这两个状态间直接流转的描述。",
                "ordered_states",
            )
        return "INSUFFICIENT", "需要留下的内容不证明该动作生成该内容。", "retention_not_production"
    if case_id == "holdout-02-02":
        if index <= 6:
            return (
                "SUPPORT",
                "明确说明本表状态名称不是线上固定枚举；作用域覆盖这些状态名称。",
                "scope_rule",
            )
        return "INSUFFICIENT", "保留内容与动作同一行不证明产生关系。", "retention_not_production"
    if case_id.startswith("holdout-03"):
        return (
            "SUPPORT",
            "原行明确给出资料/内容及检查约束；主资料命名是否命中 Gold 另行评估。",
            "material_rule",
        )
    if case_id == "holdout-04-01":
        return (
            "SUPPORT",
            "本行核对事项与经办要求对应；对象和动作的类型另作实体审核。",
            "release_requirement",
        )
    if case_id == "holdout-04-02":
        if index <= 5:
            return "INSUFFICIENT", "检查组是分类，不是承担动作的 Actor。", "category_as_actor"
        return (
            "INSUFFICIENT",
            "清单的范围声明不规定该动作受此规则约束；不得把范围说明直接变成事实边。",
            "scope_as_edge",
        )
    if case_id.startswith("holdout-05"):
        if case_id == "holdout-05-02" and index >= 17:
            if index == 18:
                return (
                    "INSUFFICIENT",
                    "同一支出不能重复办理约束支出办理，未规定原申请定位信息本身受此规则约束。",
                    "scope_as_edge",
                )
            return "SUPPORT", "本行票据/替代依据及其原始性要求有明确对应。", "material_rule"
        return (
            "INSUFFICIENT",
            "批准口径中说明内容是 Field，原 Fact 用 Step 端点强建职责或材料关系。",
            "field_as_step",
        )
    if case_id == "holdout-06-01" and index % 3 == 0:
        return (
            "REVIEW_REQUIRED",
            "只保留触发条件；仍需裁决 Fact 是否表达触发后暂停/退回的效果。",
            "missing_pause_effect",
        )
    if case_id == "holdout-06-02" and index % 4 == 0:
        return (
            "REVIEW_REQUIRED",
            "只保留触发条件；仍需裁决暂停/退回效果是否丢失。",
            "missing_pause_effect",
        )
    if case_id.startswith("holdout-06"):
        return (
            "SUPPORT",
            "本行主责、输入或形成结果明确对应；这里只解释流程定义，不断言采购实际发生。",
            "procurement_definition",
        )
    raise ValueError(f"没有逐条初审依据: {case_id}:{index}")


def entity_key(entity):
    return entity["type"], normalize_name(entity["canonical_name"])


def triple(relation, entities):
    return (
        entity_key(entities[relation["source_id"]]),
        relation["type"],
        entity_key(entities[relation["target_id"]]),
    )


def analyze_recall(case, extraction):
    """Partition all misses and use multiset matching so hits cannot exceed Gold."""
    predicted = extraction.model_dump()
    pe = {e["id"]: e for e in predicted["entities"]}
    ge = {e["id"]: e for e in case["entities"]}
    entity_pool = Counter(entity_key(e) for e in pe.values())
    available_entities = entity_pool.copy()
    triples = Counter(triple(r, pe) for r in predicted["relations"])
    remaining = triples.copy()
    entity_outcomes, relation_outcomes = [], []
    for entity in ge.values():
        key = entity_key(entity)
        hit = available_entities[key] > 0
        if hit:
            available_entities[key] -= 1
        same_name = [
            e["type"] for e in pe.values() if normalize_name(e["canonical_name"]) == key[1]
        ]
        reason = "HIT" if hit else "WRONG_TYPE" if same_name else "NAME_NOT_EXTRACTED"
        entity_outcomes.append(
            {
                "case_id": case["case_id"],
                "entity_id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "outcome": reason,
                "same_name_predicted_types": same_name,
            }
        )
    for relation in case["relations"]:
        key = triple(relation, ge)
        source_present = entity_pool[key[0]] > 0
        target_present = entity_pool[key[2]] > 0
        hit = remaining[key] > 0
        if hit:
            remaining[key] -= 1
        reason = (
            "HIT"
            if hit
            else "BOTH_ENDPOINTS_MISSING"
            if not source_present and not target_present
            else "SOURCE_MISSING"
            if not source_present
            else "TARGET_MISSING"
            if not target_present
            else "RELATION_TYPE_OR_DIRECTION_OR_EDGE_MISSING"
        )
        reverse = (key[2], key[1], key[0])
        wrong_types = [
            r["type"]
            for r in predicted["relations"]
            if entity_key(pe[r["source_id"]]) == key[0]
            and entity_key(pe[r["target_id"]]) == key[2]
            and r["type"] != key[1]
        ]
        relation_outcomes.append(
            {
                "case_id": case["case_id"],
                "relation_id": relation["id"],
                "source": ge[relation["source_id"]]["name"],
                "type": relation["type"],
                "target": ge[relation["target_id"]]["name"],
                "outcome": reason,
                "endpoints_present": source_present and target_present,
                "reverse_triple_present": triples[reverse] > 0,
                "other_predicted_types": wrong_types,
                "evidence_quote": relation["evidence_quote"],
            }
        )
    return entity_outcomes, relation_outcomes


def prepare(cases, predictions):
    gold = expand_gold(cases)
    entities, relations, evidence, entity_reviews, normalization = [], [], [], [], []
    for case in gold:
        saved = predictions[case["case_id"]]
        if saved["content_sha256"] != case["content_sha256"]:
            raise ValueError("预测正文不属于当前冻结 Chunk")
        extraction = Extraction.model_validate(saved["extraction"])
        validate_evidence(extraction, case["content"])
        eo, ro = analyze_recall(case, extraction)
        entities.extend(eo)
        relations.extend(ro)
        by_id = {e.id: e for e in extraction.entities}
        rows = table_rows(case)
        groups = {r["values"].get("检查组") for r in rows} - {None, ""}
        fields = {r["values"].get("申请人应说明") for r in rows} - {None, ""}
        audit_rules = {r["values"].get("审核关注点") for r in rows} - {None, ""}
        for entity in extraction.entities:
            verdict, reason = (
                "REVIEW_REQUIRED",
                "尚未逐条确认业务实体边界；不因未命中 Gold 就自动计 FP。",
            )
            if entity.type == "Actor" and entity.name in groups:
                verdict, reason = "INVALID_TYPE", "批准规则2：检查组是分类。"
            elif entity.type != "Field" and entity.name in fields:
                verdict, reason = "INVALID_TYPE", "批准规则3：说明内容是 Field。"
            elif entity.type == "Step" and entity.name in audit_rules:
                verdict, reason = (
                    "TYPE_REVIEW",
                    "审核关注点更接近 Rule，属于新增草稿口径，尚待确认。",
                )
            elif entity_key(entity.model_dump()) in {entity_key(e) for e in case["entities"]}:
                verdict, reason = (
                    "MATCHES_DRAFT",
                    "名称和类型匹配来源派生 Gold 草稿；仍待逐条人工确认。",
                )
            entity_reviews.append(
                {
                    "review_id": f"{case['case_id']}:entity:{entity.id}",
                    "case_id": case["case_id"],
                    **entity.model_dump(),
                    "codex_suggestion": verdict,
                    "reason": reason,
                    "item_human_review": "PENDING",
                }
            )
        for index, relation in enumerate(extraction.relations, 1):
            verdict, reason, group = evidence_decision(case["case_id"], index)
            source, target = by_id[relation.source_id], by_id[relation.target_id]
            cited_rows = [r["quote"] for r in rows if relation.evidence_quote in r["quote"]]
            endpoint_rows = [
                r["quote"]
                for r in rows
                if (source.name in r["quote"] or source.name in case["content"].splitlines()[0])
                and target.name in r["quote"]
            ]
            evidence.append(
                {
                    "review_id": f"{case['case_id']}:relation:{index}",
                    "case_id": case["case_id"],
                    "relation_index": index,
                    **relation.model_dump(),
                    "source": source.name,
                    "source_type": source.type,
                    "target": target.name,
                    "target_type": target.type,
                    "header": case["content"].splitlines()[0],
                    "cited_rows": cited_rows,
                    "endpoint_context_rows": endpoint_rows,
                    "codex_suggestion": verdict,
                    "quote_support_suggestion": (
                        "INSUFFICIENT" if group == "quote_mismatch_with_chunk_support" else verdict
                    ),
                    "support_scope": "frozen_chunk_and_its_table_header",
                    "reason": reason,
                    "group": group,
                    "item_human_review": "PENDING",
                    "human_verdict": None,
                }
            )
        if case["case_id"].startswith("holdout-03"):
            for row in rows:
                values = row["values"]
                if not values["文件或成果"]:
                    continue
                normalization.append(
                    {
                        "case_id": case["case_id"],
                        "left": values["资料类别"],
                        "right": values["文件或成果"],
                        "expected": "DO_NOT_AUTO_MERGE",
                        "approved_policy": 1,
                        "reason": "资料名称与内容描述不是已证明的别名。",
                        "system_merge_observed": None,
                        "item_human_review": "PENDING",
                    }
                )
    ec = Counter(r["outcome"] for r in entities)
    rc = Counter(r["outcome"] for r in relations)
    conditional = [r for r in relations if r["endpoints_present"]]
    summary = {
        "documents": 6,
        "chunks": len(gold),
        "gold_entities": len(entities),
        "gold_relations": len(relations),
        "entity_outcomes": dict(ec),
        "relation_outcomes": dict(rc),
        "draft_entity_recall": ec["HIT"] / len(entities),
        "draft_triple_recall": rc["HIT"] / len(relations),
        "conditional_triple_recall": {
            "hits": sum(r["outcome"] == "HIT" for r in conditional),
            "eligible": len(conditional),
        },
        "evidence_codex_suggestions": dict(Counter(r["codex_suggestion"] for r in evidence)),
        "quote_support_codex_suggestions": dict(
            Counter(r["quote_support_suggestion"] for r in evidence)
        ),
        "predicted_entity_review_count": len(entity_reviews),
        "predicted_relation_review_count": len(evidence),
        "normalization_pair_count": len(normalization),
        "full_item_human_review": "PENDING",
        "annotation_timing": "post_prediction",
        "gold_coverage": (
            "all data rows; explicitly selected columns/relations; not universally exhaustive"
        ),
        "final_precision": None,
        "final_recall": None,
        "final_f1": None,
        "formal_evidence_precision": None,
        "duplicate_entity_rate": None,
        "merge_error_rate": None,
        "semantic_condition_recall": None,
        "metrics_scope": (
            "draft structural matches exclude pause/return effect correctness "
            "and independent human acceptance"
        ),
    }
    return {
        "cases.jsonl": gold,
        "entity_recall_outcomes.jsonl": entities,
        "relation_loss_outcomes.jsonl": relations,
        "entity_review.jsonl": entity_reviews,
        "evidence_review.jsonl": evidence,
        "normalization_review.jsonl": normalization,
    }, summary


def markdown_review(items, summary):
    evidence = items["evidence_review.jsonl"]
    groups = {}
    for item in evidence:
        groups.setdefault((item["codex_suggestion"], item["group"]), []).append(item)
    text = [
        "# 逐条人工裁决清单",
        "",
        "这里是 Codex 初审建议。查看原文后按 review_id 裁决；批准规则不等于批准所有条目。",
        "最终可选 SUPPORT / CONTRADICT / INSUFFICIENT。",
        "REVIEW_REQUIRED 是未决状态，不是第四种最终证据标签。",
        "引用支持按所引原文、其所在表格行和列头判断；Chunk 支持允许查看本块其他行。",
        "SUPPORT 仅判断证据是否支持 Fact；实体类型、主实体命名、条件效果需分别检查。",
        "",
        "## 优先检查的批次",
        "",
        "| 建议 | 分组 | 条数 | 示例 |",
        "| --- | --- | ---: | --- |",
    ]
    for (verdict, group), batch in groups.items():
        sample = batch[0]
        text.append(
            f"| {verdict} | {group} | {len(batch)} | {sample['review_id']}："
            f"{sample['source']} → {sample['target']} |"
        )
    text.extend(
        [
            "",
            "仅批准批次时，应明确批次名称和全部条目范围；当前所有人工状态仍为 PENDING。",
            "14 条暂停/退回关系请先裁决：裸 Rule 是否不足以表达触发后的处置效果。",
            "新增 Gold 类型待审：权限环节=Process；费用审核关注点=Rule。",
            "发布表每行类型见 cases.jsonl，均待人工确认。",
            "",
            "## 所有预测关系及原文",
            "",
        ]
    )
    for item in evidence:
        text.extend(
            [
                f"### {item['review_id']}",
                "",
                f"`{item['source']} ({item['source_type']}) --{item['type']}--> "
                f"{item['target']} ({item['target_type']})`",
                "",
                f"- Codex 建议：**{item['codex_suggestion']}**；人工：PENDING",
                f"- 引用本身的支持建议：{item['quote_support_suggestion']}",
                f"- 理由：{item['reason']}",
                f"- 条件：{item['condition_text'] or '未表达'}",
                f"- 模型引用：{item['evidence_quote']}",
                f"- 原列头：`{item['header']}`",
            ]
        )
        for row in dict.fromkeys(item["cited_rows"] + item["endpoint_context_rows"]):
            text.append(f"- 上下文：`{row}`")
        text.append("")
    return "\n".join(text)


def build(output=OUTPUT):
    if output.exists():
        raise FileExistsError("草稿目录已存在，不覆盖；请使用 verify")
    if sha(RUN / "replay.jsonl") != EXPECTED_REPLAY_SHA256:
        raise ValueError("当前逐条初审只绑定已阅读的 v2 预测；预测变化后必须重新初审")
    cases = read_jsonl(POLICY / "cases.jsonl")
    predictions = {r["case_id"]: r for r in read_jsonl(RUN / "replay.jsonl")}
    items, summary = prepare(cases, predictions)
    output.mkdir(parents=True)
    for name, rows in items.items():
        write_jsonl(output / name, rows)
    write_json(output / "summary.json", summary)
    (output / "人工裁决清单.md").write_text(markdown_review(items, summary), encoding="utf-8")
    first_batch = [
        r
        for r in items["evidence_review.jsonl"]
        if r["group"] in ("cross_row_binding", "quote_mismatch_with_chunk_support")
    ]
    expected_row = next(
        row["quote"]
        for case in cases
        if case["case_id"] == "holdout-01-02"
        for row in table_rows(case)
        if row["values"]["工作环节"] == "未确认权限"
    )
    batch_text = [
        "# 第一批人工裁决：8 条跨行错配",
        "",
        "人工状态：PENDING。下面展示本批全部 8 条，并非随机示例。",
        "",
        "## 对照原文",
        "",
        "列头：工作环节 / 申请人 / 直属负责人 / 资源负责人 / 系统管理员 / 完成证据",
        "",
        f"`{expected_row}`",
        "",
        "本行实际动作分别是：补充说明、明确临时责任、决定暂停或继续、不擅自扩大。",
        "预测使用了共享链接检查或定期复核等其他行的动作。",
        "",
        "## 待确认条目",
        "",
        "| review_id | 预测 Fact | 模型引用 | Chunk 支持建议 | 引用支持建议 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in first_batch:
        batch_text.append(
            f"| {item['review_id']} | {item['source']} --{item['type']}--> "
            f"{item['target']} | {item['evidence_quote']} | {item['codex_suggestion']} | "
            f"{item['quote_support_suggestion']} |"
        )
    batch_text.extend(["", "## 支持 37–40 的其他原始行", ""])
    for item in first_batch:
        if item["group"] == "quote_mismatch_with_chunk_support":
            batch_text.extend(
                f"\n支持 {item['review_id']} 的其他行：`{row}`\n"
                for row in item["endpoint_context_rows"]
            )
    batch_text.extend(
        [
            "",
            "33–36：Chunk 不支持 Fact，建议 INSUFFICIENT。",
            "37–40：Chunk 其他行支持 Fact，建议 SUPPORT；模型引用均错位，引用支持为 INSUFFICIENT。",
            "这里只记录其他行依据，没有修改模型保存的证据引用。",
            "若认可本批全部条目，可回复：第一批按清单裁决，"
            "4 条事实依据不足，4 条事实支持但引用错位。",
            "这只确认上述 8 条；其他预测、Gold 新增口径和暂停/退回效果继续待审。",
            "",
        ]
    )
    (output / "第一批裁决-跨行错配.md").write_text("\n".join(batch_text), encoding="utf-8")
    by_doc = []
    for doc in range(1, 7):
        prefix = f"holdout-{doc:02d}-"
        source = next(c["source"] for c in cases if c["case_id"].startswith(prefix))
        label = source.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".xlsx")
        ent = [r for r in items["entity_recall_outcomes.jsonl"] if r["case_id"].startswith(prefix)]
        rel = [r for r in items["relation_loss_outcomes.jsonl"] if r["case_id"].startswith(prefix)]
        by_doc.append(
            f"| {label} | {sum(r['outcome'] == 'HIT' for r in ent)}/{len(ent)} | "
            f"{sum(r['outcome'] == 'HIT' for r in rel)}/{len(rel)} |"
        )
    rc = summary["relation_outcomes"]
    missing_endpoint = sum(
        rc.get(k, 0) for k in ("BOTH_ENDPOINTS_MISSING", "SOURCE_MISSING", "TARGET_MISSING")
    )
    ch = summary["conditional_triple_recall"]
    readme = f"""# 六份文档的 Gold 扩展草稿与逐条裁决材料

本次使用已冻结的 13 Chunk 和已保存的同一份 v2 名称反馈预测，没有重新请求模型。
所有数据来源是虚构企业测试文档。本版本在看过预测后补标，属于诊断/回归资料，不能作为独立留出验收集。

## 新 Gold 范围

覆盖六份文档全部数据行，草稿 {summary["gold_entities"]} 个实体、{summary["gold_relations"]} 条关系。
列和关系按原文与显式口径建模：权限职责、状态进入/离开约束、资料检查约束、发布事项及通过依据、
费用 Field/材料/审核 Rule、采购职责/输入/结果/暂停条件。范围说明与未建模列保留在 row_annotations。
没有把范围说明当负例，没有按模型输出删去漏抽目标，没有将所有共现列强建关系。

这不是“所有可能实体和关系均穷尽”的正式 Gold。费用 Field 与材料的语义在当前关系白名单中
没有直接对应关系，因此保留实体与关系缺口说明；未扩展生产 schema。
发布事项逐行类型、权限环节 Process、费用审核 Rule 等新增口径均待人工审核。
采购暂停条件的处置效果保存在 assertions 中，与裸三元组命中分开；未用结构命中代表语义通过。

## 草稿结构指标（尚未人工验收）

| 文档 | Entity 命中/目标 | Triple 命中/目标 |
| --- | --- | --- |
{chr(10).join(by_doc)}

- Entity：{summary["entity_outcomes"].get("HIT", 0)}/{summary["gold_entities"]}。
- Directed Triple：{rc.get("HIT", 0)}/{summary["gold_relations"]}。
- 未命中的关系中，至少一个端点缺失：{missing_endpoint}。
- 端点已齐但关系类型/方向/边未命中：{rc.get("RELATION_TYPE_OR_DIRECTION_OR_EDGE_MISSING", 0)}。
- 两端已抽到的条件下，Triple 命中：{ch["hits"]}/{ch["eligible"]}。
  这是结构 Conditional Recall，未包含暂停/退回等语义效果是否正确。
- 旧 45/57、19/30 保留在原人工口径版本；本版本分母和建模范围扩大，不能直接当作模型性能变化。

## 初审与人工审核

预测实体：{summary["predicted_entity_review_count"]}；预测关系：{summary["predicted_relation_review_count"]}。
均有逐条审核记录，人工状态全部待确认。
关系的 Codex 初审建议：{json.dumps(summary["evidence_codex_suggestions"], ensure_ascii=False)}。
引用本身的初审建议：{json.dumps(summary["quote_support_codex_suggestions"], ensure_ascii=False)}。
Chunk 支持 Fact 与预测引用支持 Fact 分开保存；引用错位不自动等于该 Chunk 无法支持 Fact。
引用支持包括所引文本所在行及表格列头；不得用其他行替换模型原引用来认定引用正确。
REVIEW_REQUIRED 代表未决，最终三分类仍是 SUPPORT / CONTRADICT / INSUFFICIENT。
当前无已确认的人工证据标签。
Evidence 支持、实体类型正确、主实体边界命中、条件完整是不同判断，不能互相代替。

先看 [第一批 8 条跨行错配](第一批裁决-跨行错配.md)，再按 [完整清单](人工裁决清单.md) 继续。
未命中 Gold 的预测并未自动判为 FP。
15 对资料类别/内容描述按批准规则列出“不可自动合并”候选，但没有系统全局合并轨迹，
因此 Duplicate Entity Rate / Merge Error Rate 仍不可计算；跨 Chunk 同名角色也未直接判作重复错误。
Precision、最终 Recall/F1、正式 Evidence Precision 和条件语义 Recall 仍为 null。
最终发布指标需人工确认新增 Gold 口径、逐条裁决和完整性范围；本材料不宣告离线测评通过。

## 可复核文件

- cases.jsonl：冻结原文、Gold 草稿、证据字符偏移、条件语义和未建模说明。
- entity_recall_outcomes.jsonl：命中/同名错误类型/名称未抽到。
- relation_loss_outcomes.jsonl：三元组命中与端点损失拆分、反向及其他关系类型线索。
- entity_review.jsonl / evidence_review.jsonl：全部预测的初审与 PENDING 人工状态。
- normalization_review.jsonl：15 对名称与内容描述的不可自动合并规则。
- manifest.json：父级 Gold、预测、生成脚本及输出文件哈希。

```powershell
cd D:\\code\\zhiqing-rag-platform
uv run python -m scripts.prepare_graph_holdout_review verify
```

本次只创建本地评估草稿与裁决材料，未改生产抽取、未外发数据、未写 PostgreSQL 或 Neo4j。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "schema_version": "graph-six-document-review-draft-v1",
            "parent_cases_sha256": sha(POLICY / "cases.jsonl"),
            "prediction_sha256": sha(RUN / "replay.jsonl"),
            "generator_sha256": sha(Path(__file__)),
            "annotation_timing": "post_prediction",
            "human_review": "PENDING",
            "files": {p.name: sha(p) for p in output.iterdir()},
        },
    )
    verify(output)
    print(json.dumps(summary, ensure_ascii=False))


def verify(output=OUTPUT):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in (
        (POLICY / "cases.jsonl", manifest["parent_cases_sha256"]),
        (RUN / "replay.jsonl", manifest["prediction_sha256"]),
        (Path(__file__), manifest["generator_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError(f"父级输入/预测/生成器哈希改变: {path.name}")
    for name, expected in manifest["files"].items():
        if sha(output / name) != expected:
            raise ValueError(f"审核文件哈希改变: {name}")
    originals = read_jsonl(POLICY / "cases.jsonl")
    predictions = {r["case_id"]: r for r in read_jsonl(RUN / "replay.jsonl")}
    items, summary = prepare(originals, predictions)
    for name, expected in items.items():
        if read_jsonl(output / name) != expected:
            raise ValueError(f"审核草稿无法重现: {name}")
    if json.loads((output / "summary.json").read_text(encoding="utf-8")) != summary:
        raise ValueError("指标草稿无法重现")
    for case in items["cases.jsonl"]:
        validate_evidence(
            Extraction.model_validate(
                {
                    "entities": case["entities"],
                    "relations": [
                        {k: v for k, v in r.items() if k != "id"} for r in case["relations"]
                    ],
                }
            ),
            case["content"],
        )
        for evidence in case["evidence"]:
            if case["content"][evidence["start"] : evidence["end"]] != evidence["quote"]:
                raise ValueError("证据字符偏移错误")
    if len(items["entity_review.jsonl"]) != 266 or len(items["evidence_review.jsonl"]) != 202:
        raise ValueError("预测审核覆盖不完整")
    print("本地草稿校验通过：六份文档、13 Chunk；266实体/202关系逐条初审，人工状态仍 PENDING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.command == "build" else verify()
