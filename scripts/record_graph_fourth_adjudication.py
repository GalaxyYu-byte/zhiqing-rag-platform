"""Record the user's 'all' reply for the fourteen displayed fourth-batch items only."""

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.prepare_graph_holdout_review import OUTPUT as DRAFT
from scripts.record_graph_item_adjudication import summarize
from scripts.record_graph_third_adjudication import OUTPUT as PARENT
from scripts.record_graph_third_adjudication import verify as verify_parent

OUTPUT = PILOT / "human-item-adjudication-v4"
USER_REPLY = "全部都判为INSUFFICIENT"
BATCH_IDS = [f"holdout-02-01:relation:{i}" for i in range(8, 16)] + [
    f"holdout-02-02:relation:{i}" for i in range(7, 13)
]


def adjudicate_fourth_batch(records):
    ids = [r["review_id"] for r in records]
    if len(ids) != len(set(ids)) or not set(BATCH_IDS).issubset(ids):
        raise ValueError("第四批条目缺失或审核 ID 重复")
    revised = deepcopy(records)
    for row in revised:
        if row["review_id"] not in BATCH_IDS:
            continue
        if (
            row["type"] != "PRODUCES_MATERIAL"
            or row["source_type"] != "Step"
            or row["target_type"] != "Material"
            or row["group"] != "retention_not_production"
            or row["codex_suggestion"] != "INSUFFICIENT"
            or row["quote_support_suggestion"] != "INSUFFICIENT"
            or row["item_human_review"] != "PENDING"
            or row["human_verdict"] is not None
        ):
            raise ValueError("条目与用户已批准的第四批不一致")
        row.update(
            item_human_review="REVIEWED",
            human_verdict="INSUFFICIENT",
            human_quote_verdict="INSUFFICIENT",
            human_reviewer="human_user_in_current_chat",
            human_review_basis="explicit_all_reply_to_fourteen_displayed_fourth_batch_items",
            human_batch_id="fourth_batch_retention_not_production",
        )
    return revised


def summary_for(records):
    summary = summarize(records)
    summary["scope"] = "34 explicit evidence decisions; entities and new Gold policies pending"
    return summary


def next_batch_document(records):
    selected = [r for r in records if r["group"] == "field_as_step"]
    if len(selected) != 30 or any(r["item_human_review"] != "PENDING" for r in selected):
        raise ValueError("第五批待审范围变化")
    cases = {c["case_id"]: c for c in read_jsonl(DRAFT / "cases.jsonl")}
    lines = [
        "# 第五批待裁决：30 条把说明字段当步骤的关系",
        "",
        "人工状态：PENDING。本文件展示本批全部 30 条关系。",
        "已批准规则3：申请人应说明的内容按 Field 处理；原 Step 及其关系需重新标注。",
        "",
        "本批预测将出行目的与行程变化等说明内容建成 Step，",
        "进而建立 Actor RESPONSIBLE_FOR Step 或 Step REQUIRES_MATERIAL Material。",
        "请检查这些具体带类型的 Fact 是否得到原文支持。说明字段和依据材料的同一行共现，",
        "需要按批准的 Field 口径重新建模；当前关系端点仍写成 Step。",
    ]
    for case_id in dict.fromkeys(r["case_id"] for r in selected):
        lines.extend(
            [
                "",
                f"## {case_id} 原始 Chunk",
                "",
                cases[case_id]["content"],
                "",
                "| review_id | 预测 Fact（含端点类型） | 模型原引用 | Codex 建议 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in selected:
            if row["case_id"] != case_id:
                continue
            quote = row["evidence_quote"].replace("|", "\\|").replace("\n", "<br>")
            lines.append(
                f"| {row['review_id']} | {row['source']} ({row['source_type']}) "
                f"--{row['type']}--> {row['target']} ({row['target_type']}) | "
                f"{quote} | INSUFFICIENT |"
            )
    lines.extend(
        [
            "",
            "建议理由：这 30 条具体 Fact 使用的 Step 端点与批准的 Field 口径不符，",
            "当前 Fact 的职责/步骤材料关系需重标，建议证据判 INSUFFICIENT。",
            "此裁决只覆盖上述 30 条关系的证据支持；实体类型和修订后的正确关系仍另行审核。",
            "",
            "若认可本批全部条目，可回复：第五批 30 条均判 INSUFFICIENT。",
            "也可以指出具体条目的不同意见。当前第五批人工状态仍为 PENDING。",
            "",
        ]
    )
    return "\n".join(lines)


def build(output=OUTPUT):
    if output.exists():
        raise FileExistsError("第四批裁决版本已存在，请使用 verify；不覆盖历史记录")
    verify_parent()
    revised = adjudicate_fourth_batch(read_jsonl(PARENT / "evidence_review.jsonl"))
    summary = summary_for(revised)
    decisions = json.loads((PARENT / "human_decisions.json").read_text(encoding="utf-8"))
    decisions.append(
        {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "reviewer": "human_user_in_current_chat",
            "verbatim_reply": USER_REPLY,
            "approval_basis": "direct_all_reply_to_the_complete_displayed_fourth_batch",
            "approved_review_ids": BATCH_IDS,
            "displayed_batch_sha256": sha(PARENT / "第四批裁决-保留与生成.md"),
            "scope_resolution": "all refers to the current fourteen-item fourth batch only",
            "scope": "chunk support and quote support for these fourteen relation predictions only",
        }
    )
    output.mkdir(parents=True)
    write_jsonl(output / "evidence_review.jsonl", revised)
    write_jsonl(
        output / "human_verdicts.jsonl",
        [r for r in revised if r["item_human_review"] == "REVIEWED"],
    )
    write_json(output / "human_decisions.json", decisions)
    write_json(output / "summary.json", summary)
    (output / "第五批裁决-说明字段与步骤.md").write_text(
        next_batch_document(revised), encoding="utf-8"
    )
    lines = [
        "# 第一至四批人工裁决记录",
        "",
        f"第四批用户原回复：**{USER_REPLY}**。",
        "",
        "依据紧接的第四批清单，将“全部”解释为该清单已展示的 14 条。",
        "本版仅新增第四批 14 条裁决，前三批 20 条完整保留。",
        "",
        "| 第四批 review_id | Chunk 支持 | 引用支持 | 人工状态 |",
        "| --- | --- | --- | --- |",
    ]
    for row in revised:
        if row["review_id"] in BATCH_IDS:
            lines.append(f"| {row['review_id']} | INSUFFICIENT | INSUFFICIENT | REVIEWED |")
    lines.extend(
        [
            "",
            "## 累计审核进度",
            "",
            "- 关系证据：34/202 已人工裁决，168 条待审。",
            "- 已审 Fact：4 SUPPORT、30 INSUFFICIENT；原引用：34 INSUFFICIENT。",
            "- 266 个预测实体、新增 Gold 类型口径、Normalization 仍待审核。",
            "- 异常批次的比例不作为整批预测的 Evidence Precision。",
            "- 正式 Precision / Recall / F1、Evidence Precision 仍未发布。",
            "",
            "## 可复核记录",
            "",
            "evidence_review.jsonl 保留全部 202 条；仅指定 14 条新增人工状态。",
            "human_verdicts.jsonl 保存累计 34 条人工裁决。",
            "human_decisions.json 保存四次原回复、批准范围和所展示清单的哈希。",
            "manifest.json 绑定父级文件、草稿原文、生成脚本和所有输出。",
            "原预测、Gold 草稿、前三版裁决和生成脚本保持原样。",
            "",
            "[第四批全部条目与原文](../human-item-adjudication-v3/第四批裁决-保留与生成.md)",
            "[第五批 30 条说明字段/步骤关系待裁决](第五批裁决-说明字段与步骤.md)",
            "",
            "```powershell",
            "cd D:\\code\\zhiqing-rag-platform",
            "uv run python -m scripts.record_graph_fourth_adjudication verify",
            "```",
            "",
            "本次仅记录本地人工裁决，未调用模型、修改生产抽取或写入 PostgreSQL/Neo4j。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "schema_version": "graph-human-item-adjudication-v4",
            "generator_sha256": sha(Path(__file__)),
            "parent_files": {
                name: sha(PARENT / name)
                for name in (
                    "manifest.json",
                    "evidence_review.jsonl",
                    "human_decisions.json",
                    "第四批裁决-保留与生成.md",
                )
            },
            "draft_cases_sha256": sha(DRAFT / "cases.jsonl"),
            "files": {p.name: sha(p) for p in output.iterdir()},
            "full_item_human_review": "PARTIAL",
        },
    )
    verify(output)


def verify(output=OUTPUT):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if (
        sha(Path(__file__)) != manifest["generator_sha256"]
        or sha(DRAFT / "cases.jsonl") != manifest["draft_cases_sha256"]
    ):
        raise ValueError("生成脚本或草稿原文改变")
    for name, expected in manifest["parent_files"].items():
        if sha(PARENT / name) != expected:
            raise ValueError(f"父级裁决输入改变: {name}")
    for name, expected in manifest["files"].items():
        if sha(output / name) != expected:
            raise ValueError(f"裁决文件改变: {name}")
    revised = read_jsonl(output / "evidence_review.jsonl")
    if revised != adjudicate_fourth_batch(read_jsonl(PARENT / "evidence_review.jsonl")):
        raise ValueError("第四批裁决无法重现，或改变了前三批结果")
    decisions = json.loads((output / "human_decisions.json").read_text(encoding="utf-8"))
    prior = json.loads((PARENT / "human_decisions.json").read_text(encoding="utf-8"))
    if (
        len(decisions) != 4
        or decisions[:-1] != prior
        or decisions[-1]["verbatim_reply"] != USER_REPLY
        or decisions[-1]["approved_review_ids"] != BATCH_IDS
        or decisions[-1]["scope_resolution"]
        != "all refers to the current fourteen-item fourth batch only"
        or decisions[-1]["displayed_batch_sha256"] != sha(PARENT / "第四批裁决-保留与生成.md")
    ):
        raise ValueError("批准来源或条目范围不一致")
    if read_jsonl(output / "human_verdicts.jsonl") != [
        r for r in revised if r["item_human_review"] == "REVIEWED"
    ]:
        raise ValueError("人工裁决与完整队列不一致")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary != summary_for(revised) or summary["human_reviewed_relations"] != 34:
        raise ValueError("累计人工审核进度不一致")
    if (output / "第五批裁决-说明字段与步骤.md").read_text(encoding="utf-8") != next_batch_document(
        revised
    ):
        raise ValueError("第五批材料无法重现")
    print("第四批裁决校验通过：累计34/202已审，168待审；Fact 4支持/30依据不足，引用34依据不足")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.command == "build" else verify()
