"""Record the seven state-transition adjudications without modifying the first version."""

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.prepare_graph_holdout_review import OUTPUT as DRAFT
from scripts.record_graph_item_adjudication import OUTPUT as PARENT
from scripts.record_graph_item_adjudication import summarize
from scripts.record_graph_item_adjudication import verify as verify_parent

OUTPUT = PILOT / "human-item-adjudication-v2"
USER_REPLY = "第二批 7 条均判 INSUFFICIENT"
BATCH_IDS = [f"holdout-02-01:relation:{index}" for index in range(1, 8)]


def adjudicate_second_batch(records):
    ids = [r["review_id"] for r in records]
    if len(ids) != len(set(ids)) or not set(BATCH_IDS).issubset(ids):
        raise ValueError("第二批条目缺失或审核 ID 重复")
    revised = deepcopy(records)
    for row in revised:
        if row["review_id"] not in BATCH_IDS:
            continue
        if (
            row["case_id"] != "holdout-02-01"
            or row["type"] != "TRANSITIONS_TO"
            or row["group"] != "ordered_states"
            or row["codex_suggestion"] != "INSUFFICIENT"
            or row["quote_support_suggestion"] != "INSUFFICIENT"
            or row["item_human_review"] != "PENDING"
            or row["human_verdict"] is not None
        ):
            raise ValueError("条目与用户已批准的第二批不一致")
        row.update(
            item_human_review="REVIEWED",
            human_verdict="INSUFFICIENT",
            human_quote_verdict="INSUFFICIENT",
            human_reviewer="human_user_in_current_chat",
            human_review_basis="explicit_reply_approving_all_seven_displayed_batch_items",
            human_batch_id="second_batch_ordered_states",
        )
    return revised


def summary_for(records):
    summary = summarize(records)
    summary["scope"] = "fifteen explicit evidence decisions; entities and new Gold policies pending"
    return summary


def next_batch_document(records):
    selected = [
        r for r in records if r["case_id"] == "holdout-04-02" and r["group"] == "category_as_actor"
    ]
    if len(selected) != 5 or any(r["item_human_review"] != "PENDING" for r in selected):
        raise ValueError("第三批待审范围变化")
    case = next(c for c in read_jsonl(DRAFT / "cases.jsonl") if c["case_id"] == "holdout-04-02")
    lines = [
        "# 第三批待裁决：5 条把检查分类当责任人的关系",
        "",
        "人工状态：PENDING。用户此前批准规则2：检查组是分类，不能仅据此建 Actor。",
        "这里展示本批全部 5 条具体预测；第二批的批准不覆盖本批。",
        "",
        "## 原始 Chunk",
        "",
        case["content"],
        "",
        "## 待裁决条目",
        "",
        "| review_id | 预测 Fact | 模型原引用 | Codex 建议 |",
        "| --- | --- | --- | --- |",
    ]
    for row in selected:
        quote = row["evidence_quote"].replace("|", "\\|").replace("\n", "<br>")
        lines.append(
            f"| {row['review_id']} | {row['source']} (Actor) --{row['type']}--> "
            f"{row['target']} | {quote} | INSUFFICIENT |"
        )
    lines.extend(
        [
            "",
            "理由：沟通、执行、观察、归档来自检查组列，是分类，原文未把这些分类当作责任角色。",
            "因此本批 Fact 的 Actor 职责关系缺少依据。没有依据不等于明确被原文否定，",
            "建议 INSUFFICIENT，不判 CONTRADICT。",
            "",
            "本批只裁决上述 5 条关系的证据支持，不同时确认所有实体类型或其他关系。",
            "若核对原文后认可本批全部条目，可回复：第三批 5 条均判 INSUFFICIENT。",
            "",
        ]
    )
    return "\n".join(lines)


def build(output=OUTPUT):
    if output.exists():
        raise FileExistsError("第二批裁决版本已存在，请使用 verify；不覆盖历史记录")
    verify_parent()
    revised = adjudicate_second_batch(read_jsonl(PARENT / "evidence_review.jsonl"))
    summary = summary_for(revised)
    prior = json.loads((PARENT / "human_decision.json").read_text(encoding="utf-8"))
    decision = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "reviewer": "human_user_in_current_chat",
        "verbatim_reply": USER_REPLY,
        "approval_basis": "direct_user_reply_to_the_complete_displayed_second_batch",
        "approved_review_ids": BATCH_IDS,
        "displayed_batch_sha256": sha(PARENT / "第二批裁决-状态流转.md"),
        "scope": "chunk support and quote support for these seven relation predictions only",
    }
    output.mkdir(parents=True)
    write_jsonl(output / "evidence_review.jsonl", revised)
    write_jsonl(
        output / "human_verdicts.jsonl",
        [r for r in revised if r["item_human_review"] == "REVIEWED"],
    )
    write_json(output / "human_decisions.json", [prior, decision])
    write_json(output / "summary.json", summary)
    (output / "第三批裁决-分类责任人.md").write_text(next_batch_document(revised), encoding="utf-8")
    lines = [
        "# 第一、二批人工裁决记录",
        "",
        f"第二批用户回复：**{USER_REPLY}**。",
        "",
        "第一批 8 条裁决完整保留；本版仅新增第二批 7 条具体条目的裁决。",
        "",
        "| 条目 | Chunk 支持 | 引用支持 | 人工状态 |",
        "| --- | --- | --- | --- |",
    ]
    for row in revised:
        if row["item_human_review"] == "REVIEWED":
            lines.append(
                f"| {row['review_id']} | {row['human_verdict']} | "
                f"{row['human_quote_verdict']} | REVIEWED |"
            )
    lines.extend(
        [
            "",
            "## 累计进度",
            "",
            "- 关系证据：15/202 已人工裁决，187 条待审。",
            "- 已审 Fact：4 SUPPORT、11 INSUFFICIENT；原引用：15 INSUFFICIENT。",
            "- 266 个预测实体、新增 Gold 类型口径、Normalization 仍待审核。",
            "- 异常批次不是随机样本；不将本批比例当作全部预测的 Evidence Precision。",
            "- 正式 Precision / Recall / F1、Evidence Precision 仍未发布。",
            "",
            "## 版本",
            "",
            "本版 evidence_review.jsonl 保留全部 202 条，仅第二批 7 条新增人工裁决。",
            "human_decisions.json 保留两次用户回复、明确条目范围和原展示清单哈希。",
            "原模型预测、Gold 草稿、第一版及其生成脚本均未修改。",
            "",
            "[第二批原文与全部条目](../human-item-adjudication-v1/第二批裁决-状态流转.md)",
            "[第三批 5 条分类责任人关系待裁决](第三批裁决-分类责任人.md)",
            "",
            "```powershell",
            "cd D:\\code\\zhiqing-rag-platform",
            "uv run python -m scripts.record_graph_second_adjudication verify",
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
            "schema_version": "graph-human-item-adjudication-v2",
            "generator_sha256": sha(Path(__file__)),
            "parent_files": {
                name: sha(PARENT / name)
                for name in (
                    "manifest.json",
                    "evidence_review.jsonl",
                    "human_decision.json",
                    "第二批裁决-状态流转.md",
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
    if revised != adjudicate_second_batch(read_jsonl(PARENT / "evidence_review.jsonl")):
        raise ValueError("第二批裁决无法重现，或改变了第一批结果")
    decisions = json.loads((output / "human_decisions.json").read_text(encoding="utf-8"))
    prior = json.loads((PARENT / "human_decision.json").read_text(encoding="utf-8"))
    if (
        len(decisions) != 2
        or decisions[0] != prior
        or decisions[1]["verbatim_reply"] != USER_REPLY
        or decisions[1]["approved_review_ids"] != BATCH_IDS
        or decisions[1]["displayed_batch_sha256"] != sha(PARENT / "第二批裁决-状态流转.md")
    ):
        raise ValueError("用户批准来源或条目范围不一致")
    if read_jsonl(output / "human_verdicts.jsonl") != [
        r for r in revised if r["item_human_review"] == "REVIEWED"
    ]:
        raise ValueError("人工裁决与完整队列不一致")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary != summary_for(revised) or summary["human_reviewed_relations"] != 15:
        raise ValueError("累计人工审核进度不一致")
    if (output / "第三批裁决-分类责任人.md").read_text(encoding="utf-8") != next_batch_document(
        revised
    ):
        raise ValueError("第三批材料无法复核")
    print("第二批裁决校验通过：累计15/202已审，187待审；Fact 4支持/11依据不足，引用15依据不足")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.command == "build" else verify()
