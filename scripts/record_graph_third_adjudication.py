"""Record only the five explicitly approved third-batch evidence decisions."""

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.prepare_graph_holdout_review import OUTPUT as DRAFT
from scripts.record_graph_item_adjudication import summarize
from scripts.record_graph_second_adjudication import OUTPUT as PARENT
from scripts.record_graph_second_adjudication import verify as verify_parent

OUTPUT = PILOT / "human-item-adjudication-v3"
USER_REPLY = "第三批 5 条均判 INSUFFICIENT"
BATCH_IDS = [f"holdout-04-02:relation:{index}" for index in range(1, 6)]


def adjudicate_third_batch(records):
    ids = [r["review_id"] for r in records]
    if len(ids) != len(set(ids)) or not set(BATCH_IDS).issubset(ids):
        raise ValueError("第三批条目缺失或审核 ID 重复")
    revised = deepcopy(records)
    for row in revised:
        if row["review_id"] not in BATCH_IDS:
            continue
        if (
            row["case_id"] != "holdout-04-02"
            or row["type"] != "RESPONSIBLE_FOR"
            or row["source_type"] != "Actor"
            or row["group"] != "category_as_actor"
            or row["codex_suggestion"] != "INSUFFICIENT"
            or row["quote_support_suggestion"] != "INSUFFICIENT"
            or row["item_human_review"] != "PENDING"
            or row["human_verdict"] is not None
        ):
            raise ValueError("条目与用户已批准的第三批不一致")
        row.update(
            item_human_review="REVIEWED",
            human_verdict="INSUFFICIENT",
            human_quote_verdict="INSUFFICIENT",
            human_reviewer="human_user_in_current_chat",
            human_review_basis="explicit_reply_approving_all_five_displayed_batch_items",
            human_batch_id="third_batch_category_as_actor",
        )
    return revised


def summary_for(records):
    summary = summarize(records)
    summary["scope"] = "twenty explicit evidence decisions; entities and new Gold policies pending"
    return summary


def next_batch_document(records):
    selected = [r for r in records if r["group"] == "retention_not_production"]
    if len(selected) != 14 or any(r["item_human_review"] != "PENDING" for r in selected):
        raise ValueError("第四批待审范围变化")
    cases = {c["case_id"]: c for c in read_jsonl(DRAFT / "cases.jsonl")}
    lines = [
        "# 第四批待裁决：14 条把保留内容推断为生成结果的关系",
        "",
        "人工状态：PENDING。本文件展示本批全部 14 条，不是抽样示例。",
        "",
        "表格的列名是“需要留下的内容”，并非“该动作生成的结果”。",
        "有些内容可能来源于客户或既有记录；同一行出现动作与内容不自动证明生成关系。",
        "请结合每条具体原文判断；可全部同意，也可指出某条不同意见。",
    ]
    for case_id in dict.fromkeys(r["case_id"] for r in selected):
        lines.extend(
            [
                "",
                f"## {case_id} 原始 Chunk",
                "",
                cases[case_id]["content"],
                "",
                "| review_id | 预测 Fact | 模型原引用 | Codex 建议 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in selected:
            if row["case_id"] != case_id:
                continue
            quote = row["evidence_quote"].replace("|", "\\|").replace("\n", "<br>")
            lines.append(
                f"| {row['review_id']} | {row['source']} --{row['type']}--> "
                f"{row['target']} | {quote} | INSUFFICIENT |"
            )
    lines.extend(
        [
            "",
            "建议理由：当前原文规定保留这些内容，没有明确建立这些具体动作的生成关系。",
            "建议 INSUFFICIENT，不据此断言生成关系被原文否定。",
            "",
            "若认可本批全部 14 条，可回复：第四批 14 条均判 INSUFFICIENT。",
            "此清单只准备待裁决材料；尚未将第四批写入人工裁决。",
            "",
        ]
    )
    return "\n".join(lines)


def build(output=OUTPUT):
    if output.exists():
        raise FileExistsError("第三批裁决版本已存在，请使用 verify；不覆盖历史记录")
    verify_parent()
    revised = adjudicate_third_batch(read_jsonl(PARENT / "evidence_review.jsonl"))
    summary = summary_for(revised)
    decisions = json.loads((PARENT / "human_decisions.json").read_text(encoding="utf-8"))
    decisions.append(
        {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "reviewer": "human_user_in_current_chat",
            "verbatim_reply": USER_REPLY,
            "approval_basis": "direct_user_reply_to_the_complete_displayed_third_batch",
            "approved_review_ids": BATCH_IDS,
            "displayed_batch_sha256": sha(PARENT / "第三批裁决-分类责任人.md"),
            "scope": "chunk support and quote support for these five relation predictions only",
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
    (output / "第四批裁决-保留与生成.md").write_text(next_batch_document(revised), encoding="utf-8")
    lines = [
        "# 第一至三批人工裁决记录",
        "",
        f"第三批用户回复：**{USER_REPLY}**。",
        "",
        "前两批 15 条完整保留；本版本只新增第三批 5 条具体关系的裁决。",
        "",
        "| 第三批 review_id | Chunk 支持 | 引用支持 | 人工状态 |",
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
            "- 关系证据：20/202 已人工裁决，182 条待审。",
            "- 已审 Fact：4 SUPPORT、16 INSUFFICIENT；原引用：20 INSUFFICIENT。",
            "- 266 个预测实体、新增 Gold 类型口径、Normalization 仍待审核。",
            "- 批次专门选取异常关系，不以已审批次的比例推算整体 Evidence Precision。",
            "- 正式 Precision / Recall / F1、Evidence Precision 仍未发布。",
            "",
            "## 版本与证据",
            "",
            "evidence_review.jsonl 保留全部 202 条；仅指定 5 条新增人工状态。",
            "human_verdicts.jsonl 保存累计 20 条人工裁决。",
            "human_decisions.json 保存三次原回复和条目范围。",
            "manifest.json 绑定父级文件、草稿原文、脚本和所有输出哈希。",
            "原模型预测、Gold 草稿、前两版裁决及生成脚本保持原样。",
            "",
            "[第三批原文及全部条目](../human-item-adjudication-v2/第三批裁决-分类责任人.md)",
            "[第四批 14 条保留与生成关系待裁决](第四批裁决-保留与生成.md)",
            "",
            "```powershell",
            "cd D:\\code\\zhiqing-rag-platform",
            "uv run python -m scripts.record_graph_third_adjudication verify",
            "```",
            "",
            "本次仅记录本地人工裁决；未调用模型、修改生产抽取或写入 PostgreSQL/Neo4j。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "schema_version": "graph-human-item-adjudication-v3",
            "generator_sha256": sha(Path(__file__)),
            "parent_files": {
                name: sha(PARENT / name)
                for name in (
                    "manifest.json",
                    "evidence_review.jsonl",
                    "human_decisions.json",
                    "第三批裁决-分类责任人.md",
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
    if revised != adjudicate_third_batch(read_jsonl(PARENT / "evidence_review.jsonl")):
        raise ValueError("第三批裁决无法重现，或改变了前两批结果")
    decisions = json.loads((output / "human_decisions.json").read_text(encoding="utf-8"))
    prior = json.loads((PARENT / "human_decisions.json").read_text(encoding="utf-8"))
    if (
        len(decisions) != 3
        or decisions[:-1] != prior
        or decisions[-1]["verbatim_reply"] != USER_REPLY
        or decisions[-1]["approved_review_ids"] != BATCH_IDS
        or decisions[-1]["displayed_batch_sha256"] != sha(PARENT / "第三批裁决-分类责任人.md")
    ):
        raise ValueError("批准来源或条目范围不一致")
    if read_jsonl(output / "human_verdicts.jsonl") != [
        r for r in revised if r["item_human_review"] == "REVIEWED"
    ]:
        raise ValueError("人工裁决与完整队列不一致")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary != summary_for(revised) or summary["human_reviewed_relations"] != 20:
        raise ValueError("累计人工审核进度不一致")
    if (output / "第四批裁决-保留与生成.md").read_text(encoding="utf-8") != next_batch_document(
        revised
    ):
        raise ValueError("第四批材料无法重现")
    print("第三批裁决校验通过：累计20/202已审，182待审；Fact 4支持/16依据不足，引用20依据不足")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.command == "build" else verify()
