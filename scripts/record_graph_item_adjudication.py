"""Record the user's explicit first-batch evidence adjudication in a new local version."""

import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.prepare_graph_holdout_review import OUTPUT as PARENT
from scripts.prepare_graph_holdout_review import verify as verify_parent

OUTPUT = PILOT / "human-item-adjudication-v1"
USER_REPLY = "第一批按清单裁决"
BATCH_IDS = [f"holdout-01-02:relation:{index}" for index in range(33, 41)]


def adjudicate_first_batch(records):
    """Approve exactly the eight displayed decisions, leaving all other records untouched."""
    ids = [r["review_id"] for r in records]
    if len(ids) != len(set(ids)) or not set(BATCH_IDS).issubset(ids):
        raise ValueError("第一批条目缺失或审核 ID 重复")
    revised = deepcopy(records)
    for row in revised:
        if row["review_id"] not in BATCH_IDS:
            continue
        index = row["relation_index"]
        expected = "INSUFFICIENT" if index <= 36 else "SUPPORT"
        if (
            row["case_id"] != "holdout-01-02"
            or row["codex_suggestion"] != expected
            or row["quote_support_suggestion"] != "INSUFFICIENT"
            or row["item_human_review"] != "PENDING"
            or row["human_verdict"] is not None
        ):
            raise ValueError("条目与用户看到的第一批裁决建议不一致")
        row.update(
            item_human_review="REVIEWED",
            human_verdict=expected,
            human_quote_verdict="INSUFFICIENT",
            human_reviewer="human_user_in_current_chat",
            human_review_basis="explicit_reply_approving_all_eight_displayed_batch_items",
            human_batch_id="first_batch_cross_row_binding",
        )
    return revised


def summarize(records):
    reviewed = [r for r in records if r["item_human_review"] == "REVIEWED"]
    return {
        "prediction_relation_count": len(records),
        "human_reviewed_relations": len(reviewed),
        "human_pending_relations": len(records) - len(reviewed),
        "human_chunk_support_counts": dict(Counter(r["human_verdict"] for r in reviewed)),
        "human_quote_support_counts": dict(Counter(r["human_quote_verdict"] for r in reviewed)),
        "reviewed_ids": [r["review_id"] for r in reviewed],
        "entity_review": "PENDING",
        "gold_added_policies_review": "PENDING",
        "normalization_review": "PENDING",
        "full_item_human_review": "PARTIAL",
        "final_precision": None,
        "final_recall": None,
        "final_f1": None,
        "formal_evidence_precision": None,
        "scope": (
            "eight explicit evidence decisions only; not entity, Gold or total quality approval"
        ),
    }


def build(output=OUTPUT):
    if output.exists():
        raise FileExistsError("人工裁决版本已存在，请使用 verify；不覆盖历史记录")
    verify_parent()
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    revised = adjudicate_first_batch(original)
    summary = summarize(revised)
    output.mkdir(parents=True)
    write_jsonl(output / "evidence_review.jsonl", revised)
    write_jsonl(
        output / "human_verdicts.jsonl", [r for r in revised if r["review_id"] in BATCH_IDS]
    )
    write_json(
        output / "human_decision.json",
        {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "reviewer": "human_user_in_current_chat",
            "verbatim_reply": USER_REPLY,
            "approval_basis": "direct_user_reply_to_the_complete_displayed_first_batch",
            "approved_review_ids": BATCH_IDS,
            "displayed_batch_sha256": sha(PARENT / "第一批裁决-跨行错配.md"),
            "scope": "chunk support and quote support for these eight relation predictions only",
            "does_not_approve": [
                "other relation predictions",
                "entities",
                "new Gold policies",
                "normalization",
                "final quality acceptance",
            ],
        },
    )
    write_json(output / "summary.json", summary)
    next_batch = [
        r for r in original if r["case_id"] == "holdout-02-01" and r["group"] == "ordered_states"
    ]
    if len(next_batch) != 7:
        raise ValueError("下一批状态流转候选范围发生变化")
    state_case = next(
        c for c in read_jsonl(PARENT / "cases.jsonl") if c["case_id"] == "holdout-02-01"
    )
    next_lines = [
        "# 第二批待裁决：7 条状态流转",
        "",
        "人工状态：PENDING。此前已批准规则4：表格行排列不足以证明直接状态流转。",
        "下面列出本批全部 7 条具体预测；当前第一批的批准不覆盖本批。",
        "",
        "## 原始 Chunk",
        "",
        state_case["content"],
        "",
        "## 待裁决的预测",
        "",
        "| review_id | 预测 Fact | 模型原引用 | Codex 建议 |",
        "| --- | --- | --- | --- |",
    ]
    for row in next_batch:
        quote = row["evidence_quote"].replace("|", "\\|").replace("\n", "<br>")
        next_lines.append(
            f"| {row['review_id']} | {row['source']} --{row['type']}--> "
            f"{row['target']} | {quote} | INSUFFICIENT |"
        )
    next_lines.extend(
        [
            "",
            "依据：表格规定各状态的进入条件、动作和离开前核对要求，",
            "没有明确说明这些相邻状态之间存在直接转移。没有转移证据不等于证明禁止转移，",
            "因此建议 INSUFFICIENT，不判 CONTRADICT。",
            "",
            "若核对原文后认可本批全部 7 条，可回复：第二批 7 条均判 INSUFFICIENT。",
            "当前仅准备材料，尚未写入本批人工裁决。",
            "",
        ]
    )
    (output / "第二批裁决-状态流转.md").write_text("\n".join(next_lines), encoding="utf-8")
    lines = [
        "# 第一批逐条人工裁决：已确认",
        "",
        f"用户在当前对话明确回复：**{USER_REPLY}**。",
        "",
        "本版本记录完整展示的 8 条具体条目，不将批准扩大到其他预测或新增 Gold 口径。",
        "",
        "| 条目 | Chunk 对 Fact 的支持 | 模型引用支持 | 人工状态 |",
        "| --- | --- | --- | --- |",
    ]
    for row in revised:
        if row["review_id"] in BATCH_IDS:
            lines.append(
                f"| {row['review_id']} | {row['human_verdict']} | "
                f"{row['human_quote_verdict']} | REVIEWED |"
            )
    lines.extend(
        [
            "",
            "## 审核进度",
            "",
            "- 关系证据逐条人工裁决：8/202；剩余 194 条待审。",
            "- 已审 Fact：4 条 SUPPORT、4 条 INSUFFICIENT；引用：8 条 INSUFFICIENT。",
            "- 266 个预测实体、新增 Gold 类型口径、Normalization 尚待审核。",
            "- 本批是专门选择的异常条目，不将 4/8 当作整批预测的 Evidence Precision。",
            "- 正式 Precision / Recall / F1 和 Evidence Precision 仍未发布。",
            "",
            "## 版本与证据",
            "",
            "原冻结文档、Gold 草稿、202 条模型预测和父级初审记录均保留。",
            "evidence_review.jsonl 保留全部 202 条，仅对指定 8 条记录人工裁决。",
            "其他条目仍为 PENDING。",
            "human_decision.json 保存原回复、批准范围和所展示清单的哈希。",
            "manifest.json 绑定父级文件、生成脚本和新裁决文件，可重现仅 8 条的状态变化。",
            "",
            "[查看原文和第一批全部条目](../six-document-review-draft-v1/第一批裁决-跨行错配.md)",
            "[完整初审清单](../six-document-review-draft-v1/人工裁决清单.md)",
            "[第二批 7 条状态流转待裁决](第二批裁决-状态流转.md)",
            "",
            "```powershell",
            "cd D:\\code\\zhiqing-rag-platform",
            "uv run python -m scripts.record_graph_item_adjudication verify",
            "```",
            "",
            "本次仅记录本地人工裁决，没有调用模型、修改生产抽取或写入 PostgreSQL/Neo4j。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(
        output / "manifest.json",
        {
            "schema_version": "graph-human-item-adjudication-v1",
            "generator_sha256": sha(Path(__file__)),
            "parent_files": {
                name: sha(PARENT / name)
                for name in (
                    "manifest.json",
                    "cases.jsonl",
                    "evidence_review.jsonl",
                    "entity_review.jsonl",
                    "normalization_review.jsonl",
                    "第一批裁决-跨行错配.md",
                )
            },
            "files": {p.name: sha(p) for p in output.iterdir()},
            "full_item_human_review": "PARTIAL",
        },
    )
    verify(output)


def verify(output=OUTPUT):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if sha(Path(__file__)) != manifest["generator_sha256"]:
        raise ValueError("裁决生成脚本已改变")
    for name, expected in manifest["parent_files"].items():
        if sha(PARENT / name) != expected:
            raise ValueError(f"父级审核输入改变: {name}")
    for name, expected in manifest["files"].items():
        if sha(output / name) != expected:
            raise ValueError(f"人工裁决文件改变: {name}")
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    revised = read_jsonl(output / "evidence_review.jsonl")
    if revised != adjudicate_first_batch(original):
        raise ValueError("人工裁决范围或内容无法重现")
    decision = json.loads((output / "human_decision.json").read_text(encoding="utf-8"))
    if (
        decision["verbatim_reply"] != USER_REPLY
        or decision["approved_review_ids"] != BATCH_IDS
        or decision["displayed_batch_sha256"] != sha(PARENT / "第一批裁决-跨行错配.md")
    ):
        raise ValueError("批准来源或批次范围不一致")
    if read_jsonl(output / "human_verdicts.jsonl") != [
        r for r in revised if r["review_id"] in BATCH_IDS
    ]:
        raise ValueError("单项人工裁决与完整队列不一致")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary != summarize(revised) or summary["human_reviewed_relations"] != 8:
        raise ValueError("人工审核进度无法重现")
    print("人工裁决校验通过：8/202 已审核，194 待审；Fact 4支持/4依据不足，引用8依据不足")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.command == "build" else verify()
