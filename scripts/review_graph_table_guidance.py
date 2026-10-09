"""Record explicit Codex preliminary evidence decisions for one frozen pilot replay."""

from collections import Counter

from scripts.evaluate_graph_table_replay import (
    PILOT,
    TARGETS,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
)

RUN = PILOT / "replay-20260930-table-guidance"


def main():
    cases = {case["case_id"]: case for case in read_jsonl(TARGETS / "cases.jsonl")}
    reviewed = []
    counts = Counter()
    for row in read_jsonl(RUN / "replay.jsonl"):
        case = cases[row["case_id"]]
        entities = {e["id"]: e for e in row["extraction"]["entities"]}
        header = case["content"].splitlines()[0]
        for index, relation in enumerate(row["extraction"]["relations"], 1):
            source = entities[relation["source_id"]]
            target = entities[relation["target_id"]]
            unsupported = row["case_id"] == "v2-83-02" and index in (14, 15)
            if unsupported:
                if source["name"] not in ("资料", "岗位资料") or "本表列" not in target["name"]:
                    raise ValueError("固定裁决对应的候选发生变化")
                verdict = "INSUFFICIENT"
                reason = "原文限制的是本表存放内容，未说明岗位资料或一般资料受该限制。"
            else:
                verdict = "SUPPORT"
                reason = ("本行与本 Chunk 列头明确说明角色的动作职责。" if
                          relation["type"] == "RESPONSIBLE_FOR" else
                          "本行的资料类型与保管关注点列明确给出对象和约束。")
            rows = [line for line in case["content"].splitlines()[2:]
                    if relation["evidence_quote"] in line and target["name"] in line]
            if len(rows) != 1:
                raise ValueError("候选证据无法定位到唯一表格行")
            if not unsupported and source["name"] != "员工" and source["name"] not in rows[0]:
                raise ValueError("候选主体未出现在行内，需重新裁决")
            if source["name"] == "员工" and "员工应做" not in header:
                raise ValueError("员工主体缺少当前 Chunk 列头依据")
            counts[verdict] += 1
            reviewed.append({
                "case_id": case["case_id"], "relation_index": index,
                "source": source["name"], "type": relation["type"],
                "target": target["name"], "condition_text": relation["condition_text"],
                "model_quote": relation["evidence_quote"],
                "support_context": {"header": header, "row": rows[0]},
                "evidence_verdict": verdict, "reason": reason,
                "reviewer": "Codex", "independent_human_review": "PENDING",
            })
    if counts != Counter(SUPPORT=47, INSUFFICIENT=2):
        raise ValueError("候选计数与本次逐条初审不一致")
    write_jsonl(RUN / "codex_evidence_review.jsonl", reviewed)
    write_json(RUN / "codex_review_manifest.json", {
        "replay_sha256": sha(RUN / "replay.jsonl"),
        "cases_sha256": sha(TARGETS / "cases.jsonl"),
        "review_sha256": sha(RUN / "codex_evidence_review.jsonl"),
        "reviewer": "Codex", "independent_human_review": "PENDING",
        "counts": dict(counts), "formal_evidence_precision": None,
    })
    print("Codex 证据初审：47 SUPPORT，2 INSUFFICIENT；独立人工裁决待完成")


if __name__ == "__main__":
    main()
