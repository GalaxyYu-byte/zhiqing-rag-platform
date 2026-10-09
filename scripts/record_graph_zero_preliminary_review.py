"""Record a preliminary, non-Gold review of four saved extraction responses."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/graph_build/runs/2026-09-29-live/zero-chunk-investigation/replay-20260930"

RELATION_LABELS = {
    "gb-35-00": {
        "SUPPORT": [*range(1, 9), *range(14, 18)],
        "INSUFFICIENT": [*range(9, 14)],
    },
    "gb-35-01": {
        "SUPPORT": [1, 2, 3, 4, 5, 7, 9, 10],
        "INSUFFICIENT": [6, 8, 11, 12, 13, 14],
    },
    "gb-45-00": {
        "SUPPORT": [*range(1, 10), 12],
        "INSUFFICIENT": [10, 11],
    },
    "gb-73-03": {
        "SUPPORT": [1, 2, 3, 5, 9, 10],
        "CONTRADICT": [4],
        "INSUFFICIENT": [6, 7, 8, 11, 12],
    },
}

NOTES = {
    "gb-35-00-R09": "告知资料清单不等于对资料清单执行 OPERATES_ON。",
    "gb-35-00-R10": "告知提交渠道不等于操作该渠道。",
    "gb-35-00-R11": "引文只列出姓名，未直接说明它是个人资料的字段。",
    "gb-35-00-R12": "引文只列出联系方式，未直接说明它是个人资料的字段。",
    "gb-35-00-R13": "引文只列出证件信息，未直接说明它是个人资料的字段。",
    "gb-35-01-R06": "保存事项资料未被明确限定为例外处理的一步。",
    "gb-35-01-R08": "受控存放是一般要求，未明确为例外处理步骤。",
    "gb-35-01-R11": "附件可见人员检查未被明确限定为例外处理步骤。",
    "gb-35-01-R12": "账号开通规则与例外处理的治理关系不明确。",
    "gb-35-01-R13": "转岗访问权规则与例外处理的治理关系不明确。",
    "gb-35-01-R14": "核对对象是邮件、清单和正式文件；目标把三类状态合成了 Material。",
    "gb-45-00-R10": "步骤本身是同一句规则的片段，独立的 GOVERNED_BY 事实缺少依据。",
    "gb-45-00-R11": "步骤本身是同一句规则的片段，独立的 GOVERNED_BY 事实缺少依据。",
    "gb-73-03-R04": "原文称前一项为最后一道核对，后句更像核对内容而非下一步。",
    "gb-73-03-R06": "相邻句子没有声明这两个动作的先后依赖。",
    "gb-73-03-R07": "条件说责任角色未确定时列待确认，并未确认前一步已完成。",
    "gb-73-03-R08": "凭证用于核验事实，未说明分开业务事项这一步操作凭证。",
    "gb-73-03-R11": "两条要求相邻，但未明确后一规则治理列待确认这一步。",
    "gb-73-03-R12": "原文没有待确认事项到自身的状态转移。",
    "gb-35-01-E20": "把经办意见、批准决定和实际执行合为单个 Material，边界和类型均不成立。",
    "gb-73-03-E19": "待确认事项是待处理项目，作为 State 的类型需人工裁定。",
}


def main() -> None:
    queue = [
        json.loads(line)
        for line in (BASE / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    relation_decisions = {
        f"{case_id}-R{index:02d}": label
        for case_id, labels in RELATION_LABELS.items()
        for label, indices in labels.items()
        for index in indices
    }
    keys = {item["key"] for item in queue}
    relation_keys = {item["key"] for item in queue if item["kind"] == "relation"}
    if len(relation_decisions) != len(relation_keys) or set(relation_decisions) != relation_keys:
        raise ValueError("关系裁决与候选队列不一致")
    if set(NOTES) - keys:
        raise ValueError("裁决说明引用了未知候选")
    for item in queue:
        key = item["key"]
        if item["kind"] == "relation":
            item["review_decision"] = relation_decisions[key]
        elif key == "gb-35-01-E20":
            item["review_decision"] = "INVALID_ENTITY"
        elif key == "gb-73-03-E19":
            item["review_decision"] = "TYPE_REVIEW"
        else:
            item["review_decision"] = "SUPPORTED_ENTITY"
        item["review_note"] = NOTES.get(key)
        item["reviewer"] = "Codex preliminary review; human Gold approval pending"
    output = BASE / "assistant_preliminary_review.jsonl"
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in queue),
        encoding="utf-8",
    )
    counts = Counter((item["kind"], item["review_decision"]) for item in queue)
    for (kind, decision), count in sorted(counts.items()):
        print(kind, decision, count)


if __name__ == "__main__":
    main()
