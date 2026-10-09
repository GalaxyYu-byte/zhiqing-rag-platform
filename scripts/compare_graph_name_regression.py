"""Compare name-feedback regression with immutable pre-fix holdout output."""

import json
from collections import Counter

from scripts.evaluate_graph_table_replay import PILOT, read_jsonl, sha, write_json, write_jsonl
from scripts.replay_graph_table_holdout import FROZEN, verify
from zhiqing_rag.workers.graph_extraction import (
    Extraction,
    _entity_name_feedback,
    normalize_name,
    validate_evidence,
)

BASELINE = PILOT / "holdout-replay-20260930"
CURRENT = PILOT / "name-feedback-regression-20260930"


def main():
    cases, _ = verify()
    by_case = {case["case_id"]: case for case in cases}
    previous_meta = json.loads((BASELINE / "metadata.json").read_text(encoding="utf-8"))
    current_meta = json.loads((CURRENT / "metadata.json").read_text(encoding="utf-8"))
    for field in ("frozen_manifest_sha256", "system_prompt_sha256", "model", "authorized_case_ids"):
        if previous_meta[field] != current_meta[field]:
            raise ValueError(f"回归比较参数发生变化: {field}")
    if sha(CURRENT / "extractor_source.py") != current_meta["extractor_source_sha256"]:
        raise ValueError("v2 源码快照损坏")
    if sha(CURRENT / "system_prompt.txt") != current_meta["system_prompt_sha256"]:
        raise ValueError("回归初始提示词快照损坏")
    previous = {row["case_id"]: row for row in read_jsonl(BASELINE / "replay.jsonl")}
    current = {row["case_id"]: row for row in read_jsonl(CURRENT / "replay.jsonl")}
    if set(previous) != set(current) or set(current) != set(by_case):
        raise ValueError("回归输入集合不一致")
    transitions = []
    errors = []
    recovery_counts = Counter()
    for case_id, case in by_case.items():
        before, after = previous[case_id], current[case_id]
        if (before["content_sha256"] != case["content_sha256"]
                or after["content_sha256"] != case["content_sha256"]):
            raise ValueError("回归输入正文变化")
        extraction = Extraction.model_validate(after.get("extraction", {"entities": [],
                                                                       "relations": []}))
        validate_evidence(extraction, case["content"])
        recovered = (before["status"] == "ERROR" and after["status"] == "VALID_EXTRACTION"
                     and bool(extraction.entities or extraction.relations))
        recovery_counts["nonempty_recovered_chunks"] += recovered
        recovery_counts["new_failed_chunks"] += (
            before["status"] == "VALID_EXTRACTION" and after["status"] == "ERROR")
        transitions.append({"case_id": case_id, "previous_status": before["status"],
                            "current_status": after["status"], "nonempty_recovery": recovered,
                            "previous_attempts": len(before["attempts"]),
                            "current_attempts": len(after["attempts"]),
                            "current_entities": len(extraction.entities),
                            "current_relations": len(extraction.relations)})
        if not recovered:
            continue
        first = Extraction.model_validate_json(after["attempts"][0]["raw_output"])
        for index, entity in enumerate(first.entities):
            fields = []
            if entity.name not in case["content"]:
                fields.append("name")
            if entity.canonical_name not in case["content"] and normalize_name(
                entity.canonical_name
            ) != normalize_name(entity.name):
                fields.append("canonical_name")
            if fields:
                errors.append({"case_id": case_id, "attempt": 1, "entity_index": index,
                               "invalid_fields": fields, "entity": entity.model_dump()})
        # This is reconstructed from the saved first response and v2's exact helper.
        # It is not a capture of the live outbound API request.
        errors.append({"case_id": case_id, "reconstructed_retry_feedback":
                       _entity_name_feedback(first, case["content"])})
    before_summary = json.loads((BASELINE / "summary.json").read_text(encoding="utf-8"))
    after_summary = json.loads((CURRENT / "summary.json").read_text(encoding="utf-8"))
    write_jsonl(CURRENT / "status_comparison.jsonl", transitions)
    write_jsonl(CURRENT / "name_recovery_diagnostics.jsonl", errors)
    write_json(CURRENT / "comparison.json", {
        "baseline_replay_sha256": sha(BASELINE / "replay.jsonl"),
        "current_replay_sha256": sha(CURRENT / "replay.jsonl"),
        "frozen_cases_sha256": sha(FROZEN / "cases.jsonl"),
        "same_input_and_initial_prompt": True,
        "changed_component": "v2 entity-name retry feedback; frozen initial prompt retained",
        "counts": dict(recovery_counts),
        "baseline_counts": before_summary["counts"], "current_counts": after_summary["counts"],
        "attribution_limit": "one rerun; successful first responses also varied",
        "final_precision": None, "final_f1": None, "independent_human_review": "PENDING",
    })
    print(json.dumps(dict(recovery_counts), ensure_ascii=False))


if __name__ == "__main__":
    main()
