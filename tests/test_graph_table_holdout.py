"""Holdout Gold is selective, row-grounded, and frozen before model predictions."""

import pytest

from scripts.replay_graph_table_holdout import selected_gold


def test_unannotated_chunk_is_not_a_negative_gold_case():
    case = selected_gold({"case_id": "unannotated", "content": "原文"}, [])
    assert case["gold_scope"] == "unannotated_not_negative"
    assert case["review_status"] == "codex_draft_independent_human_review_pending"


def test_shared_role_is_deduplicated_within_one_chunk_and_evidence_is_relocated():
    first = "| 检查 | 经办人 |"
    second = "| 归档 | 经办人 |"
    facts = [{"source_entity": "经办人", "target_entity": step,
              "relation_type": "RESPONSIBLE_FOR", "evidence_quote": quote}
             for step, quote in (("检查", first), ("归档", second))]
    case = selected_gold({"case_id": "table", "content": first + "\n" + second}, facts)
    assert len(case["entities"]) == 3
    assert len(case["relations"]) == 2
    assert case["relations"][0]["source_id"] == case["relations"][1]["source_id"]
    for evidence in case["evidence"]:
        assert case["content"][evidence["start"]:evidence["end"]] == evidence["quote"]


def test_gold_rejects_endpoint_elsewhere_in_chunk_but_not_in_its_evidence_row():
    fact = {"source_entity": "经办人", "target_entity": "核查",
            "relation_type": "RESPONSIBLE_FOR", "evidence_quote": "财务核查。"}
    with pytest.raises(ValueError, match="两端不在同一行"):
        selected_gold({"case_id": "table", "content": "经办人归档。财务核查。"}, [fact])
