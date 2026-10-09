"""Human approval must add only seven judgments and preserve all previous verdicts."""

from copy import deepcopy

import pytest

from scripts.evaluate_graph_table_replay import read_jsonl
from scripts.record_graph_second_adjudication import (
    BATCH_IDS,
    PARENT,
    adjudicate_second_batch,
    summary_for,
)


def test_exactly_seven_changes_and_first_batch_stays_intact():
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    before = deepcopy(original)
    revised = adjudicate_second_batch(original)
    assert original == before
    assert [b["review_id"] for a, b in zip(original, revised, strict=True) if a != b] == BATCH_IDS
    for old, new in zip(original, revised, strict=True):
        if old["item_human_review"] == "REVIEWED":
            assert old == new
    summary = summary_for(revised)
    assert summary["human_reviewed_relations"] == 15
    assert summary["human_pending_relations"] == 187
    assert summary["human_chunk_support_counts"] == {"SUPPORT": 4, "INSUFFICIENT": 11}
    assert summary["human_quote_support_counts"] == {"INSUFFICIENT": 15}
    assert summary["full_item_human_review"] == "PARTIAL"
    assert summary["entity_review"] == "PENDING"
    assert summary["gold_added_policies_review"] == "PENDING"
    assert summary["normalization_review"] == "PENDING"
    for metric in ("formal_evidence_precision", "final_precision", "final_recall", "final_f1"):
        assert summary[metric] is None


def test_missing_or_duplicate_items_are_rejected():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="缺失"):
        adjudicate_second_batch([r for r in records if r["review_id"] != BATCH_IDS[0]])
    with pytest.raises(ValueError, match="重复"):
        adjudicate_second_batch(records + [records[0]])


def test_changed_relation_cannot_inherit_approval():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    next(r for r in records if r["review_id"] == BATCH_IDS[0])["type"] = "NEXT_STEP"
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_second_batch(records)


def test_approval_cannot_be_reapplied_to_reviewed_version():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_second_batch(adjudicate_second_batch(records))
