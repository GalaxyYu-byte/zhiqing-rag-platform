"""Check third-batch approval scope, preserved history and incomplete metric status."""

from copy import deepcopy

import pytest

from scripts.evaluate_graph_table_replay import read_jsonl
from scripts.record_graph_third_adjudication import (
    BATCH_IDS,
    PARENT,
    adjudicate_third_batch,
    summary_for,
)


def test_only_five_items_change_and_previous_fifteen_stay_intact():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    before = deepcopy(records)
    revised = adjudicate_third_batch(records)
    assert records == before
    assert [b["review_id"] for a, b in zip(records, revised, strict=True) if a != b] == BATCH_IDS
    assert sum(r["item_human_review"] == "REVIEWED" for r in records) == 15
    for old, new in zip(records, revised, strict=True):
        if old["item_human_review"] == "REVIEWED":
            assert new == old
    summary = summary_for(revised)
    assert summary["human_reviewed_relations"] == 20
    assert summary["human_pending_relations"] == 182
    assert summary["human_chunk_support_counts"] == {"SUPPORT": 4, "INSUFFICIENT": 16}
    assert summary["human_quote_support_counts"] == {"INSUFFICIENT": 20}
    assert summary["full_item_human_review"] == "PARTIAL"
    for status in ("entity_review", "gold_added_policies_review", "normalization_review"):
        assert summary[status] == "PENDING"
    for metric in ("formal_evidence_precision", "final_precision", "final_recall", "final_f1"):
        assert summary[metric] is None


def test_missing_or_duplicate_batch_items_are_rejected():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="缺失"):
        adjudicate_third_batch([r for r in records if r["review_id"] != BATCH_IDS[0]])
    with pytest.raises(ValueError, match="重复"):
        adjudicate_third_batch(records + [records[0]])


def test_changed_source_type_cannot_inherit_approval():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    next(r for r in records if r["review_id"] == BATCH_IDS[0])["source_type"] = "Process"
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_third_batch(records)


def test_approval_cannot_be_applied_twice():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_third_batch(adjudicate_third_batch(records))
