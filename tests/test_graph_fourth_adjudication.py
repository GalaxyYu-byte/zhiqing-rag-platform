"""The user's 'all' applies to the displayed fourteen-item batch, not the whole queue."""

from copy import deepcopy

import pytest

from scripts.evaluate_graph_table_replay import read_jsonl
from scripts.record_graph_fourth_adjudication import (
    BATCH_IDS,
    PARENT,
    adjudicate_fourth_batch,
    summary_for,
)


def test_all_means_exactly_fourteen_and_preserves_previous_twenty():
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    before = deepcopy(original)
    revised = adjudicate_fourth_batch(original)
    assert original == before
    assert [b["review_id"] for a, b in zip(original, revised, strict=True) if a != b] == BATCH_IDS
    assert len(BATCH_IDS) == 14
    for old, new in zip(original, revised, strict=True):
        if old["item_human_review"] == "REVIEWED":
            assert old == new
        elif old["review_id"] not in BATCH_IDS:
            assert new["item_human_review"] == "PENDING"
            assert new["human_verdict"] is None
    summary = summary_for(revised)
    assert summary["human_reviewed_relations"] == 34
    assert summary["human_pending_relations"] == 168
    assert summary["human_chunk_support_counts"] == {"SUPPORT": 4, "INSUFFICIENT": 30}
    assert summary["human_quote_support_counts"] == {"INSUFFICIENT": 34}
    assert summary["full_item_human_review"] == "PARTIAL"
    for status in ("entity_review", "gold_added_policies_review", "normalization_review"):
        assert summary[status] == "PENDING"
    for metric in ("formal_evidence_precision", "final_precision", "final_recall", "final_f1"):
        assert summary[metric] is None


def test_reject_missing_or_duplicate_batch_items():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="缺失"):
        adjudicate_fourth_batch([r for r in records if r["review_id"] != BATCH_IDS[0]])
    with pytest.raises(ValueError, match="重复"):
        adjudicate_fourth_batch(records + [records[0]])


def test_changed_relation_cannot_inherit_approval():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    next(r for r in records if r["review_id"] == BATCH_IDS[0])["type"] = "REQUIRES_MATERIAL"
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_fourth_batch(records)


def test_reapplying_approval_to_reviewed_version_is_rejected():
    records = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_fourth_batch(adjudicate_fourth_batch(records))
