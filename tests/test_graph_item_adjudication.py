"""Verify human approval only applies to the eight explicitly displayed decisions."""

from copy import deepcopy

import pytest

from scripts.evaluate_graph_table_replay import read_jsonl
from scripts.record_graph_item_adjudication import (
    BATCH_IDS,
    PARENT,
    adjudicate_first_batch,
    summarize,
)


def test_only_eight_items_change_and_original_stays_frozen():
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    before = deepcopy(original)
    revised = adjudicate_first_batch(original)
    assert original == before
    changed = [r["review_id"] for a, r in zip(original, revised, strict=True) if a != r]
    assert changed == BATCH_IDS
    assert all(r["human_verdict"] is None for r in revised if r["review_id"] not in BATCH_IDS)
    summary = summarize(revised)
    assert summary["human_reviewed_relations"] == 8
    assert summary["human_pending_relations"] == 194
    assert summary["human_chunk_support_counts"] == {"SUPPORT": 4, "INSUFFICIENT": 4}
    assert summary["human_quote_support_counts"] == {"INSUFFICIENT": 8}
    assert summary["full_item_human_review"] == "PARTIAL"
    assert summary["entity_review"] == "PENDING"
    assert summary["gold_added_policies_review"] == "PENDING"
    assert summary["formal_evidence_precision"] is None
    assert summary["final_f1"] is None


def test_reject_missing_or_duplicate_items():
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="缺失"):
        adjudicate_first_batch([r for r in original if r["review_id"] != BATCH_IDS[0]])
    with pytest.raises(ValueError, match="重复"):
        adjudicate_first_batch(original + [original[0]])


def test_changed_suggestion_cannot_inherit_approval():
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    target = next(r for r in original if r["review_id"] == BATCH_IDS[4])
    target["codex_suggestion"] = "INSUFFICIENT"
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_first_batch(original)


def test_reapplying_approval_to_reviewed_version_is_rejected():
    original = read_jsonl(PARENT / "evidence_review.jsonl")
    with pytest.raises(ValueError, match="不一致"):
        adjudicate_first_batch(adjudicate_first_batch(original))
