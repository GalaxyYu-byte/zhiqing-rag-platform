"""Check the review boundary and metric accounting; no network or graph database."""

from copy import deepcopy

from scripts.evaluate_graph_table_replay import read_jsonl
from scripts.prepare_graph_holdout_review import (
    POLICY,
    RUN,
    analyze_recall,
    evidence_decision,
    expand_gold,
    prepare,
)
from zhiqing_rag.workers.graph_extraction import Extraction, validate_evidence


def test_expanded_gold_is_source_derived_and_keeps_pause_effect_and_relation_gaps():
    original = read_jsonl(POLICY / "cases.jsonl")
    before = deepcopy(original)
    cases = expand_gold(original)
    assert original == before
    assert len(cases) == 13
    assert {c["case_id"].split("-")[1] for c in cases if c["entities"]} == {
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
    }
    for case in cases:
        validate_evidence(
            Extraction.model_validate(
                {
                    "entities": case["entities"],
                    "relations": [
                        {k: v for k, v in r.items() if k != "id"} for r in case["relations"]
                    ],
                }
            ),
            case["content"],
        )
        assert all(case["content"][e["start"] : e["end"]] == e["quote"] for e in case["evidence"])
        if case["case_id"].startswith("holdout-05"):
            assert not case["relations"]  # Do not invent Field→Material schema edges.
        if case["case_id"].startswith("holdout-06"):
            rules = [a for a in case["assertions"] if a.get("required_effect")]
            assert len(rules) == sum(r["type"] == "GOVERNED_BY" for r in case["relations"])
            assert all(a["required_effect"] == "暂停或退回" for a in rules)


def test_relation_matching_partitions_losses_and_does_not_reuse_one_prediction():
    def entity(name, kind="Step"):
        return {
            "id": name,
            "name": name,
            "canonical_name": name,
            "type": kind,
            "evidence_quote": name,
        }

    def relation(index, source, target):
        return {
            "id": f"r{index}",
            "type": "NEXT_STEP",
            "source_id": source,
            "target_id": target,
            "evidence_quote": "a b c d",
            "condition_text": None,
        }

    gold = {
        "case_id": "synthetic",
        "entities": [entity(n) for n in "abcd"],
        "relations": [
            relation(1, "a", "b"),
            relation(2, "a", "b"),
            relation(3, "b", "a"),
            relation(4, "b", "d"),
            relation(5, "c", "d"),
        ],
    }
    extraction = Extraction.model_validate(
        {
            "entities": [entity("a"), entity("b"), entity("c", "Actor")],
            "relations": [{k: v for k, v in relation(1, "a", "b").items() if k != "id"}],
        }
    )
    entities, relations = analyze_recall(gold, extraction)
    assert [r["outcome"] for r in entities] == ["HIT", "HIT", "WRONG_TYPE", "NAME_NOT_EXTRACTED"]
    assert [r["outcome"] for r in relations] == [
        "HIT",
        "RELATION_TYPE_OR_DIRECTION_OR_EDGE_MISSING",
        "RELATION_TYPE_OR_DIRECTION_OR_EDGE_MISSING",
        "TARGET_MISSING",
        "BOTH_ENDPOINTS_MISSING",
    ]
    assert relations[2]["reverse_triple_present"]
    assert sum(r["outcome"] == "HIT" for r in relations) == 1
    assert sum(r["endpoints_present"] for r in relations) == 3


def test_full_review_keeps_human_pending_and_no_formal_metrics():
    cases = read_jsonl(POLICY / "cases.jsonl")
    predictions = {r["case_id"]: r for r in read_jsonl(RUN / "replay.jsonl")}
    items, summary = prepare(cases, predictions)
    assert len(items["entity_review.jsonl"]) == 266
    assert len(items["evidence_review.jsonl"]) == 202
    assert len({r["review_id"] for r in items["evidence_review.jsonl"]}) == 202
    assert all(r["item_human_review"] == "PENDING" for r in items["entity_review.jsonl"])
    assert all(r["human_verdict"] is None for r in items["evidence_review.jsonl"])
    mismatched = [
        r
        for r in items["evidence_review.jsonl"]
        if r["group"] == "quote_mismatch_with_chunk_support"
    ]
    assert len(mismatched) == 4
    assert all(
        r["codex_suggestion"] == "SUPPORT"
        and r["quote_support_suggestion"] == "INSUFFICIENT"
        and r["endpoint_context_rows"]
        for r in mismatched
    )
    assert sum(summary["relation_outcomes"].values()) == summary["gold_relations"]
    assert sum(summary["entity_outcomes"].values()) == summary["gold_entities"]
    assert summary["annotation_timing"] == "post_prediction"
    for metric in (
        "final_precision",
        "final_recall",
        "final_f1",
        "formal_evidence_precision",
        "semantic_condition_recall",
        "merge_error_rate",
        "duplicate_entity_rate",
    ):
        assert summary[metric] is None


def test_evidence_support_is_separate_from_primary_entity_boundary_and_pending_effect():
    # Old content descriptions can still have evidence support while missing approved main names.
    assert evidence_decision("holdout-03-01", 1)[0] == "SUPPORT"
    assert evidence_decision("holdout-01-02", 33)[0] == "INSUFFICIENT"
    # These actor/action pairs ARE supported elsewhere in the chunk; only citation is wrong.
    assert evidence_decision("holdout-01-02", 37) == (
        "SUPPORT",
        "本 Chunk 的其他行支持该角色与动作，但模型引用来自未确认权限行，引用错位。",
        "quote_mismatch_with_chunk_support",
    )
    assert evidence_decision("holdout-02-01", 1)[0] == "INSUFFICIENT"
    assert evidence_decision("holdout-06-01", 3)[0] == "REVIEW_REQUIRED"
    assert evidence_decision("holdout-06-01", 2)[0] == "SUPPORT"
