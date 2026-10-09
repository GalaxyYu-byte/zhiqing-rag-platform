"""Regression checks for strict fact matching and projection integrity."""

from scripts.evaluate_graph_quality import (
    audit_projection,
    compare_gold,
    conditional_relation_review,
    evidence_support_review,
    normalization_review,
)


def test_directed_relation_does_not_match_inverse_or_publish_precision():
    left = {"key": "left", "type": "Step", "name": "步骤 A"}
    right = {"key": "right", "type": "Step", "name": "步骤 B"}
    evidence = {"key": "evidence", "chunk_id": 1, "quote": "步骤 A 依赖步骤 B"}
    snapshot = {
        "neo4j": {
            "evidence": [evidence],
            "mentions": [
                {"entity": left, "evidence_key": "evidence"},
                {"entity": right, "evidence_key": "evidence"},
            ],
            "assertions": [
                {
                    "assertion": {"key": "inverse", "predicate": "DEPENDS_ON"},
                    "subjects": [right],
                    "objects": [left],
                    "evidences": [evidence],
                }
            ],
        }
    }
    cases = [
        {
            "case_id": "case-1",
            "source_variant": "frozen",
            "chunk_id": 1,
            "entities": [
                {"id": "a", "type": "Step", "name": "步骤 A"},
                {"id": "b", "type": "Step", "name": "步骤 B"},
            ],
            "relations": [
                {
                    "source_id": "a",
                    "target_id": "b",
                    "type": "DEPENDS_ON",
                    "condition_text": None,
                    "evidence_quote": "步骤 A 依赖步骤 B",
                }
            ],
            "hard_negatives": [],
            "coverage_tags": ["cross_sentence"],
        }
    ]

    quality, review = compare_gold(snapshot, cases)

    assert quality["entity"]["targeted_recall"] == 1
    assert quality["relation"]["exact_directed_triple_match_count"] == 0
    assert quality["relation"]["direction_error_candidates"] == 1
    assert quality["relation"]["precision"] is None
    assert review[0]["reasons"] == ["not_in_targeted_gold", "possible_direction_error"]

    snapshot["neo4j"]["assertions"][0]["subjects"] = [left]
    snapshot["neo4j"]["assertions"][0]["objects"] = [right]
    snapshot["neo4j"]["assertions"][0]["assertion"]["condition_text"] = "仅当获批"
    quality, _ = compare_gold(snapshot, cases)
    assert quality["relation"]["exact_directed_triple_match_count"] == 1
    assert quality["relation"]["condition_mismatch_on_matched_relation"] == 1
    assert quality["relation"]["directed_triple_and_condition_match_count"] == 0

    conditional, outcomes = conditional_relation_review(snapshot, cases)
    assert conditional["both_gold_endpoints_found_count"] == 1
    assert conditional["triple_match_count_given_endpoints"] == 1
    assert conditional["triple_and_condition_match_count_given_endpoints"] == 0
    assert outcomes[0]["condition_status"] == "MISMATCH"

    support, evidence_rows = evidence_support_review(snapshot, cases)
    assert support["support_from_exact_verified_gold"] == 0
    assert evidence_rows[0]["verdict"] is None

    snapshot["neo4j"]["assertions"][0]["assertion"]["condition_text"] = None
    support, evidence_rows = evidence_support_review(snapshot, cases)
    assert support["support_from_exact_verified_gold"] == 1
    assert evidence_rows[0]["verdict"] == "SUPPORT"

    snapshot["neo4j"]["mentions"].pop()
    conditional, outcomes = conditional_relation_review(snapshot, cases)
    assert conditional["both_gold_endpoints_found_count"] == 0
    assert outcomes[0]["triple_status"] == "INELIGIBLE"


def test_projection_audit_detects_orphan_fact_and_entity_count_loss():
    snapshot = {
        "graph_namespace": "test",
        "postgres": {
            "chunks": [{"extraction_cache": None}],
            "documents": [
                {
                    "active_generation_id": 1,
                    "document_id": 2,
                    "entity_count": 1,
                    "relation_count": 1,
                    "projection_status": "READY",
                    "generation_status": "READY",
                    "desired_revision_id": 3,
                    "generation_revision_id": 3,
                    "graph_namespace": "test",
                    "status": "PUBLISHED",
                }
            ],
        },
        "neo4j": {
            "evidence": [],
            "mentions": [],
            "assertions": [
                {
                    "assertion": {"key": "fact", "generation_id": 1},
                    "subjects": [],
                    "objects": [],
                    "evidences": [],
                }
            ],
            "orphan_entity_keys": [],
            "duplicate_entity_keys": [],
            "other_generation_evidence": [],
        },
    }

    audit = audit_projection(snapshot, [])

    assert audit["issue_counts"]["FACT_WITHOUT_EVIDENCE"] == 1
    assert audit["issue_counts"]["ORPHAN_OR_AMBIGUOUS_FACT_ENDPOINT"] == 1
    assert audit["issue_counts"]["ENTITY_COUNT_MISMATCH"] == 1
    assert audit["raw_to_neo4j_status"] == "unverified_missing_raw_extractions"


def test_normalization_review_keeps_merge_and_split_as_candidates():
    common = {"tenant_id": 1, "kb_id": 2, "type": "Step", "normalized_name": "审批"}
    first = {**common, "key": "first", "name": "审批"}
    second = {**common, "key": "second", "name": "审批"}
    evidences = [{"key": f"e{i}", "chunk_id": i, "quote": f"审批记录 {i}"} for i in (1, 2, 3)]
    snapshot = {
        "neo4j": {
            "evidence": evidences,
            "mentions": [
                {"entity": first, "evidence_key": "e1"},
                {"entity": first, "evidence_key": "e2"},
                {"entity": second, "evidence_key": "e3"},
            ],
        }
    }

    summary, reviews = normalization_review(snapshot, [])

    assert summary["possible_wrong_merge_pairs"] == 1
    assert summary["possible_duplicate_pairs"] == 1
    assert {row["system_merged"] for row in reviews} == {True, False}
    assert all(row["identity_verdict"] is None for row in reviews)
