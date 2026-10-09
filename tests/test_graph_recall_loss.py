"""Check that relation loss is attributed to the missing endpoint, not double counted."""

from scripts.analyze_graph_recall_loss import assess


def test_relation_loss_endpoint_partition():
    cases = []
    evidences = []
    mentions = []
    assertions = []
    for chunk_id in range(1, 6):
        cases.append(
            {
                "source_variant": "frozen",
                "case_id": f"case-{chunk_id}",
                "chunk_id": chunk_id,
                "document_id": 1,
                "coverage_tags": [],
                "entities": [
                    {"id": "a", "type": "Step", "name": "步骤 A", "evidence_quote": "A"},
                    {"id": "b", "type": "Step", "name": "步骤 B", "evidence_quote": "B"},
                ],
                "evidence": [],
                "relations": [
                    {
                        "source_id": "a",
                        "target_id": "b",
                        "type": "NEXT_STEP",
                        "evidence_quote": "A 后是 B",
                    }
                ],
            }
        )
        evidences.append({"key": f"e{chunk_id}", "chunk_id": chunk_id})
        for name, present in (
            ("步骤 A", chunk_id in (2, 4, 5)),
            ("步骤 B", chunk_id in (3, 4, 5)),
        ):
            if present:
                mentions.append(
                    {
                        "entity": {"key": f"{chunk_id}:{name}", "type": "Step", "name": name},
                        "evidence_key": f"e{chunk_id}",
                    }
                )
        if chunk_id == 5:
            assertions.append(
                {
                    "assertion": {"key": "fact", "predicate": "NEXT_STEP"},
                    "subjects": [{"type": "Step", "name": "步骤 A"}],
                    "objects": [{"type": "Step", "name": "步骤 B"}],
                    "evidences": [evidences[-1]],
                }
            )
    snapshot = {"neo4j": {"evidence": evidences, "mentions": mentions, "assertions": assertions}}

    summary, entity_rows, relation_rows = assess(snapshot, cases)

    assert summary["gold_entities"] == 10
    assert summary["matched_entities"] == 6
    assert summary["relation_loss_categories"] == {
        "BOTH_ENDPOINTS_MISSING": 1,
        "MATCH": 1,
        "RELATION_MISSING_WITH_ENDPOINTS": 1,
        "SOURCE_MISSING": 1,
        "TARGET_MISSING": 1,
    }
    assert summary["zero_prediction_chunks"] == 1
    assert summary["gold_relations_in_zero_prediction_chunks"] == 1
    assert len(entity_rows) == 10
    assert len(relation_rows) == 5
