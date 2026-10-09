"""Relocated Gold preserves target counts and requires evidence within the new chunk."""

import pytest

from scripts import evaluate_graph_table_replay as evaluation
from scripts.evaluate_graph_table_replay import relocate


def test_relocating_shared_actor_adds_only_unscored_endpoint():
    chunks = [
        {"local_chunk_id": "first", "document_id": 83, "metadata": {},
         "content": "主管给出入口。"},
        {"local_chunk_id": "second", "document_id": 83, "metadata": {},
         "content": "主管确定动作。"},
    ]
    old = [{"case_id": "old", "document_id": 83, "entities": [
        {"id": "e1", "type": "Actor", "name": "主管",
         "evidence_quote": "主管给出入口。"},
        {"id": "e2", "type": "Step", "name": "给出入口",
         "evidence_quote": "主管给出入口。"},
        {"id": "e3", "type": "Step", "name": "确定动作",
         "evidence_quote": "主管确定动作。"},
    ], "relations": [
        {"source_id": "e1", "target_id": "e2", "type": "RESPONSIBLE_FOR",
         "condition_text": None, "evidence_quote": "主管给出入口。"},
        {"source_id": "e1", "target_id": "e3", "type": "RESPONSIBLE_FOR",
         "condition_text": None, "evidence_quote": "主管确定动作。"},
    ]}]
    cases = relocate(chunks, old)
    assert sum(len(c["entities"]) for c in cases) == 4
    assert sum(len(e["scored_target_ids"]) for c in cases for e in c["entities"]) == 3
    second_actor = next(e for e in cases[1]["entities"] if e["name"] == "主管")
    assert second_actor["scored_target_ids"] == []
    assert second_actor["evidence_quote"] == "主管确定动作。"
    for case in cases:
        for evidence in case["evidence"]:
            assert case["content"][evidence["start"]:evidence["end"]] == evidence["quote"]


def test_relocation_rejects_gold_endpoint_not_supported_by_its_row():
    chunks = [{"local_chunk_id": "first", "document_id": 83, "metadata": {},
               "content": "主管给出入口。人事确定动作。"}]
    old = [{"case_id": "old", "document_id": 83, "entities": [
        {"id": "e1", "type": "Actor", "name": "主管",
         "evidence_quote": "主管给出入口。"},
        {"id": "e2", "type": "Step", "name": "确定动作",
         "evidence_quote": "人事确定动作。"},
    ], "relations": [{"source_id": "e1", "target_id": "e2",
                      "type": "RESPONSIBLE_FOR", "condition_text": None,
                      "evidence_quote": "人事确定动作。"}]}]
    with pytest.raises(ValueError, match="需人工裁决"):
        relocate(chunks, old)


@pytest.mark.parametrize("api_unavailable", [False, True])
def test_report_distinguishes_endpoint_loss_relation_error_and_transport_failure(
    tmp_path, monkeypatch, api_unavailable
):
    content = "主管批准后管理员开通。"
    entities = [
        {"id": "e1", "type": "Actor", "name": "管理员", "canonical_name": "管理员",
         "evidence_quote": content, "scored_target_ids": ["old:e1"]},
        {"id": "e2", "type": "Step", "name": "开通", "canonical_name": "开通",
         "evidence_quote": content, "scored_target_ids": ["old:e2"]},
    ]
    relation = {"source_id": "e1", "target_id": "e2", "type": "RESPONSIBLE_FOR",
                "evidence_quote": content, "condition_text": "主管批准后",
                "origin_target_id": "old:r1"}
    cases = [{"case_id": f"case{i}", "content": content, "content_sha256": "fixed",
              "entities": entities, "relations": [relation]} for i in range(4)]
    monkeypatch.setattr(evaluation, "verify", lambda: cases)
    monkeypatch.setattr(evaluation, "TARGETS", tmp_path)
    evaluation.write_json(tmp_path / "manifest.json", {})
    evaluation.write_json(tmp_path / "metadata.json", {
        "targets_manifest_sha256": evaluation.sha(tmp_path / "manifest.json")})
    rows = []
    for i, case in enumerate(cases):
        row = {"case_id": case["case_id"], "content_sha256": "fixed", "attempts": []}
        if api_unavailable:
            row.update(status="ERROR", error_type="APIConnectionError")
        else:
            predicted_entities = [{k: v for k, v in e.items() if k != "scored_target_ids"}
                                  for e in (entities[:1] if i == 1 else entities)]
            predicted_relation = {k: v for k, v in relation.items()
                                  if k != "origin_target_id"}
            if i == 0:
                predicted_relation["condition_text"] = None
            elif i == 2:
                predicted_relation["type"] = "REVIEWS"
            row.update(status="VALID_EXTRACTION", extraction={
                "entities": predicted_entities,
                "relations": [] if i == 1 else [predicted_relation]})
        rows.append(row)
    evaluation.write_jsonl(tmp_path / "replay.jsonl", rows)
    evaluation.report(tmp_path)
    import json

    result = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    if api_unavailable:
        assert result["entity_recall"] is None
        assert result["triple_recall"] is None
        assert result["counts"]["EXTRACTION_UNAVAILABLE"] == 4
        assert result["counts"].get("MISSING_ENTITY_ENDPOINT", 0) == 0
    else:
        assert result["counts"]["MISSING_ENTITY_ENDPOINT"] == 1
        assert result["counts"]["RELATION_WITH_ENDPOINTS_PRESENT"] == 1
        assert result["counts"]["conditional_eligible"] == 3
        assert result["counts"]["conditional_hits"] == 2
        assert result["counts"]["qualified_hits"] == 1
        assert result["final_precision"] is None
