"""User policies apply consistently and do not falsely certify pending item review."""

from scripts.record_graph_policy_adjudication import (
    apply_prediction_policies,
    revise_material_subjects,
)
from zhiqing_rag.workers.graph_extraction import Extraction


def entity(entity_id, name, entity_type, quote):
    return {"id": entity_id, "name": name, "canonical_name": name,
            "type": entity_type, "evidence_quote": quote}


def test_material_policy_changes_every_row_and_preserves_parent_case():
    first = "| 发布记录 | 交付版本变化 | 与实际包一致 |"
    second = "| 培训资料 | 面向使用者的讲义 | 内容对应现行版本 |"
    case = {"case_id": "holdout-03-01", "content":
            "| 资料类别 | 文件或成果 | 检查要点 |\n| --- | --- | --- |\n"
            + first + "\n" + second,
            "entities": [entity("a", "交付版本变化", "Material", first),
                         entity("b", "与实际包一致", "Rule", first),
                         entity("c", "面向使用者的讲义", "Material", second),
                         entity("d", "内容对应现行版本", "Rule", second)],
            "relations": [{"source_id": "a", "target_id": "b", "type": "GOVERNED_BY",
                           "evidence_quote": first},
                          {"source_id": "c", "target_id": "d", "type": "GOVERNED_BY",
                           "evidence_quote": second}]}
    revised, changes = revise_material_subjects(case)
    assert len(changes) == 2
    assert [e["name"] for e in revised["entities"] if e["type"] == "Material"] == [
        "发布记录", "培训资料"]
    assert case["entities"][0]["name"] == "交付版本变化"
    assert all(change["item_human_review"] == "PENDING" for change in changes)


def test_field_policy_marks_connected_step_relations_for_reannotation():
    quote = "| 出行目的与行程变化 | 交通凭证 |"
    case = {"case_id": "holdout-05-01", "content":
            "| 申请人应说明 | 需要对应的依据 |\n| --- | --- |\n" + quote}
    extraction = Extraction.model_validate({"entities": [
        entity("a", "申请人", "Actor", "申请人应说明"),
        entity("s", "出行目的与行程变化", "Step", quote),
        entity("m", "交通凭证", "Material", quote)], "relations": [
        {"source_id": "a", "target_id": "s", "type": "RESPONSIBLE_FOR",
         "evidence_quote": quote},
        {"source_id": "s", "target_id": "m", "type": "REQUIRES_MATERIAL",
         "evidence_quote": quote}]})
    flags = apply_prediction_policies(case, extraction)
    assert len(flags) == 3
    assert next(flag for flag in flags if flag["kind"] == "entity")["expected_type"] == "Field"
    assert all(flag["item_human_review"] == "PENDING" for flag in flags)
    assert extraction.entities[1].type == "Step"


def test_pause_policy_approval_keeps_specific_fact_pending():
    quote = "| 交付跟进 | 交付内容发生变化 |"
    case = {"case_id": "holdout-06-02", "content":
            "| 环节 | 暂停或退回条件 |\n| --- | --- |\n" + quote}
    extraction = Extraction.model_validate({"entities": [
        entity("s", "交付跟进", "Step", quote),
        entity("r", "交付内容发生变化", "Rule", quote)], "relations": [
        {"source_id": "s", "target_id": "r", "type": "GOVERNED_BY",
         "evidence_quote": quote}]})
    flag, = apply_prediction_policies(case, extraction)
    assert flag["human_policy_status"] == "APPROVED"
    assert flag["derived_verdict"] == "REVIEW_REQUIRED"
    assert flag["item_human_review"] == "PENDING"
