"""冻结来源门禁与分开的路径、回答检查，失败结果不得用模型复核标签遮盖。"""

import hashlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_online_graph_v2.py"
SPEC = importlib.util.spec_from_file_location("online_graph_v2_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_result(case_index=0):
    data = MODULE.load_dataset()
    case = data["cases"][case_index]
    documents = data["documents"][:2]
    evidence, hops, projections = [], [], []
    for index, document in enumerate(documents, 1):
        content = document["content"]
        projection = {"tenant_id": "1", "document_id": str(document["id"]),
                      "generation_id": str(document["id"]), "projection_id": str(index),
                      "namespace": f"isolated:{index}"}
        projections.append(projection)
        source = {**projection, "knowledge_base_id": "990", "chunk_id": str(document["id"]),
                  "revision_id": str(document["id"]), "source_quote": content,
                  "quote_start": 0, "quote_end": len(content),
                  "content_sha256": hashlib.sha256(content.encode()).hexdigest()}
        subject, object_ = ("step", "material1") if index == 1 else ("material2", "field")
        relation = {"predicate": "REQUIRES_MATERIAL" if index == 1 else "HAS_FIELD",
                    "subject_entity_uid": subject, "object_entity_uid": object_,
                    "source_quote": content,
                    "condition_text": "预算已确认后" if index == 1 else ""}
        evidence.append({**source, "citation_id": f"E{index}", "content": content,
                         "graph_relations": [relation]})
        hops.append({"source": source, "relation": relation})
    path = {"hops": hops, "bridges": [{"source_entity_uid": "material1",
            "target_entity_uid": "material2", "sources": [hop["source"] for hop in hops]}],
            "required_chunk_ids": [item["chunk_id"] for item in evidence]}
    plan = {"route": "hybrid_graph", "params": {"max_hops": 2,
            "seed_entity_uids": ["step"], "projections": projections}}
    retrieval = {"evidence": evidence, "graph_paths": [path],
                 "trace": {"graph_hops": 2, "graph_path_count": 1}}
    answer = {"status": "answered", "statements": [{
        "text": "预算已确认后，采购申请步骤需要采购申请表（又称购置需求单），"
                "该表包含用途、数量和申请人。",
        "citations": [{"citation_id": item["citation_id"], "quote": item["content"]}
                      for item in evidence],
    }]}
    return case, plan, retrieval, answer


def test_frozen_fixture_accepts_only_exact_approved_bytes(tmp_path):
    original = MODULE.DATASET.read_bytes()
    path = tmp_path / "cases.json"
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="SHA"):
        MODULE.load_dataset(path)
    path.write_bytes(original)
    assert MODULE.load_dataset(path)["dataset_kind"] == "synthetic_development"


def test_all_frozen_relations_pass_current_raw_evidence_validator():
    data = MODULE.load_dataset()
    for document in data["documents"]:
        extraction = MODULE.fixture_extraction(document)
        assert extraction.relations
        assert MODULE.fixture_record(document).content == document["content"]


def test_path_check_is_independent_of_answer_success_and_does_not_claim_semantics():
    case, plan, retrieval, answer = fixture_result()
    value = MODULE.assess_result(case, "two_hop", plan, retrieval, answer)
    assert value == {"path_passed": True, "answer_evidence_checks_passed": True,
                     "semantic_correct": None}
    answer = {"status": "insufficient", "statements": [], "review_status": "failed"}
    value = MODULE.assess_result(case, "two_hop", plan, retrieval, answer)
    assert value["path_passed"] and not value["answer_evidence_checks_passed"]


def test_python_model_dump_tuple_sources_are_valid_before_json_serialization():
    case, plan, retrieval, answer = fixture_result()
    bridge = retrieval["graph_paths"][0]["bridges"][0]
    bridge["sources"] = tuple(bridge["sources"])
    assert MODULE.assess_result(case, "two_hop", plan, retrieval, answer)["path_passed"]


@pytest.mark.parametrize("tamper", ["sha", "namespace", "missing_source", "partial_condition"])
def test_path_without_current_contiguous_source_and_full_condition_cannot_pass(tamper):
    case, plan, retrieval, _ = fixture_result()
    if tamper == "sha":
        retrieval["graph_paths"][0]["hops"][0]["source"]["content_sha256"] = "0" * 64
    elif tamper == "namespace":
        retrieval["graph_paths"][0]["hops"][1]["source"]["namespace"] = "isolated:1"
    elif tamper == "missing_source":
        retrieval["evidence"].pop()
    else:
        retrieval["graph_paths"][0]["hops"][0]["relation"]["condition_text"] = "预算"
    assert not MODULE.assess_result(case, "two_hop", plan, retrieval)["path_passed"]


@pytest.mark.parametrize("tamper", ["fields_only", "condition", "fake_quote", "alias_proof"])
def test_answer_needs_its_own_condition_fields_and_applicable_source_quotes(tamper):
    case, plan, retrieval, answer = fixture_result(1)
    answer = deepcopy(answer)
    statement = answer["statements"][0]
    if tamper == "fields_only":
        statement["citations"] = statement["citations"][1:]
    elif tamper == "condition":
        statement["text"] = statement["text"].replace("预算已确认后，", "")
    elif tamper == "fake_quote":
        statement["citations"][0]["quote"] = "采购申请步骤需审核后再填写。"
    else:
        statement["citations"][1]["quote"] = "采购申请表包含用途、数量和申请人。"
    value = MODULE.assess_result(case, "two_hop", plan, retrieval, answer)
    assert value["path_passed"] and not value["answer_evidence_checks_passed"]


def test_negative_case_rejects_published_guess_even_when_there_is_no_path():
    _, plan, retrieval, answer = fixture_result()
    case = MODULE.load_dataset()["cases"][4]
    retrieval["graph_paths"] = []
    retrieval["trace"]["graph_hops"] = 1
    value = MODULE.assess_result(case, "two_hop", plan, retrieval, answer)
    assert value["path_passed"] and not value["answer_evidence_checks_passed"]


def test_hybrid_baseline_must_not_execute_or_publish_a_graph_path():
    case, plan, retrieval, _ = fixture_result()
    assert not MODULE.assess_result(case, "hybrid", plan, retrieval)["path_passed"]
    retrieval["graph_paths"] = []
    retrieval["trace"]["graph_status"] = "skipped"
    plan["route"] = "hybrid"
    assert MODULE.assess_result(case, "hybrid", plan, retrieval)["path_passed"]


def test_output_cannot_overwrite_a_previous_report_or_escape_the_workspace(tmp_path):
    with pytest.raises(ValueError, match="输出必须"):
        MODULE.checked_output(tmp_path / "outside")
    with pytest.raises(ValueError, match="新运行目录"):
        MODULE.checked_output(MODULE.DATASET.parent.parent / "runs")


def test_run_gate_requires_every_new_route_case_but_preserves_baseline_failures():
    results = [{"mode": mode, "path_passed": True,
                "answer_evidence_checks_passed": mode == "two_hop"}
               for _ in range(5) for mode in ("hybrid", "one_hop", "two_hop")]
    report = {"cleanup": True, "results": results}
    assert MODULE.report_passed(report)
    results[2]["answer_evidence_checks_passed"] = False
    assert not MODULE.report_passed(report)
    results[2]["answer_evidence_checks_passed"] = True
    report["cleanup"] = False
    assert not MODULE.report_passed(report)
    report["cleanup"] = True
    report["results"] = results[:-1]
    assert not MODULE.report_passed(report)
