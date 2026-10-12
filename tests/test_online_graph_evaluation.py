"""验收报告不能把降级、伪出处或模型复核状态计作图链路或语义通过。"""

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from zhiqing_rag.core.config import Settings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_online_graph.py"
SPEC = importlib.util.spec_from_file_location("online_graph_evaluation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CASE = MODULE.CASES[3]
CONTENT = "接收方确认后，原受理人负责更新处理记录。"


def response_body():
    return {
        "plan": {
            "route": "hybrid_graph",
            "requested_route": "hybrid_graph",
            "original_query": CASE["query"],
            "query": CASE["query"],
            "reason_code": "GRAPH_APPROVED",
            "params": {
                "candidate_k": 30,
                "top_k": 5,
                "rerank": True,
                "seed_entity_uids": ["seed-1"],
                "max_hops": 1,
                "projections": [
                    {
                        "tenant_id": "1",
                        "document_id": "76",
                        "generation_id": "200",
                        "projection_id": "400",
                        "namespace": "test",
                    }
                ],
            },
        },
        "retrieval": {
            "trace": {
                "graph_status": "succeeded",
                "graph_count": 1,
                "graph_seed_count": 1,
                "graph_assertion_count": 1,
                "graph_hops": 1,
            },
            "evidence": [
                {
                    "citation_id": "E1",
                    "document_id": "76",
                    "content": CONTENT,
                    "graph_relations": [
                        {
                            "assertion_id": "relation-1",
                            "seed_entity_uid": "seed-1",
                            "subject_entity_uid": "seed-1",
                            "object_entity_uid": "object-1",
                            "predicate": "RESPONSIBLE_FOR",
                            "condition_text": "接收方确认后",
                            "source_quote": CONTENT,
                            "quote_start": 0,
                            "quote_end": len(CONTENT),
                        }
                    ],
                }
            ],
        },
    }


def test_valid_one_hop_path_and_preserved_condition_pass_without_claiming_semantics():
    result = MODULE.evaluate_response(CASE, response_body())
    assert result["passed"] and result["provenance_valid"] and result["condition_preserved"]
    assert "semantic_correct" not in result


@pytest.mark.parametrize(
    "change",
    [
        {"quote_start": 1},
        {"quote_end": len(CONTENT) + 1},
        {"source_quote": "接收方确认后更新处理记录", "quote_end": 14},
        {"condition_text": "未经接收方确认"},
        {"seed_entity_uid": "hidden-entity"},
        {"subject_entity_uid": "another-entity"},
    ],
)
def test_relation_requires_exact_contiguous_quote_condition_and_seed_endpoint(change):
    body = response_body()
    body["retrieval"]["evidence"][0]["graph_relations"][0].update(change)
    result = MODULE.evaluate_response(CASE, body)
    assert not result["passed"] and not result["provenance_valid"]


def test_g4_cannot_pass_when_relationship_drops_required_condition():
    body = response_body()
    body["retrieval"]["evidence"][0]["graph_relations"][0]["condition_text"] = ""
    result = MODULE.evaluate_response(CASE, body)
    assert result["provenance_valid"] and not result["condition_preserved"]
    assert not result["passed"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_status", "failed"),
        ("graph_count", 0),
        ("graph_seed_count", 0),
        ("graph_assertion_count", 0),
        ("graph_hops", 2),
    ],
)
def test_successful_nonzero_one_hop_trace_is_required(field, value):
    body = response_body()
    body["retrieval"]["trace"][field] = value
    assert not MODULE.evaluate_response(CASE, body)["passed"]


def test_graph_route_or_trace_without_returned_provenance_cannot_pass():
    body = response_body()
    body["retrieval"]["evidence"][0]["graph_relations"] = []
    assert not MODULE.evaluate_response(CASE, body)["passed"]
    body = response_body()
    body["plan"]["route"] = "hybrid"
    assert not MODULE.evaluate_response(CASE, body)["passed"]
    body = response_body()
    body["retrieval"]["evidence"][0]["document_id"] = "71"
    assert not MODULE.evaluate_response(CASE, body)["passed"]


def test_baseline_requires_no_graph_execution_or_provenance():
    body = response_body()
    assert not MODULE.evaluate_response(CASE, body, baseline=True)["passed"]
    body["plan"]["route"] = "hybrid"
    body["retrieval"]["trace"] = {
        "graph_status": "skipped",
        "graph_count": 0,
        "graph_seed_count": 0,
        "graph_assertion_count": 0,
    }
    body["retrieval"]["evidence"][0]["graph_relations"] = []
    assert MODULE.evaluate_response(CASE, body, baseline=True)["passed"]


def test_answer_quotes_are_checked_but_model_review_does_not_establish_semantic_correctness():
    body = response_body()
    body["answer"] = {
        "status": "answered",
        "review_status": "passed",
        "statements": [
            {"text": "陈述仍需事实复核。", "citations": [{"citation_id": "E1", "quote": CONTENT}]}
        ],
    }
    result = MODULE.evaluate_response(CASE, body)
    assert result["citation_valid"] and result["semantic_correct"] is None
    body["answer"]["statements"][0]["citations"][0]["quote"] = "转派即完成"
    assert not MODULE.evaluate_response(CASE, body)["citation_valid"]
    body["answer"]["statements"] = []
    assert not MODULE.evaluate_response(CASE, body)["citation_valid"]


def frozen_and_rows():
    frozen = {
        "76": {
            "document_id": 76,
            "tenant_id": 1,
            "kb_id": 7,
            "revision_id": 76,
            "source_sha256": "a" * 64,
        }
    }
    rows = [
        {
            "tenant_id": 1,
            "knowledge_base_id": 7,
            "document_id": 76,
            "revision_id": 76,
            "generation_id": 200,
            "file_sha256": "a" * 64,
            "content": CONTENT,
            "content_sha256": hashlib.sha256(CONTENT.encode()).hexdigest(),
        }
    ]
    return frozen, rows


def test_source_verification_accepts_same_revision_rebuild():
    frozen, rows = frozen_and_rows()
    verified = MODULE.verify_source_rows(frozen, rows)
    assert verified[0]["generation_id"] == "200"
    assert verified[0]["verified_chunk_count"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"tenant_id": 2},
        {"knowledge_base_id": 8},
        {"document_id": 77},
        {"revision_id": 77},
        {"file_sha256": "b" * 64},
        {"content_sha256": "b" * 64},
    ],
)
def test_source_verification_rejects_changed_identity_file_revision_or_content(change):
    frozen, rows = frozen_and_rows()
    rows[0].update(change)
    with pytest.raises(ValueError):
        MODULE.verify_source_rows(frozen, rows)


async def test_failed_source_gate_blocks_all_model_endpoints_and_still_revokes_session(
    tmp_path, monkeypatch, capsys
):
    settings = Settings(_env_file=None, app_env="test")
    calls, cleaned = [], []
    fake_app = SimpleNamespace(dependency_overrides={})

    class Client:
        headers = {}

        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, json):
            calls.append(endpoint)
            if endpoint == "/api/auth/login":
                return httpx.Response(
                    200,
                    json={
                        "selection_token": "private-selection",
                        "tenants": [{"code": "DEFAULT", "id": "1"}],
                    },
                )
            if endpoint == "/api/auth/select-tenant":
                return httpx.Response(200, json={"access_token": "private-access"})
            pytest.fail("来源核对失败后不得请求问题分析、检索或回答")

    def failed_sources(*args):
        raise ValueError("private-source-error")

    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "get_settings", lambda: settings)
    monkeypatch.setattr(MODULE, "create_app", lambda: fake_app)
    monkeypatch.setattr(MODULE, "load_frozen_documents", lambda *args: ({}, "frozen-digest"))
    monkeypatch.setattr(MODULE, "read_password", lambda args: "private-password")
    monkeypatch.setattr(MODULE, "decode_access_token", lambda *args: {"sid": "new-session"})
    monkeypatch.setattr(MODULE, "verify_sources", failed_sources)
    monkeypatch.setattr(MODULE.httpx, "AsyncClient", Client)
    monkeypatch.setattr(MODULE, "cleanup_session", lambda claims, settings: cleaned.append(claims))
    monkeypatch.setattr(MODULE, "close_database_pool", lambda settings: None)
    args = SimpleNamespace(
        output=tmp_path / "run",
        only=["G4"],
        answer=True,
        email="test@example.local",
        tenant_code="DEFAULT",
        credentials_file=None,
    )
    with pytest.raises(ValueError):
        await MODULE.run(args)
    report_text = (args.output / "results.json").read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert calls == ["/api/auth/login", "/api/auth/select-tenant"]
    assert cleaned == [{"sid": "new-session"}]
    assert not report["source_verified"] and report["evaluation_session_revoked"]
    assert report["fatal_error"] == {"stage": "source_verification", "type": "ValueError"}
    assert "private-" not in report_text
    assert not capsys.readouterr().out


def test_only_selection_rejects_unknown_cases_and_does_not_mutate_fixed_questions():
    original = deepcopy(MODULE.CASES)
    assert [row["id"] for row in MODULE.selected_cases(["G4"])] == ["G4"]
    with pytest.raises(ValueError):
        MODULE.selected_cases(["G5"])
    assert MODULE.CASES == original


async def test_script_keeps_baseline_settings_and_persists_each_completed_request(
    tmp_path, monkeypatch, capsys
):
    settings = Settings(_env_file=None, app_env="test")
    fake_app = SimpleNamespace(dependency_overrides={})
    calls, progress, cleaned = [], [], []
    original_writer = MODULE.write_report

    def record_progress(output, report):
        original_writer(output, report)
        progress.append(
            len(json.loads((output / "results.json").read_text(encoding="utf-8"))["results"])
        )

    class Client:
        headers = {}

        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, endpoint, json):
            if endpoint == "/api/auth/login":
                return httpx.Response(
                    200,
                    json={
                        "selection_token": "private-selection",
                        "tenants": [{"code": "DEFAULT", "id": "1"}],
                    },
                )
            if endpoint == "/api/auth/select-tenant":
                return httpx.Response(200, json={"access_token": "private-access"})
            assert endpoint == "/api/query/search"
            assert json == {"query": CASE["query"], "document_ids": ["76"]}
            override = fake_app.dependency_overrides.get(MODULE.get_graph_service)
            body = response_body()
            if override is not None:
                disabled = override()
                assert not disabled.settings.query_graph_enabled
                assert disabled.settings.embedding_model == settings.embedding_model
                assert disabled.settings.query_analyzer_model == settings.query_analyzer_model
                body["plan"]["route"] = "hybrid"
                body["retrieval"]["trace"] = {"graph_status": "skipped"}
                body["retrieval"]["evidence"][0]["graph_relations"] = []
            calls.append("baseline" if override else "graph")
            return httpx.Response(200, json=body)

    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "get_settings", lambda: settings)
    monkeypatch.setattr(MODULE, "create_app", lambda: fake_app)
    monkeypatch.setattr(MODULE, "load_frozen_documents", lambda *args: ({}, "frozen-digest"))
    monkeypatch.setattr(MODULE, "read_password", lambda args: "private-password")
    monkeypatch.setattr(MODULE, "decode_access_token", lambda *args: {"sid": "new-session"})
    monkeypatch.setattr(MODULE, "verify_sources", lambda *args: ({}, [{"document_id": "76"}]))
    monkeypatch.setattr(MODULE.httpx, "AsyncClient", Client)
    monkeypatch.setattr(MODULE, "cleanup_session", lambda claims, settings: cleaned.append(claims))
    monkeypatch.setattr(MODULE, "close_database_pool", lambda settings: None)
    monkeypatch.setattr(MODULE, "write_report", record_progress)
    args = SimpleNamespace(
        output=tmp_path / "run",
        only=["G4"],
        answer=False,
        email="test@example.local",
        tenant_code="DEFAULT",
        credentials_file=None,
    )
    report = await MODULE.run(args)
    assert calls == ["graph", "baseline"] and cleaned == [{"sid": "new-session"}]
    assert 1 in progress and progress[-1] == 2
    assert report["source_verified"] and report["evaluation_session_revoked"]
    assert report["summary"] == {
        "graph_passed": 1,
        "baseline_valid": 1,
        "case_count": 1,
        "semantic_correct": None,
    }
    assert not fake_app.dependency_overrides
    output_lines = capsys.readouterr().out.splitlines()
    assert len(output_lines) == 2
    assert all(
        set(json.loads(line))
        == {"id", "http_status", "route", "graph_status", "graph_count", "elapsed_ms"}
        for line in output_lines
    )
    assert "private-" not in (args.output / "results.json").read_text(encoding="utf-8")
