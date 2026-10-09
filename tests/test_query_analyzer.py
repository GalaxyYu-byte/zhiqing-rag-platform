"""分析契约、原文依据、失败边界与已登录 HTTP 入口。"""

import asyncio
import json
import logging
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AuthenticationError
from pydantic import ValidationError

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.query_analysis.models import (
    Evidence,
    GroundingError,
    QueryAnalysisRequest,
    QueryAnalysisResult,
    QueryFeatures,
)
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_analysis.service import QueryAnalyzerService

QUERY = "工单关闭前需要谁确认？"


def term(text, *, quote=None, source="query", index=None):
    return {
        "text": text,
        "evidence": {"source": source, "history_index": index, "quote": quote or text},
    }


def features(**changes):
    result = {
        "intent": "procedure",
        "entities": [term("工单")],
        "keywords": [term("关闭"), term("确认")],
        "constraints": [],
        "relation_required": True,
        "needs_context": False,
        "clarification_question": None,
    }
    return {**result, **changes}


class FakeClient:
    def __init__(self, payload=None, *, error=None, finish_reason="stop", delay=0, refusal=None):
        self.payload = features() if payload is None else payload
        self.error = error
        self.finish_reason = finish_reason
        self.delay = delay
        self.refusal = refusal
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=content, refusal=self.refusal),
                )
            ]
        )

    async def close(self):
        self.closed = True


def settings(**changes):
    # _env_file=None 仍会读取进程环境；显式置空凭据防止单测意外调用真实模型。
    values = {
        "deepseek_api_key": "",
        "query_analyzer_enabled": True,
        "query_analyzer_timeout_seconds": 15,
        "query_analyzer_max_output_tokens": 2048,
    }
    return Settings(_env_file=None, app_env="test", **{**values, **changes})


async def test_valid_analysis_keeps_original_query_and_grounded_features():
    client = FakeClient()
    request = QueryAnalysisRequest(query=QUERY)
    result = await QueryAnalyzerService(settings(), client=client).analyze(request)
    assert result.status == "ok"
    assert result.original_query == QUERY
    assert result.analysis.intent == "procedure"
    assert result.analysis.entities[0].text == "工单"
    assert result.analysis.relation_required
    assert result.fallback_hint is None
    assert result.degradation_reason is None
    assert not client.closed
    call = client.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert json.loads(call["messages"][1]["content"]) == request.model_dump()


def constraint(field="uploaded_at", operator="range", value="去年", quote="去年上传"):
    return {
        "field": field,
        "operator": operator,
        "value": value,
        "evidence": {"source": "query", "history_index": None, "quote": quote},
    }


async def test_constraints_keep_raw_time_and_negation_without_normalizing_dates():
    payload = features(
        intent="statistics",
        entities=[],
        keywords=[],
        relation_required=False,
        constraints=[constraint(), constraint("file_type", "neq", "PDF", "不是PDF")],
    )
    query = "去年上传且不是PDF的文档有多少？"
    result = await QueryAnalyzerService(settings(), client=FakeClient(payload)).analyze(
        QueryAnalysisRequest(query=query)
    )
    assert result.status == "ok"
    assert result.analysis.constraints[0].value == "去年"
    assert result.analysis.constraints[1].operator == "neq"


@pytest.mark.parametrize(
    "payload",
    [
        features(entities=[term("合同")]),
        features(entities=[term("工单", quote="工单由经理确认")]),
        features(entities=[term("工单", source="history", index=0)]),
        features(needs_context=True),
        features(constraints=[constraint(value="2025年")]),
    ],
)
async def test_ungrounded_output_is_discarded_entirely(payload):
    result = await QueryAnalyzerService(settings(), client=FakeClient(payload)).analyze(
        QueryAnalysisRequest(query=QUERY)
    )
    assert result.status == "degraded"
    assert result.analysis is None
    assert result.degradation_reason == "invalid_grounding"


@pytest.mark.parametrize(
    "payload",
    [
        features(route="hybrid"),
        features(sql="SELECT * FROM documents"),
        features(weights=[0.5, 0.5]),
        features(intent="invented"),
        features(relation_required="true"),
        features(intent="relationship", relation_required=False),
        features(intent="chitchat"),
        features(entities=[term("工单"), term("工单")]),
        {"intent": "fact"},
        "```json\n{}\n```",
        '{"intent":"fact","intent":"chitchat"}',
        "[]",
    ],
)
async def test_invalid_or_injected_model_fields_degrade(payload):
    result = await QueryAnalyzerService(settings(), client=FakeClient(payload)).analyze(
        QueryAnalysisRequest(query=QUERY)
    )
    assert result.status == "degraded"
    assert result.degradation_reason == "invalid_response"
    assert result.analysis is None


async def test_history_entity_can_be_grounded_only_for_context_dependent_query():
    payload = features(
        entities=[term("工单", source="history", index=0)],
        keywords=[],
        needs_context=True,
    )
    request = QueryAnalysisRequest(query="它由谁确认？", history=[QUERY])
    result = await QueryAnalyzerService(settings(), client=FakeClient(payload)).analyze(request)
    assert result.status == "ok"
    payload["needs_context"] = False
    result = await QueryAnalyzerService(settings(), client=FakeClient(payload)).analyze(request)
    assert result.degradation_reason == "invalid_grounding"


def test_history_filters_cannot_be_silently_inherited():
    item = constraint()
    item["evidence"].update(source="history", history_index=0)
    analysis = QueryFeatures.model_validate(
        features(
            entities=[],
            keywords=[],
            constraints=[item],
            needs_context=True,
        )
    )
    with pytest.raises(GroundingError):
        analysis.validate_grounding(
            QueryAnalysisRequest(
                query="列出工单文档",
                history=["去年上传的文档"],
            )
        )


def test_history_indices_and_request_limits_are_strict():
    for bad_evidence in (
        {"source": "query", "history_index": 0, "quote": "工单"},
        {"source": "history", "history_index": -1, "quote": "工单"},
        {"source": "history", "history_index": True, "quote": "工单"},
    ):
        with pytest.raises(ValidationError):
            Evidence.model_validate(bad_evidence)
    for body in (
        {"query": "   "},
        {"query": "a" * 2001},
        {"query": QUERY, "history": ["旧问题"] * 7},
        {"query": QUERY, "history": [{"role": "assistant", "content": "旧答案"}]},
    ):
        with pytest.raises(ValidationError):
            QueryAnalysisRequest.model_validate(body)


@pytest.mark.parametrize(
    "query,hint",
    [
        (QUERY, "basic_retrieval"),
        ("它怎么处理？", "clarify"),
        ("文档有多少份？", "unsupported"),
        ("统计所有合同的金额总额", "unsupported"),
        ("列出所有文档", "unsupported"),
    ],
)
async def test_model_failure_fallback_uses_only_current_question(query, hint):
    error = httpx.ConnectError("secret API key and raw provider response")
    client = FakeClient(error=error)
    result = await QueryAnalyzerService(settings(), client=client).analyze(
        QueryAnalysisRequest(query=query, history=["它怎么处理？"])
    )
    assert result.fallback_hint == hint
    assert result.degradation_reason == "model_unavailable"
    assert result.analysis is None
    assert "secret" not in result.model_dump_json()
    assert len(client.calls) == 1


async def test_timeout_is_bounded_and_cancellation_propagates():
    service = QueryAnalyzerService(
        settings(query_analyzer_timeout_seconds=0.01),
        client=FakeClient(delay=1),
    )
    result = await service.analyze(QueryAnalysisRequest(query=QUERY))
    assert result.degradation_reason == "timeout"
    service = QueryAnalyzerService(settings(), client=FakeClient(error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await service.analyze(QueryAnalysisRequest(query=QUERY))


async def test_failure_log_exposes_type_status_and_timing_without_sensitive_content(caplog):
    secret = "PRIVATE_PROVIDER_BODY_AND_KEY"
    error = AuthenticationError(
        secret,
        response=httpx.Response(401, request=httpx.Request("POST", "https://example.invalid")),
        body={"message": secret},
    )
    with caplog.at_level(logging.WARNING, logger="zhiqing_rag.query_analysis.service"):
        result = await QueryAnalyzerService(settings(), client=FakeClient(error=error)).analyze(
            QueryAnalysisRequest(query=QUERY)
        )
    assert result.degradation_reason == "model_unavailable"
    assert "reason=model_unavailable" in caplog.text
    assert "error_type=AuthenticationError" in caplog.text
    assert "http_status=401" in caplog.text
    assert "latency_ms=" in caplog.text
    assert secret not in caplog.text
    assert QUERY not in caplog.text


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
async def test_incomplete_responses_are_not_accepted(finish_reason):
    result = await QueryAnalyzerService(
        settings(),
        client=FakeClient(finish_reason=finish_reason),
    ).analyze(QueryAnalysisRequest(query=QUERY))
    assert result.degradation_reason == "incomplete_response"
    assert result.analysis is None


async def test_token_limit_failure_logs_metadata_without_accepting_partial_analysis(caplog):
    client = FakeClient(finish_reason="length")
    create = client.create

    async def limited(**kwargs):
        response = await create(**kwargs)
        response.usage = SimpleNamespace(
            completion_tokens=2048,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2048),
        )
        return response

    client.chat.completions.create = limited
    with caplog.at_level(logging.WARNING, logger="zhiqing_rag.query_analysis.service"):
        result = await QueryAnalyzerService(settings(), client=client).analyze(
            QueryAnalysisRequest(query=QUERY)
        )
    assert result.degradation_reason == "incomplete_response"
    assert result.analysis is None
    assert "finish_reason=length" in caplog.text
    assert "completion_tokens=2048" in caplog.text
    assert "reasoning_tokens=2048" in caplog.text
    assert QUERY not in caplog.text


async def test_untrusted_completion_metadata_is_not_echoed_to_logs(caplog):
    secret = "PRIVATE_PROVIDER_METADATA"
    client = FakeClient(finish_reason=secret)
    create = client.create

    async def malformed(**kwargs):
        response = await create(**kwargs)
        response.usage = SimpleNamespace(
            completion_tokens=secret,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=secret),
        )
        return response

    client.chat.completions.create = malformed
    with caplog.at_level(logging.WARNING, logger="zhiqing_rag.query_analysis.service"):
        result = await QueryAnalyzerService(settings(), client=client).analyze(
            QueryAnalysisRequest(query=QUERY)
        )
    assert result.degradation_reason == "incomplete_response"
    assert "finish_reason=unknown" in caplog.text
    assert secret not in caplog.text


async def test_empty_model_choices_degrade():
    async def create(**kwargs):
        return SimpleNamespace(choices=[])

    client = FakeClient()
    client.chat.completions.create = create
    result = await QueryAnalyzerService(settings(), client=client).analyze(
        QueryAnalysisRequest(query=QUERY)
    )
    assert result.degradation_reason == "invalid_response"


async def test_refusal_cannot_be_used_as_valid_analysis():
    result = await QueryAnalyzerService(
        settings(),
        client=FakeClient(refusal="provider refusal"),
    ).analyze(QueryAnalysisRequest(query=QUERY))
    assert result.status == "degraded"
    assert result.degradation_reason == "invalid_response"
    assert result.analysis is None


def test_result_contract_rejects_mixed_success_and_failure():
    values = {
        "status": "ok",
        "original_query": QUERY,
        "analysis": QueryFeatures.model_validate(features()),
        "fallback_hint": None,
        "message": None,
        "degradation_reason": None,
        "model": "test",
        "latency_ms": 0,
    }
    with pytest.raises(ValidationError):
        QueryAnalysisResult.model_validate({**values, "fallback_hint": "basic_retrieval"})
    with pytest.raises(ValidationError):
        QueryAnalysisResult.model_validate({**values, "status": "degraded"})


@pytest.mark.parametrize("query", ["", "   ", "a" * 2001])
def test_http_rejects_invalid_question_before_model_call(api, query):
    app, model = api
    app.dependency_overrides[require_identity] = lambda: SimpleNamespace(user_id=1, tenant_id=2)
    with TestClient(app) as client:
        response = client.post("/api/query/analyze", json={"query": query})
    assert response.status_code == 422
    assert model.calls == []


async def test_disabled_and_missing_credentials_do_not_call_model():
    client = FakeClient()
    result = await QueryAnalyzerService(
        settings(query_analyzer_enabled=False),
        client=client,
    ).analyze(QueryAnalysisRequest(query=QUERY))
    assert result.degradation_reason == "disabled"
    assert client.calls == []
    result = await QueryAnalyzerService(settings()).analyze(QueryAnalysisRequest(query=QUERY))
    assert result.degradation_reason == "not_configured"


async def test_owned_sdk_client_uses_no_retries_and_is_closed(monkeypatch):
    client = FakeClient()
    created = []

    class OwnedClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            await client.close()

    monkeypatch.setattr("zhiqing_rag.query_analysis.service.AsyncOpenAI", OwnedClient)
    result = await QueryAnalyzerService(settings(deepseek_api_key="unit-test-key")).analyze(
        QueryAnalysisRequest(query=QUERY)
    )
    assert result.status == "ok"
    assert client.closed
    assert created[0]["max_retries"] == 0


@pytest.fixture
def api():
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings()
    client = FakeClient()
    app.dependency_overrides[get_analyzer] = lambda: QueryAnalyzerService(settings(), client=client)
    return app, client


def test_http_rejects_anonymous_request_before_model_call(api):
    app, model = api
    with TestClient(app) as client:
        response = client.post("/api/query/analyze", json={"query": QUERY})
    assert response.status_code == 401
    assert model.calls == []


def test_http_analyzes_for_authenticated_identity_without_history(api):
    app, model = api
    app.dependency_overrides[require_identity] = lambda: SimpleNamespace(user_id=1, tenant_id=2)
    with TestClient(app) as client:
        response = client.post("/api/query/analyze", json={"query": QUERY})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["analysis"]["intent"] == "procedure"
    assert json.loads(model.calls[0]["messages"][1]["content"])["history"] == []


@pytest.mark.parametrize(
    "extra",
    [
        {"history": ["另一个用户的问题"]},
        {"session_id": "other-user"},
        {"tenant_id": 999},
        {"knowledge_base_ids": [999]},
    ],
)
def test_http_rejects_client_supplied_scope_or_history(api, extra):
    app, model = api
    app.dependency_overrides[require_identity] = lambda: SimpleNamespace(user_id=1, tenant_id=2)
    with TestClient(app) as client:
        response = client.post("/api/query/analyze", json={"query": QUERY, **extra})
    assert response.status_code == 422
    assert model.calls == []


def test_http_returns_explicit_degradation(api):
    app, model = api
    app.dependency_overrides[require_identity] = lambda: SimpleNamespace(user_id=1, tenant_id=2)
    model.payload = "bad JSON"
    with TestClient(app) as client:
        response = client.post("/api/query/analyze", json={"query": QUERY})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "degraded"
    assert body["analysis"] is None
    assert body["fallback_hint"] == "basic_retrieval"


def test_grounding_does_not_mutate_request_or_model_features():
    data = features()
    original = deepcopy(data)
    analysis = QueryFeatures.model_validate(data)
    analysis.validate_grounding(QueryAnalysisRequest(query=QUERY))
    assert data == original
