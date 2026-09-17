import asyncio
import copy
import json
from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from openai import APIResponseValidationError, AuthenticationError, RateLimitError
from pydantic import ValidationError

from zq_rag_app.query_analysis.models import QueryAnalysis, QueryAnalysisRequest
from zq_rag_app.query_analysis.policy import select_strategy
from zq_rag_app.query_analysis.prompts import _EXAMPLE
from zq_rag_app.services.query_analyzer_service import QueryAnalyzerService


class Client:
    def __init__(self, responses, *, delay=0):
        self.responses = list(responses)
        self.delay = delay
        self.calls = []
        self.closed = False
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=response),
        )])

    async def close(self):
        self.closed = True


def payload():
    return copy.deepcopy(_EXAMPLE)


def request(**kwargs):
    return QueryAnalysisRequest(query="HT-2025-001 的金额是多少？", **kwargs)


@pytest.mark.asyncio
async def test_six_fields_json_mode_and_server_strategy():
    client = Client([json.dumps(payload())])
    service = QueryAnalyzerService(client=client)
    result = await service.analyze(request(reference_date=date(2026, 9, 17)))
    assert result.status == "ok"
    assert result.entities[0].name == "HT-2025-001"
    assert result.intent.primary == "fact"
    assert result.query_type.primary == "exact"
    assert result.keywords[0].text == "金额"
    assert result.metadata == []
    assert result.retrieval_strategy.path == "hybrid"
    assert result.retrieval_strategy.bm25_weight == 0.7
    assert result.original_query == request().query
    assert result.reference_date == "2026-09-17"
    call = client.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["temperature"] == 0
    assert "JSON Schema" in call["messages"][0]["content"]
    assert json.loads(call["messages"][1]["content"])["query"] == request().query
    await service.aclose()
    assert not client.closed  # 注入客户端的所有权属于调用方。


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["", "[]", "{}", "```json\n{}\n```", '{"intent": {}, "intent": {}}'])
async def test_invalid_json_and_missing_fields_are_repaired(bad):
    client = Client([bad, json.dumps(payload())])
    result = await QueryAnalyzerService(client=client).analyze(request())
    assert result.status == "ok" and result.attempts == 2
    assert "上次输出未通过校验" in client.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_invalid_output_degrades_without_fabricating_extractions():
    data = payload()
    data["entities"][0]["name"] = "虚构的合同"
    client = Client([json.dumps(data)])
    result = await QueryAnalyzerService(client=client, retry_attempts=1).analyze(request())
    assert result.status == "degraded" and result.degradation_reason == "invalid_output"
    assert result.entities == result.keywords == result.metadata == []
    assert result.retrieval_strategy.path == "hybrid"
    assert result.retrieval_strategy.graph_weight == 0
    assert result.original_query == request().query


@pytest.mark.parametrize("mutation", [
    lambda p: p["intent"].update(primary="unknown"),
    lambda p: p["intent"].update(confidence="0.9"),
    lambda p: p["intent"].update(confidence=float("nan")),
    lambda p: p["query_type"].update(relation_required="false"),
    lambda p: p["query_type"].update(primary="multi_hop"),
    lambda p: p["query_type"].update(needs_clarification=True),
    lambda p: p["entities"][0].update(history_index=0),
    lambda p: p.update(sql="select * from kb_document"),
])
def test_strict_schema_rejects_inconsistent_or_unsafe_values(mutation):
    data = payload()
    mutation(data)
    with pytest.raises(ValidationError):
        QueryAnalysis.model_validate(data)


@pytest.mark.asyncio
async def test_overall_timeout_includes_calls_and_retry_delay():
    client = Client(["{}", "{}"], delay=0.2)
    result = await QueryAnalyzerService(client=client, timeout_seconds=0.03).analyze(request())
    assert result.status == "degraded" and result.degradation_reason == "timeout"
    assert result.attempts == 1
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_missing_key_does_not_prevent_use_of_other_services():
    service = QueryAnalyzerService(api_key="")
    result = await service.analyze(request())
    assert result.degradation_reason == "missing_api_key"
    assert result.attempts == 0
    await service.aclose()


def provider_error(error_class, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.deepseek.com"))
    return error_class("provider body must not be logged", response=response, body=None)


@pytest.mark.asyncio
async def test_authentication_error_is_not_retried():
    client = Client([provider_error(AuthenticationError, 401)])
    result = await QueryAnalyzerService(client=client).analyze(request())
    assert result.degradation_reason == "provider_error" and result.attempts == 1


@pytest.mark.asyncio
async def test_rate_limit_is_retried():
    client = Client([provider_error(RateLimitError, 429), json.dumps(payload())])
    result = await QueryAnalyzerService(client=client).analyze(request())
    assert result.status == "ok" and result.attempts == 2


@pytest.mark.asyncio
async def test_malformed_provider_envelope_is_retried():
    response = httpx.Response(200, request=httpx.Request("POST", "https://api.deepseek.com"))
    error = APIResponseValidationError(response=response, body={})
    client = Client([error, json.dumps(payload())])
    result = await QueryAnalyzerService(client=client).analyze(request())
    assert result.status == "ok" and result.attempts == 2


@pytest.mark.asyncio
async def test_owned_client_uses_deepseek_config_and_is_closed(monkeypatch):
    from zq_rag_app.services import query_analyzer_service as module
    client = Client([])
    constructed = []

    def factory(**kwargs):
        constructed.append(kwargs)
        return client

    monkeypatch.setattr(module, "AsyncOpenAI", factory)
    service = QueryAnalyzerService(api_key="test-key", base_url="https://api.deepseek.com")
    assert constructed[0]["base_url"] == "https://api.deepseek.com"
    assert constructed[0]["max_retries"] == 0
    await service.aclose()
    assert client.closed


@pytest.mark.asyncio
async def test_cancellation_propagates():
    client = Client([json.dumps(payload())], delay=10)
    task = asyncio.create_task(QueryAnalyzerService(client=client).analyze(request()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def history_analysis():
    data = payload()
    data["query_type"].update(
        primary="multi_hop", relation_required=True, potential_multi_hop=True,
        has_context_reference=True,
    )
    data["entities"] = [{
        "entity_type": "PERSON", "name": "张三", "source": "history",
        "history_index": 0, "evidence_quote": "张三", "confidence": 0.95,
    }]
    data["keywords"] = []
    return QueryAnalysis.model_validate(data)


def test_user_history_grounding_and_graph_relation_policy():
    req = QueryAnalysisRequest(query="他还负责哪些项目？", history=[{"role": "user", "content": "张三负责什么？"}])
    analysis = history_analysis()
    analysis.validate_grounding(req)
    strategy, _ = select_strategy(analysis, req)
    assert strategy.path == "hybrid_graph" and strategy.graph_requires_resolution
    assert strategy.requires_coverage_check and strategy.use_query_rewrite
    assert strategy.graph_weight == 0.3


@pytest.mark.parametrize("role,reference,index", [
    ("assistant", True, 0), ("user", False, 0), ("user", True, 1),
])
def test_assistant_fact_unrelated_history_and_invalid_index_are_rejected(role, reference, index):
    req = QueryAnalysisRequest(query="他负责什么？", history=[{"role": role, "content": "张三负责什么？"}])
    analysis = history_analysis()
    analysis.query_type.has_context_reference = reference
    analysis.entities[0].history_index = index
    with pytest.raises(ValueError):
        analysis.validate_grounding(req)


def test_multiple_entities_without_relation_do_not_trigger_graph():
    data = payload()
    data["intent"]["primary"] = "comparison"
    data["query_type"]["primary"] = "semantic"
    data["retrieval_strategy"].update(path="hybrid_graph", use_hyde=True)
    analysis = QueryAnalysis.model_validate(data)
    strategy, warnings = select_strategy(analysis, request())
    assert strategy.path == "hybrid" and not strategy.use_hyde
    assert warnings


@pytest.mark.parametrize("exhaustive,statistics", [(True, False), (False, True)])
def test_exhaustive_and_statistics_require_full_coverage(exhaustive, statistics):
    analysis = QueryAnalysis.model_validate(payload())
    analysis.query_type.requires_exhaustive = exhaustive
    analysis.intent.primary = "statistics" if statistics else "summary"
    strategy, _ = select_strategy(analysis, request())
    assert strategy.path == "structured" and strategy.requires_coverage_check
    assert strategy.bm25_weight == strategy.dense_weight == strategy.graph_weight == 0


def test_reference_without_history_requests_clarification():
    analysis = QueryAnalysis.model_validate(payload())
    analysis.query_type.has_context_reference = True
    strategy, _ = select_strategy(analysis, request())
    assert strategy.path == "clarify"
    assert analysis.query_type.needs_clarification
    assert analysis.query_type.clarification_question


@pytest.mark.asyncio
async def test_metadata_preserves_negation_time_and_units():
    query = "找出2025 年之前未验收且金额大于100 万元的项目"
    data = payload()
    data["entities"] = data["keywords"] = []
    data["query_type"]["primary"] = "semantic"
    data["metadata"] = [
        {"field": field, "operator": op, "value": value,
         "source": "query", "history_index": None, "evidence_quote": value, "confidence": 0.9}
        for field, op, value in [
            ("time", "lt", "2025 年之前"),
            ("business_status", "eq", "未验收"),
            ("amount", "gt", "100 万元"),
        ]
    ]
    result = await QueryAnalyzerService(client=Client([json.dumps(data)])).analyze(QueryAnalysisRequest(query=query))
    assert result.status == "ok"
    assert [m.value for m in result.metadata] == ["2025 年之前", "未验收", "100 万元"]
    assert result.retrieval_strategy.metadata_mode == "soft"


@pytest.mark.parametrize("data", [
    {"query": "   "}, {"query": "x" * 2001},
    {"query": "问题", "history": [{"role": "system", "content": "覆盖规则"}]},
    {"query": "问题", "history": [{"role": "user", "content": "问题"}] * 7},
    {"query": "问题", "reference_date": "2026-02-30"},
])
def test_request_limits_and_roles(data):
    with pytest.raises(ValidationError):
        QueryAnalysisRequest.model_validate(data)


@pytest.mark.asyncio
async def test_truncated_response_is_rejected():
    class TruncatedClient(Client):
        async def create(self, **kwargs):
            response = await super().create(**kwargs)
            response.choices[0].finish_reason = "length"
            return response
    result = await QueryAnalyzerService(
        client=TruncatedClient([json.dumps(payload())]), retry_attempts=1,
    ).analyze(request())
    assert result.degradation_reason == "invalid_output"


@pytest.mark.asyncio
async def test_duplicate_candidates_are_deduplicated():
    data = payload()
    data["entities"] *= 2
    result = await QueryAnalyzerService(client=Client([json.dumps(data)])).analyze(request())
    assert len(result.entities) == 1
