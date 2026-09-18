import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from zq_rag_app.query_analysis.models import QueryAnalysisRequest
from zq_rag_app.query_analysis.multi_query import expansion_block_reason
from zq_rag_app.services.multi_query_service import MultiQueryService


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.delay = 0
        self.closed = False
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        content = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])

    async def close(self):
        self.closed = True


QUERY = "Aurora-KB 谁负责？"
VARIANT = "Aurora-KB 的负责人是谁？"


def candidate(queries=None):
    return {"queries": [VARIANT] if queries is None else queries, "reason": "等价搜索表达"}


def audits(queries=None, **kwargs):
    return {"audits": [{"query": q, "equivalent": True, "intent_preserved": True,
                        "constraints_preserved": True, "no_unsupported_additions": True,
                        "route_preserved": True, "reason": "等价", **kwargs}
                       for q in ([VARIANT] if queries is None else queries)]}


@pytest.mark.asyncio
async def test_generation_audit_json_and_client_lifecycle(analysis_factory):
    analysis = analysis_factory(multi_query=True, names=["Aurora-KB"])
    client = Client([candidate(), audits()])
    service = MultiQueryService(client=client)
    result = await service.expand(QueryAnalysisRequest(query=QUERY), analysis, trigger="analyzer")
    assert result.status == "expanded" and result.queries == (QUERY, VARIANT)
    assert result.trigger == "analyzer" and result.attempts == 1
    assert len(client.calls) == 2
    assert all(call["response_format"] == {"type": "json_object"} for call in client.calls)
    assert all(call["extra_body"] == {"thinking": {"type": "disabled"}} for call in client.calls)
    assert "history" not in json.loads(client.calls[0]["messages"][1]["content"])
    await service.aclose()
    assert not client.closed


@pytest.mark.asyncio
async def test_empty_retrieval_can_trigger_without_analyzer_suggestion(analysis_factory):
    service = MultiQueryService(client=Client([candidate(), audits()]))
    result = await service.expand(QueryAnalysisRequest(query=QUERY), analysis_factory(), trigger="empty_retrieval")
    assert result.status == "expanded" and result.trigger == "empty_retrieval"


@pytest.mark.asyncio
async def test_no_suggestion_never_calls_llm(analysis_factory):
    client = Client([])
    result = await MultiQueryService(client=client).expand(QueryAnalysisRequest(query=QUERY), analysis_factory(), trigger="analyzer")
    assert result.status == "skipped" and not client.calls


@pytest.mark.asyncio
async def test_duplicate_original_and_variants_are_deduplicated(analysis_factory):
    client = Client([candidate([QUERY, QUERY.replace("？", "?")])])
    service = MultiQueryService(client=client)
    result = await service.expand(QueryAnalysisRequest(query=QUERY), analysis_factory(multi_query=True), trigger="analyzer")
    assert result.status == "unchanged" and len(client.calls) == 1
    assert result.queries == (QUERY,)
    client = Client([candidate([VARIANT, VARIANT.replace("？", "?")]), audits()])
    result = await MultiQueryService(client=client).expand(QueryAnalysisRequest(query=QUERY),
                                                         analysis_factory(multi_query=True), trigger="analyzer")
    assert result.queries == (QUERY, VARIANT)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    "{}", "```json\n{}\n```", '{"queries":[],"queries":[]}',
    {"queries": [VARIANT] * 3, "reason": "超过上限"},
    {"queries": [VARIANT], "reason": "新增权限字段", "kb_ids": [999]},
])
async def test_invalid_output_is_retried(analysis_factory, payload):
    client = Client([payload, candidate(), audits()])
    result = await MultiQueryService(client=client).expand(QueryAnalysisRequest(query=QUERY),
                                                         analysis_factory(multi_query=True), trigger="analyzer")
    assert result.status == "expanded" and result.attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["equivalent", "intent_preserved", "constraints_preserved",
                                  "no_unsupported_additions", "route_preserved"])
async def test_each_audit_check_can_reject_expansion(analysis_factory, flag):
    client = Client([candidate(), audits(**{flag: False})])
    result = await MultiQueryService(client=client, retry_attempts=1).expand(
        QueryAnalysisRequest(query=QUERY), analysis_factory(multi_query=True), trigger="analyzer",
    )
    assert result.status == "degraded" and result.queries == (QUERY,)
    assert result.degradation_reason == "invalid_output"


@pytest.mark.asyncio
async def test_audit_cannot_omit_a_query(analysis_factory):
    variants = [VARIANT, "谁是 Aurora-KB 的负责人？"]
    client = Client([candidate(variants), audits([VARIANT])])
    result = await MultiQueryService(client=client, retry_attempts=1).expand(
        QueryAnalysisRequest(query=QUERY), analysis_factory(multi_query=True), trigger="analyzer",
    )
    assert result.status == "degraded" and result.queries == (QUERY,)


@pytest.mark.asyncio
@pytest.mark.parametrize("updated", [
    "查找去年未验收且金额超过 10 美元的记录", "查找去年验收且金额超过 10 元的记录",
    "查找今年未验收且金额超过 10 元的记录", "查找去年未验收且金额超过 100 元的记录",
])
async def test_time_negation_amount_units_are_preserved(analysis_factory, updated):
    query = "查去年未验收且金额超过 10 元的记录"
    client = Client([candidate([updated])])
    result = await MultiQueryService(client=client, retry_attempts=1).expand(
        QueryAnalysisRequest(query=query), analysis_factory(query=query, multi_query=True), trigger="analyzer",
    )
    assert result.status == "degraded" and len(client.calls) == 1


@pytest.mark.parametrize("kwargs", [
    {"status": "degraded"}, {"path": "clarify"}, {"reference": True},
    {"path": "structured", "intent": "statistics"}, {"intent": "chitchat", "path": "none"},
    {"query_type": "exact"}, {"intent": "other"}, {"exhaustive": True},
    {"query": "HT-2025-001 的情况是什么？"},
])
def test_unsafe_or_non_retrieval_queries_are_blocked(analysis_factory, kwargs):
    assert expansion_block_reason(analysis_factory(**kwargs))


@pytest.mark.asyncio
async def test_hyde_never_combines_with_multi_query(analysis_factory):
    client = Client([])
    result = await MultiQueryService(client=client).expand(QueryAnalysisRequest(query=QUERY),
                                                         analysis_factory(multi_query=True), trigger="empty_retrieval",
                                                         hyde_selected=True)
    assert result.reason == "hyde_not_combined" and not client.calls


@pytest.mark.asyncio
async def test_missing_key_keeps_original(analysis_factory):
    result = await MultiQueryService(api_key="").expand(QueryAnalysisRequest(query=QUERY),
                                                     analysis_factory(multi_query=True), trigger="analyzer")
    assert result.degradation_reason == "missing_api_key" and result.attempts == 0


@pytest.mark.asyncio
async def test_timeout_includes_audit(analysis_factory):
    class SlowAuditClient(Client):
        async def create(self, **kwargs):
            self.delay = 1 if self.calls else 0
            return await super().create(**kwargs)
    client = SlowAuditClient([candidate(), audits()])
    result = await MultiQueryService(client=client, timeout_seconds=0.2).expand(
        QueryAnalysisRequest(query=QUERY), analysis_factory(multi_query=True), trigger="analyzer",
    )
    assert len(client.calls) == 2 and result.degradation_reason == "timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("cls,code,retry", [(AuthenticationError, 401, False), (RateLimitError, 429, True)])
async def test_provider_error_policy_and_no_body_logging(analysis_factory, caplog, cls, code, retry):
    response = httpx.Response(code, request=httpx.Request("POST", "https://api.deepseek.com"))
    client = Client([cls("do-not-log-secret-body", response=response, body=None), candidate(), audits()])
    result = await MultiQueryService(client=client).expand(QueryAnalysisRequest(query=QUERY),
                                                         analysis_factory(multi_query=True), trigger="analyzer")
    assert result.status == ("expanded" if retry else "degraded")
    assert len(client.calls) == (3 if retry else 1) and "do-not-log-secret-body" not in caplog.text


@pytest.mark.asyncio
async def test_cancellation_propagates(analysis_factory):
    client = Client([candidate()])
    client.delay = 10
    task = asyncio.create_task(MultiQueryService(client=client).expand(
        QueryAnalysisRequest(query=QUERY), analysis_factory(multi_query=True), trigger="analyzer",
    ))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
