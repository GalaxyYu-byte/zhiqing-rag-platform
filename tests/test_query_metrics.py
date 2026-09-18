import asyncio
import json
from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY
from starlette.requests import Request
from starlette.responses import Response

from zq_rag_app.query_analysis.models import QueryAnalysisRequest
from zq_rag_app.query_analysis.prompts import _EXAMPLE
from zq_rag_app.services.query_analyzer_service import QueryAnalyzerService


def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=value))])


@pytest.mark.asyncio
async def test_model_calls_and_actual_retry_are_separate_from_logical_requests():
    before = [sample("zq_query_analyzer_calls_total"),
              sample("zq_query_analyzer_retries_total", reason="invalid_output"),
              sample("zq_query_analyzer_requests_total", status="ok", reason="none"),
              sample("zq_query_analyzer_duration_seconds_count")]
    client = Client(["{}", json.dumps(_EXAMPLE)])
    result = await QueryAnalyzerService(client=client).analyze(QueryAnalysisRequest(query="HT-2025-001 的金额是多少？"))
    assert result.status == "ok" and result.attempts == 2
    after = [sample("zq_query_analyzer_calls_total"),
             sample("zq_query_analyzer_retries_total", reason="invalid_output"),
             sample("zq_query_analyzer_requests_total", status="ok", reason="none"),
             sample("zq_query_analyzer_duration_seconds_count")]
    assert [end - start for start, end in zip(before, after)] == [2, 1, 1, 1]


@pytest.mark.asyncio
async def test_missing_key_counts_degradation_without_a_model_call():
    before_calls = sample("zq_query_analyzer_calls_total")
    before = sample("zq_query_analyzer_requests_total", status="degraded", reason="missing_api_key")
    await QueryAnalyzerService(api_key="").analyze(QueryAnalysisRequest(query="问题"))
    assert sample("zq_query_analyzer_calls_total") == before_calls
    assert sample("zq_query_analyzer_requests_total", status="degraded", reason="missing_api_key") == before + 1


@pytest.mark.asyncio
async def test_cancelled_analysis_propagates_and_records_cancellation():
    before = sample("zq_query_analyzer_requests_total", status="cancelled", reason="cancelled")
    with pytest.raises(asyncio.CancelledError):
        await QueryAnalyzerService(client=Client([asyncio.CancelledError()])).analyze(QueryAnalysisRequest(query="问题"))
    assert sample("zq_query_analyzer_requests_total", status="cancelled", reason="cancelled") == before + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status,outcome", [(200, "success"), (403, "client_error"), (503, "server_error")])
async def test_chat_http_metric_includes_all_response_outcomes(status, outcome):
    from zq_rag_app.main import observe_chat_latency
    request = Request({"type": "http", "method": "POST", "path": "/chat/answer", "headers": []})
    count = sample("zq_chat_http_duration_seconds_count", outcome=outcome)
    async def endpoint(current):
        assert current is request
        return Response(status_code=status)

    response = await observe_chat_latency(request, endpoint)
    assert response.status_code == status
    assert sample("zq_chat_http_duration_seconds_count", outcome=outcome) == count + 1


@pytest.mark.asyncio
async def test_http_exception_and_cancellation_are_included_in_latency_histogram():
    from zq_rag_app.main import observe_chat_latency
    request = Request({"type": "http", "method": "POST", "path": "/chat/answer", "headers": []})
    for error, outcome in [(RuntimeError("error"), "server_error"), (asyncio.CancelledError(), "cancelled")]:
        count = sample("zq_chat_http_duration_seconds_count", outcome=outcome)
        async def endpoint(current):
            raise error
        with pytest.raises(type(error)):
            await observe_chat_latency(request, endpoint)
        assert sample("zq_chat_http_duration_seconds_count", outcome=outcome) == count + 1
