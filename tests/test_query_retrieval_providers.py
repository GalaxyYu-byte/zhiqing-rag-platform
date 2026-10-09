"""检索供应商的载荷、完整性、超时与客户端所有权。"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from zhiqing_rag.core.config import Settings
from zhiqing_rag.query_retrieval.providers import (
    ProviderResponseError,
    ProviderUnavailable,
    QueryEmbedder,
    QueryReranker,
)

QUERY = "工单关闭前需要谁确认？"
DOCUMENTS = ("主管确认关闭条件。", "经办人记录关闭原因。")
VECTOR = [0.1] * 1024


def settings(**changes):
    # 不读取 .env，也显式覆盖可能存在的进程凭据，保证单测只调用伪客户端。
    values = {
        "dashscope_api_key": "",
        "query_embedding_timeout_seconds": 10,
        "query_reranker_timeout_seconds": 10,
        "query_reranker_enabled": True,
    }
    return Settings(_env_file=None, app_env="test", **{**values, **changes})


def embedding_response(*, index=0, vector=None):
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=VECTOR if vector is None else vector)]
    )


class EmbeddingClient:
    def __init__(self, response=None, *, error=None, delay=0, close_error=None):
        self.response = embedding_response() if response is None else response
        self.error = error
        self.delay = delay
        self.close_error = close_error
        self.calls = []
        self.closed = False
        self.embeddings = SimpleNamespace(create=self.create)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.response

    async def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def rerank_response(results=None):
    return {
        "output": {
            "results": results
            if results is not None
            else [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
            ]
        }
    }


class RerankClient:
    def __init__(self, body=None, *, error=None, delay=0, status=200, close_error=None):
        self.body = rerank_response() if body is None else body
        self.error = error
        self.delay = delay
        self.status = status
        self.close_error = close_error
        self.calls = []
        self.closed = False

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        response = httpx.Response(self.status, request=httpx.Request("POST", url))
        # 用回调保留 NaN 等非法供应商响应，以验证生产解析器拒绝它们。
        response.json = lambda: self.body
        return response

    async def aclose(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


async def test_embed_uses_the_index_contract_and_returns_immutable_floats():
    client = EmbeddingClient(embedding_response(vector=[1] * 1024))
    result = await QueryEmbedder(settings(), client).embed(QUERY)
    assert result == (1.0,) * 1024
    assert all(type(value) is float for value in result)
    assert client.calls == [
        {
            "model": "text-embedding-v3",
            "input": [QUERY],
            "dimensions": 1024,
            "encoding_format": "float",
        }
    ]
    assert not client.closed


@pytest.mark.parametrize("index", [-1, 1, True, False, 0.0, "0", None])
async def test_embed_rejects_invalid_or_coerced_indices(index):
    client = EmbeddingClient(embedding_response(index=index))
    with pytest.raises(ProviderResponseError):
        await QueryEmbedder(settings(), client).embed(QUERY)
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(),
        SimpleNamespace(data=None),
        SimpleNamespace(data={"index": 0}),
        SimpleNamespace(data=[]),
        SimpleNamespace(data=[SimpleNamespace(index=0, embedding=VECTOR)] * 2),
        SimpleNamespace(data=[SimpleNamespace(index=0)]),
    ],
)
async def test_embed_requires_exactly_one_complete_result(response):
    with pytest.raises(ProviderResponseError):
        await QueryEmbedder(settings(), EmbeddingClient(response)).embed(QUERY)


@pytest.mark.parametrize(
    "vector",
    [
        [],
        [0.1] * 1023,
        [0.0] * 1024,
        [float("nan")] * 1024,
        [float("inf")] * 1024,
        [True] * 1024,
        ["0.1"] * 1024,
        [4e38] * 1024,
        [10**1000] * 1024,
        [1e-50] * 1024,
        "not a vector",
    ],
)
async def test_embed_rejects_vectors_unsafe_for_pgvector(vector):
    with pytest.raises(ProviderResponseError):
        await QueryEmbedder(settings(), EmbeddingClient(embedding_response(vector=vector))).embed(
            QUERY
        )


async def test_rerank_requests_all_candidates_and_ignores_returned_documents():
    body = rerank_response(
        [
            {"index": 1, "relevance_score": 0.9, "document": {"text": "篡改的原文"}},
            {"index": 0, "relevance_score": 0.2, "document": {"text": "伪造的证据"}},
        ]
    )
    client = RerankClient(body)
    result = await QueryReranker(settings(dashscope_api_key="unit-test-key"), client).rerank(
        QUERY, DOCUMENTS
    )
    assert result == ((1, 0.9), (0, 0.2))
    assert not client.closed
    url, call = client.calls[0]
    assert url == settings().reranker_endpoint
    assert call["json"] == {
        "model": "gte-rerank-v2",
        "input": {"query": QUERY, "documents": list(DOCUMENTS)},
        "parameters": {"return_documents": False, "top_n": 2},
    }
    assert call["timeout"] == 10
    assert call["headers"] == {
        "Authorization": "Bearer unit-test-key",
        "Content-Type": "application/json",
    }


async def test_rerank_equal_scores_sort_by_original_candidate_index():
    client = RerankClient(
        rerank_response([{"index": 1, "relevance_score": 1}, {"index": 0, "relevance_score": 1}])
    )
    assert await QueryReranker(settings(), client).rerank(QUERY, DOCUMENTS) == (
        (0, 1.0),
        (1, 1.0),
    )


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"code": "InvalidApiKey", "message": "secret provider body"},
        {"output": None},
        {"output": {"results": None}},
        {"output": {"results": {"index": 0}}},
        rerank_response([]),
        rerank_response([{"index": 0, "relevance_score": 0.5}]),
        rerank_response([{"index": 0, "relevance_score": 0.5}] * 3),
        rerank_response([{"index": 0, "relevance_score": 0.5}] * 2),
        rerank_response([None, {"index": 1, "relevance_score": 0.5}]),
    ],
)
async def test_rerank_rejects_missing_extra_duplicate_or_failed_results(body):
    client = RerankClient(body)
    with pytest.raises(ProviderResponseError) as raised:
        await QueryReranker(settings(), client).rerank(QUERY, DOCUMENTS)
    assert "secret" not in str(raised.value)
    assert len(client.calls) == 1


@pytest.mark.parametrize("index", [-1, 2, True, False, 0.0, "0", None])
async def test_rerank_indices_are_strict_integers(index):
    client = RerankClient(
        rerank_response(
            [{"index": index, "relevance_score": 0.2}, {"index": 1, "relevance_score": 0.9}]
        )
    )
    with pytest.raises(ProviderResponseError):
        await QueryReranker(settings(), client).rerank(QUERY, DOCUMENTS)


@pytest.mark.parametrize("score", [True, False, "0.2", None, float("nan"), float("inf"), 10**1000])
async def test_rerank_scores_are_numeric_finite_and_not_coerced(score):
    client = RerankClient(
        rerank_response(
            [{"index": 0, "relevance_score": score}, {"index": 1, "relevance_score": 0.9}]
        )
    )
    with pytest.raises(ProviderResponseError):
        await QueryReranker(settings(), client).rerank(QUERY, DOCUMENTS)


@pytest.mark.parametrize("status", [301, 400, 401, 429, 500])
async def test_rerank_http_failures_are_not_retried(status):
    client = RerankClient(status=status)
    with pytest.raises(httpx.HTTPStatusError):
        await QueryReranker(settings(), client).rerank(QUERY, DOCUMENTS)
    assert len(client.calls) == 1


@pytest.mark.parametrize("query", [None, 12, "", "  "])
async def test_invalid_query_fails_before_any_provider_call(query):
    embed_client, rank_client = EmbeddingClient(), RerankClient()
    with pytest.raises(ValueError):
        await QueryEmbedder(settings(), embed_client).embed(query)
    with pytest.raises(ValueError):
        await QueryReranker(settings(), rank_client).rerank(query, DOCUMENTS)
    assert embed_client.calls == rank_client.calls == []


@pytest.mark.parametrize("documents", [["正文"], (None,), (1,), ("",), ("  ",)])
async def test_rerank_rejects_invalid_candidate_text_before_http_call(documents):
    client = RerankClient()
    with pytest.raises(ValueError):
        await QueryReranker(settings(), client).rerank(QUERY, documents)
    assert client.calls == []


async def test_empty_rerank_candidates_do_not_need_credentials_or_make_calls():
    client = RerankClient()
    assert await QueryReranker(settings(), client).rerank(QUERY, ()) == ()
    assert client.calls == []


async def test_missing_credentials_and_disabled_reranker_fail_before_client_creation(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("不应创建供应商客户端")

    monkeypatch.setattr("zhiqing_rag.query_retrieval.providers.AsyncOpenAI", forbidden)
    monkeypatch.setattr("zhiqing_rag.query_retrieval.providers.httpx.AsyncClient", forbidden)
    with pytest.raises(ProviderUnavailable) as raised:
        await QueryEmbedder(settings()).embed(QUERY)
    assert raised.value.code == "EMBEDDING_NOT_CONFIGURED"
    with pytest.raises(ProviderUnavailable) as raised:
        await QueryReranker(settings()).rerank(QUERY, DOCUMENTS)
    assert raised.value.code == "RERANKER_NOT_CONFIGURED"
    client = RerankClient()
    with pytest.raises(ProviderUnavailable) as raised:
        await QueryReranker(settings(query_reranker_enabled=False), client).rerank(QUERY, DOCUMENTS)
    assert raised.value.code == "RERANKER_DISABLED"
    assert client.calls == []


async def test_dimension_mismatch_does_not_request_an_incompatible_vector():
    client = EmbeddingClient()
    with pytest.raises(ProviderUnavailable) as raised:
        await QueryEmbedder(settings(embedding_dimensions=512), client).embed(QUERY)
    assert raised.value.code == "EMBEDDING_DIMENSIONS_UNSUPPORTED"
    assert client.calls == []


@pytest.mark.parametrize(
    "changes,provider,code",
    [
        ({"embedding_model": " "}, "embed", "EMBEDDING_NOT_CONFIGURED"),
        ({"openai_base_url": " "}, "embed", "EMBEDDING_NOT_CONFIGURED"),
        ({"reranker_model": " "}, "rerank", "RERANKER_NOT_CONFIGURED"),
        ({"reranker_endpoint": " "}, "rerank", "RERANKER_NOT_CONFIGURED"),
    ],
)
async def test_blank_model_or_endpoint_is_explicitly_unavailable(changes, provider, code):
    with pytest.raises(ProviderUnavailable) as raised:
        if provider == "embed":
            await QueryEmbedder(settings(**changes), EmbeddingClient()).embed(QUERY)
        else:
            await QueryReranker(settings(**changes), RerankClient()).rerank(QUERY, DOCUMENTS)
    assert raised.value.code == code


async def test_injected_clients_obey_the_full_call_timeout_and_are_not_closed():
    embed_client, rank_client = EmbeddingClient(delay=1), RerankClient(delay=1)
    with pytest.raises(TimeoutError):
        await QueryEmbedder(settings(query_embedding_timeout_seconds=0.01), embed_client).embed(
            QUERY
        )
    with pytest.raises(TimeoutError):
        await QueryReranker(settings(query_reranker_timeout_seconds=0.01), rank_client).rerank(
            QUERY, DOCUMENTS
        )
    assert not embed_client.closed
    assert not rank_client.closed
    assert len(embed_client.calls) == len(rank_client.calls) == 1


async def test_injected_clients_propagate_cancellation():
    embed_client = EmbeddingClient(error=asyncio.CancelledError())
    rank_client = RerankClient(error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await QueryEmbedder(settings(), embed_client).embed(QUERY)
    with pytest.raises(asyncio.CancelledError):
        await QueryReranker(settings(), rank_client).rerank(QUERY, DOCUMENTS)
    assert not embed_client.closed
    assert not rank_client.closed


@pytest.mark.parametrize("outcome", ["success", "invalid", "error", "timeout", "cancel"])
async def test_owned_embedding_client_is_closed_on_every_outcome(monkeypatch, outcome):
    client = EmbeddingClient(
        response=SimpleNamespace(data=[]) if outcome == "invalid" else None,
        error=RuntimeError("secret request body") if outcome == "error" else None,
        delay=1 if outcome == "timeout" else 0,
    )
    if outcome == "cancel":
        client.error = asyncio.CancelledError()
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return client

    monkeypatch.setattr("zhiqing_rag.query_retrieval.providers.AsyncOpenAI", factory)
    service = QueryEmbedder(
        settings(dashscope_api_key="unit-test-key", query_embedding_timeout_seconds=0.01)
    )
    if outcome == "success":
        assert await service.embed(QUERY) == tuple(VECTOR)
    else:
        error = {
            "invalid": ProviderResponseError,
            "error": RuntimeError,
            "timeout": TimeoutError,
            "cancel": asyncio.CancelledError,
        }[outcome]
        with pytest.raises(error):
            await service.embed(QUERY)
    assert client.closed
    assert created[0] == {
        "api_key": "unit-test-key",
        "base_url": settings().openai_base_url,
        "timeout": 0.01,
        "max_retries": 0,
    }
    assert len(client.calls) == 1


@pytest.mark.parametrize("outcome", ["success", "invalid", "error", "timeout", "cancel"])
async def test_owned_rerank_client_is_closed_on_every_outcome(monkeypatch, outcome):
    client = RerankClient(
        body=rerank_response([]) if outcome == "invalid" else None,
        error=RuntimeError("secret request body") if outcome == "error" else None,
        delay=1 if outcome == "timeout" else 0,
    )
    if outcome == "cancel":
        client.error = asyncio.CancelledError()
    created = []

    def factory(**kwargs):
        created.append(kwargs)
        return client

    monkeypatch.setattr("zhiqing_rag.query_retrieval.providers.httpx.AsyncClient", factory)
    service = QueryReranker(
        settings(dashscope_api_key="unit-test-key", query_reranker_timeout_seconds=0.01)
    )
    if outcome == "success":
        assert await service.rerank(QUERY, DOCUMENTS) == ((1, 0.9), (0, 0.2))
    else:
        error = {
            "invalid": ProviderResponseError,
            "error": RuntimeError,
            "timeout": TimeoutError,
            "cancel": asyncio.CancelledError,
        }[outcome]
        with pytest.raises(error):
            await service.rerank(QUERY, DOCUMENTS)
    assert client.closed
    assert created[0] == {"timeout": 0.01, "follow_redirects": False}
    assert len(client.calls) == 1


async def test_close_failure_does_not_replace_valid_response_or_log_credentials(
    monkeypatch, caplog
):
    client = EmbeddingClient(close_error=RuntimeError("unit-test-key secret query"))
    monkeypatch.setattr(
        "zhiqing_rag.query_retrieval.providers.AsyncOpenAI", lambda **kwargs: client
    )
    assert await QueryEmbedder(settings(dashscope_api_key="unit-test-key")).embed(QUERY) == tuple(
        VECTOR
    )
    assert "RuntimeError" in caplog.text
    assert "unit-test-key" not in caplog.text
    assert "secret query" not in caplog.text


async def test_each_noninjected_request_creates_and_closes_its_own_client(monkeypatch):
    clients = []

    def factory(**kwargs):
        client = EmbeddingClient()
        clients.append(client)
        return client

    monkeypatch.setattr("zhiqing_rag.query_retrieval.providers.AsyncOpenAI", factory)
    service = QueryEmbedder(settings(dashscope_api_key="unit-test-key"))
    await service.embed(QUERY)
    await service.embed(QUERY)
    assert len(clients) == 2
    assert all(client.closed for client in clients)
