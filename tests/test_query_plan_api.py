"""计划接口的鉴权顺序、资源选择、严格输入和 Analyzer 连接。"""

from contextlib import contextmanager
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.query_analysis.models import QueryAnalysisResult, QueryFeatures
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_routing import router as query_router
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument

IDENTITY = SimpleNamespace(tenant_id=1, user_id=3, member_id=4, department_id=10, clearance=1)
SCOPE = AccessScope(1, 3, (11,), (100,), (PublishedDocument(100, 11, 200),))


class FakeAnalyzer:
    def __init__(self):
        self.calls = []
        self.intent = "fact"

    async def analyze(self, request):
        self.calls.append(request)
        return QueryAnalysisResult(
            status="ok",
            original_query=request.query,
            analysis=QueryFeatures(
                intent=self.intent,
                entities=[],
                keywords=[],
                constraints=[],
                relation_required=False,
                needs_context=False,
                clarification_question=None,
            ),
            fallback_hint=None,
            message=None,
            degradation_reason=None,
            model="test",
            latency_ms=1,
        )


@pytest.fixture
def api(monkeypatch):
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        app_env="test",
        deepseek_api_key="",
    )
    model = FakeAnalyzer()
    app.dependency_overrides[get_analyzer] = lambda: model
    scopes = []

    class Repository:
        def __init__(self, connection, schema):
            pass

        def resolve(self, identity, body):
            scopes.append((identity, body))
            if body.knowledge_base_ids not in (None, ["11"]):
                raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "指定资源不可用或没有读取权限")
            if body.document_ids not in (None, ["100"]):
                raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "指定资源不可用或没有读取权限")
            return SCOPE

    closed = []

    @contextmanager
    def connection(settings):
        try:
            yield object()
        finally:
            closed.append(True)

    monkeypatch.setattr(query_router, "QueryScopeRepository", Repository)
    monkeypatch.setattr(query_router, "database_connection", connection)
    return app, model, scopes, closed


def test_anonymous_request_does_not_resolve_scope_or_call_model(api):
    app, model, scopes, _ = api
    with TestClient(app) as client:
        response = client.post("/api/query/plan", json={"query": "工单如何关闭？"})
    assert response.status_code == 401
    assert not scopes and not model.calls


def test_authenticated_query_returns_typed_plan_and_scope_without_execution(api):
    app, model, scopes, closed = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post(
            "/api/query/plan",
            json={
                "query": "工单如何关闭？",
                "knowledge_base_ids": ["11"],
                "document_ids": ["100"],
            },
        )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    data = response.json()
    assert data["plan"]["route"] == "hybrid"
    assert data["plan"]["planning_status"] == "planned"
    assert data["plan"]["params"] == {"candidate_k": 30, "top_k": 5, "rerank": True}
    assert data["scope"] == {
        "knowledge_base_ids": ["11"],
        "authorized_document_count": 1,
        "published_document_count": 1,
    }
    assert len(scopes) == len(model.calls) == len(closed) == 1
    assert model.calls[0].history == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("knowledge_base_ids", ["12"]),
        ("document_ids", ["101"]),
    ],
)
def test_forbidden_resource_is_rejected_before_model(api, field, value):
    app, model, _, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/plan", json={"query": "问题", field: value})
    assert response.status_code == 403
    assert model.calls == []


@pytest.mark.parametrize(
    "extra",
    [
        {"analysis": {}},
        {"capabilities": {"catalog_available": True}},
        {"scope": {}},
        {"tenant_id": 2},
        {"history": ["他人的问题"]},
        {"session_id": "someone-else"},
        {"params": {"candidate_k": 100}},
        {"graph": {}},
    ],
)
def test_client_cannot_supply_trusted_analysis_or_capabilities(api, extra):
    app, model, _, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/plan", json={"query": "问题", **extra})
    assert response.status_code == 422
    assert not model.calls


@pytest.mark.parametrize(
    "body",
    [
        {"query": " "},
        {"query": "q" * 2001},
        {"query": "问题", "knowledge_base_ids": []},
        {"query": "问题", "knowledge_base_ids": ["11", "11"]},
        {"query": "问题", "knowledge_base_ids": [11]},
        {"query": "问题", "knowledge_base_ids": ["0"]},
        {"query": "问题", "document_ids": ["9223372036854775808"]},
    ],
)
def test_invalid_request_cannot_trigger_model(api, body):
    app, model, _, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/plan", json=body)
    assert response.status_code == 422
    assert not model.calls


def test_database_failure_returns_sanitized_503_before_model(api, monkeypatch):
    app, model, _, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY

    @contextmanager
    def failed(settings):
        raise psycopg.OperationalError("secret database password")
        yield

    monkeypatch.setattr(query_router, "database_connection", failed)
    with TestClient(app) as client:
        response = client.post("/api/query/plan", json={"query": "问题"})
    assert response.status_code == 503
    assert "secret" not in response.text
    assert not model.calls


def test_database_connection_closes_before_model_request(api):
    app, model, _, closed = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    original = model.analyze

    async def analyze(request):
        assert closed == [True]
        return await original(request)

    model.analyze = analyze
    with TestClient(app) as client:
        response = client.post("/api/query/plan", json={"query": "问题"})
    assert response.status_code == 200


def test_structured_endpoint_explicitly_reports_unavailable_executor(api):
    app, model, _, _ = api
    model.intent = "statistics"
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/plan", json={"query": "知识库有多少份文档？"})
    assert response.status_code == 200
    plan = response.json()["plan"]
    assert plan["route"] == "structured"
    assert plan["params"]["operation"] == "count_documents"
    assert plan["planning_status"] == "unsupported"
    assert plan["reason_code"] == "CATALOG_EXECUTOR_NOT_READY"
