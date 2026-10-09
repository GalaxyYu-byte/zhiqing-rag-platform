"""检索入口在鉴权、分析、计划之后执行，不接受客户端计划或权限。"""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.query_analysis.models import QueryAnalysisResult, QueryFeatures
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_retrieval.models import Evidence, EvidenceScores, RetrievalResult
from zhiqing_rag.query_retrieval.router import get_retriever
from zhiqing_rag.query_routing.router import get_access_scope
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument

IDENTITY = SimpleNamespace(tenant_id=1, user_id=3)
SCOPE = AccessScope(1, 3, (11,), (100,), (PublishedDocument(100, 11, 200),))
ORIGINAL_CONTENT = "  工单关闭前需由负责人确认。\n"


class Analyzer:
    intent = "procedure"

    def __init__(self):
        self.calls = []

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
            model="fake",
            latency_ms=1,
        )


class Retriever:
    def __init__(self):
        self.calls = []
        self.failure = None

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        if kwargs["plan"].route != "hybrid":
            return RetrievalResult(status="unsupported", message="尚未支持")
        return RetrievalResult(
            status="ok",
            evidence=(
                Evidence(
                    citation_id="E1",
                    rank=1,
                    chunk_id="300",
                    knowledge_base_id="11",
                    document_id="100",
                    revision_id="150",
                    generation_id="200",
                    document_title="工单管理",
                    document_name="工单管理.pdf",
                    chunk_index=0,
                    content=ORIGINAL_CONTENT,
                    token_count=20,
                    page_number=2,
                    section_title="关闭流程",
                    scores=EvidenceScores(dense_rank=1, dense_score=0.8, fusion_score=1 / 61),
                ),
            ),
        )


@pytest.fixture
def api():
    app = create_app()
    analyzer, retriever = Analyzer(), Retriever()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        app_env="test",
        deepseek_api_key="",
        dashscope_api_key="",
    )
    app.dependency_overrides[get_analyzer] = lambda: analyzer
    app.dependency_overrides[get_retriever] = lambda: retriever

    def scope(body, identity):
        if body.document_ids not in (None, ["100"]):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "指定资源不可用或没有读取权限")
        return SCOPE

    # 保留正式 scope 依赖和正式鉴权顺序，只替换数据仓储的查询。
    from contextlib import contextmanager

    from zhiqing_rag.query_routing import router as module

    class Repository:
        def __init__(self, *args):
            pass

        resolve = staticmethod(lambda identity, body: scope(body, identity))

    @contextmanager
    def connection(_settings):
        yield object()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "QueryScopeRepository", Repository)
        patch.setattr(module, "database_connection", connection)
        yield app, analyzer, retriever


def test_search_requires_login_before_any_model_or_database(api):
    app, analyzer, retriever = api
    # 匿名鉴权失败意味着被替换的仓储也不会执行。
    app.dependency_overrides[get_access_scope] = lambda: pytest.fail("不应查询权限范围")
    with TestClient(app) as client:
        result = client.post("/api/query/search", json={"query": "工单如何关闭？"})
    assert result.status_code == 401
    assert not analyzer.calls and not retriever.calls


def test_search_returns_plan_and_original_evidence(api):
    app, analyzer, retriever = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        result = client.post(
            "/api/query/search",
            json={
                "query": "工单如何关闭？",
                "knowledge_base_ids": ["11"],
                "document_ids": ["100"],
            },
        )
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    data = result.json()
    assert data["plan"]["params"] == {"candidate_k": 30, "top_k": 5, "rerank": True}
    evidence = data["retrieval"]["evidence"][0]
    assert evidence["content"] == ORIGINAL_CONTENT
    assert evidence["document_name"] == "工单管理.pdf"
    assert evidence["page_number"] == 2 and evidence["chunk_id"] == "300"
    assert "answer" not in data
    assert retriever.calls[0]["scope"] is SCOPE and retriever.calls[0]["identity"] is IDENTITY
    assert analyzer.calls[0].history == []


def test_search_forbidden_selection_stops_before_analysis_and_execution(api):
    app, analyzer, retriever = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        result = client.post("/api/query/search", json={"query": "问题", "document_ids": ["999"]})
    assert result.status_code == 403
    assert not analyzer.calls and not retriever.calls


@pytest.mark.parametrize(
    "extra",
    [
        {"plan": {"route": "hybrid"}},
        {"scope": {}},
        {"tenant_id": "2"},
        {"params": {"candidate_k": 100}},
        {"history": ["他人问题"]},
    ],
)
def test_search_rejects_client_execution_and_auth_controls(api, extra):
    app, analyzer, retriever = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        result = client.post("/api/query/search", json={"query": "问题", **extra})
    assert result.status_code == 422
    assert not analyzer.calls and not retriever.calls


def test_structured_plan_is_preserved_in_response(api):
    app, analyzer, _ = api
    analyzer.intent = "statistics"
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        result = client.post("/api/query/search", json={"query": "知识库有多少份文档？"})
    assert result.status_code == 200
    data = result.json()
    assert data["plan"]["route"] == "structured"
    assert data["retrieval"]["status"] == "unsupported"
    assert data["retrieval"]["evidence"] == []


@pytest.mark.parametrize(
    "error",
    [
        AuthError(503, "RETRIEVAL_UNAVAILABLE", "检索服务暂不可用，请稍后重试"),
        AuthError(401, "SESSION_REVOKED", "登录已失效，请重新登录"),
    ],
)
def test_execution_failure_uses_authenticated_error_envelope(api, error):
    app, _, retriever = api
    retriever.failure = error
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        result = client.post("/api/query/search", json={"query": "问题"})
    assert result.status_code == error.status_code
    assert result.headers["cache-control"] == "no-store"
    assert result.json()["detail"]["code"] == error.code
