"""图增强计划由服务端生成，JSON、SSE 和共享回答链使用同一规划检查。"""

import asyncio
from types import SimpleNamespace

import pytest
import test_query_search_api as search_api
from fastapi.testclient import TestClient
from test_query_answer_api import Answerer, stream_frames
from test_query_search_api import IDENTITY, SCOPE, Analyzer, Retriever

from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.query_analysis.models import GroundedTerm, QueryFeatures
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_answering.router import execute_answer, get_answerer
from zhiqing_rag.query_retrieval.router import get_retriever
from zhiqing_rag.query_routing.models import (
    GraphParams,
    GraphProjectionRef,
    HybridPlan,
    PlanQueryBody,
    RetrievalParams,
)
from zhiqing_rag.query_routing.planning import get_graph_service
from zhiqing_rag.query_routing.service import GraphReadiness, QueryRouterService

ENDPOINTS = (
    "/api/query/plan",
    "/api/query/search",
    "/api/query/answer",
    "/api/query/answer/stream",
)
QUERY = "工单由谁确认关闭？"


class GraphAnalyzer(Analyzer):
    async def analyze(self, request):
        result = await super().analyze(request)
        return result.model_copy(
            update={
                "analysis": QueryFeatures(
                    intent="relationship",
                    entities=[
                        GroundedTerm.model_validate(
                            {
                                "text": "工单",
                                "evidence": {
                                    "source": "query",
                                    "history_index": None,
                                    "quote": "工单",
                                },
                            }
                        )
                    ],
                    keywords=[],
                    constraints=[],
                    relation_required=True,
                    needs_context=False,
                    clarification_question=None,
                )
            }
        )


class GraphService:
    def __init__(self):
        self.settings = SimpleNamespace(query_graph_timeout_seconds=0.05)
        self.calls = []
        self.error = None
        self.ready = True
        self.wait_forever = False
        self.cancelled = False
        self.invalid_projection = False

    async def readiness(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if self.wait_forever:
            try:
                await asyncio.Future()
            finally:
                self.cancelled = True
        if not self.ready:
            return None
        return GraphReadiness(
            query=kwargs["query"],
            matched_entities=kwargs["entities"],
            params=GraphParams(
                **kwargs["retrieval_params"].model_dump(),
                seed_entity_uids=("entity-1",),
                projections=(
                    GraphProjectionRef(
                        tenant_id="2" if self.invalid_projection else "1",
                        document_id="100",
                        generation_id="200",
                        projection_id="400",
                        namespace="test-graph",
                    ),
                ),
                max_hops=1,
            ),
        )


class GraphRetriever(Retriever):
    """HTTP 接线替身；底层图召回和融合另由服务测试验证。"""

    def __init__(self, graph_service):
        super().__init__()
        self.graph_service = graph_service

    async def execute(self, **kwargs):
        original = kwargs["plan"]
        if original.route != "hybrid_graph":
            return await super().execute(**kwargs)
        base = HybridPlan(
            **original.model_dump(exclude={"route", "params"}),
            params=RetrievalParams(
                candidate_k=original.params.candidate_k,
                top_k=original.params.top_k,
                rerank=original.params.rerank,
            ),
        )
        result = await super().execute(**{**kwargs, "plan": base})
        self.calls[-1] = kwargs
        return result


@pytest.fixture
def api():
    for app, _, _ in search_api.api.__wrapped__():
        analyzer, graph, answerer = GraphAnalyzer(), GraphService(), Answerer()
        retriever = GraphRetriever(graph)
        app.dependency_overrides[require_identity] = lambda: IDENTITY
        app.dependency_overrides[get_analyzer] = lambda: analyzer
        app.dependency_overrides[get_retriever] = lambda: retriever
        app.dependency_overrides[get_graph_service] = lambda: graph
        app.dependency_overrides[get_answerer] = lambda: answerer
        yield app, analyzer, graph, retriever, answerer


def result_body(response, endpoint):
    if endpoint.endswith("/stream"):
        return stream_frames(response)[-1][1]
    return response.json()


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_server_readiness_approves_graph_route_in_all_query_endpoints(api, endpoint):
    app, _, graph, retriever, answerer = api
    with TestClient(app) as client:
        response = client.post(endpoint, json={"query": QUERY, "document_ids": ["100"]})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    result = result_body(response, endpoint)
    assert result["plan"]["route"] == "hybrid_graph"
    assert result["plan"]["reason_code"] == "GRAPH_APPROVED"
    assert result["plan"]["params"]["max_hops"] == 1
    assert result["plan"]["params"]["projections"][0]["generation_id"] == "200"
    assert len(graph.calls) == 1
    assert graph.calls[0]["identity"] is IDENTITY and graph.calls[0]["scope"] is SCOPE
    assert graph.calls[0]["entities"] == ("工单",)
    if endpoint != "/api/query/plan":
        assert retriever.calls[0]["plan"].route == "hybrid_graph"
    if "/answer" in endpoint:
        assert answerer.calls[0]["plan"].route == "hybrid_graph"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("failure", ["not_ready", "unavailable", "timeout", "invalid_projection"])
def test_graph_planning_degrades_safely_without_losing_original_query(api, endpoint, failure):
    app, _, graph, retriever, _ = api
    graph.ready = failure != "not_ready"
    graph.error = RuntimeError("private-neo4j-payload") if failure == "unavailable" else None
    graph.wait_forever = failure == "timeout"
    graph.invalid_projection = failure == "invalid_projection"
    with TestClient(app) as client:
        response = client.post(endpoint, json={"query": QUERY})
    assert response.status_code == 200
    plan = result_body(response, endpoint)["plan"]
    assert plan["route"] == "hybrid" and plan["requested_route"] == "hybrid_graph"
    assert plan["reason_code"] == (
        "GRAPH_NOT_READY" if failure == "not_ready" else "GRAPH_UNAVAILABLE"
    )
    assert plan["query"] == plan["original_query"] == QUERY
    assert plan["warnings"]
    assert "private-neo4j-payload" not in response.text
    assert graph.cancelled is (failure == "timeout")
    if endpoint != "/api/query/plan":
        assert retriever.calls[0]["plan"].route == "hybrid"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize(
    "extra",
    [
        {"route": "hybrid_graph"},
        {"graph": {}},
        {"params": {"max_hops": 2}},
        {"seed_entity_uids": ["hidden-entity"]},
        {"projections": [{"tenant_id": "2"}]},
    ],
)
def test_client_cannot_choose_graph_seed_projection_or_budget(api, endpoint, extra):
    app, analyzer, graph, retriever, answerer = api
    with TestClient(app) as client:
        response = client.post(endpoint, json={"query": QUERY, **extra})
    assert response.status_code == 422
    assert not analyzer.calls and not graph.calls and not retriever.calls and not answerer.calls


@pytest.mark.parametrize("endpoint", ENDPOINTS)
@pytest.mark.parametrize("status,code", [(401, "SESSION_REVOKED"), (403, "QUERY_SCOPE_FORBIDDEN")])
def test_graph_readiness_permission_failure_is_never_swallowed(api, endpoint, status, code):
    app, _, graph, retriever, answerer = api
    graph.error = AuthError(status, code, "当前请求没有读取权限")
    with TestClient(app) as client:
        response = client.post(endpoint, json={"query": QUERY})
    if endpoint.endswith("/stream"):
        frames = stream_frames(response)
        assert [name for name, _ in frames] == ["progress", "error"]
        assert frames[-1][1]["code"] == code and frames[-1][1]["status"] == status
    else:
        assert response.status_code == status
        assert response.json()["detail"]["code"] == code
    assert not retriever.calls and not answerer.calls


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_non_relation_query_does_not_check_graph(api, endpoint):
    app, _, graph, _, _ = api
    app.dependency_overrides[get_analyzer] = Analyzer
    with TestClient(app) as client:
        response = client.post(endpoint, json={"query": QUERY})
    assert response.status_code == 200
    assert result_body(response, endpoint)["plan"]["route"] == "hybrid"
    assert not graph.calls


async def test_shared_execute_answer_used_by_chat_also_approves_graph(api):
    _, analyzer, graph, retriever, answerer = api
    result = await execute_answer(
        PlanQueryBody(query=QUERY),
        IDENTITY,
        SCOPE,
        analyzer,
        QueryRouterService(),
        retriever,
        answerer,
    )
    assert result.plan.route == "hybrid_graph"
    assert len(graph.calls) == len(retriever.calls) == len(answerer.calls) == 1
