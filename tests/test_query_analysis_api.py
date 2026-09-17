import httpx
import pytest
from fastapi import FastAPI

from zq_rag_app.api import query_analysis as query_api
from zq_rag_app.core.security import DEFAULT_ADMIN_USER, require_current_user
from zq_rag_app.services.query_analyzer_service import QueryAnalyzerService


@pytest.mark.asyncio
async def test_analysis_route_response_validation_auth_and_cleanup(monkeypatch):
    app = FastAPI()
    app.include_router(query_api.router)
    service = QueryAnalyzerService(api_key="")
    closed = []

    async def close():
        closed.append(True)

    service.aclose = close
    monkeypatch.setattr(query_api, "QueryAnalyzerService", lambda: service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query/analyze", json={"query": "查询合同"})
        assert response.status_code == 401
        app.dependency_overrides[require_current_user] = lambda: DEFAULT_ADMIN_USER
        invalid = await client.post("/query/analyze", json={"query": "   "})
        assert invalid.status_code == 422 and not closed
        response = await client.post("/query/analyze", json={"query": " 查询合同 "})
    assert response.status_code == 200
    data = response.json()
    assert data["original_query"] == "查询合同"
    assert data["status"] == "degraded" and data["degradation_reason"] == "missing_api_key"
    assert set(data) >= {"intent", "query_type", "entities", "keywords", "metadata", "retrieval_strategy"}
    assert data["retrieval_strategy"]["path"] == "hybrid"
    assert closed == [True]


@pytest.mark.asyncio
async def test_cleanup_when_analysis_raises(monkeypatch):
    closed = []

    class Service:
        async def analyze(self, request):
            raise RuntimeError("unexpected bug")

        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(query_api, "QueryAnalyzerService", Service)
    with pytest.raises(RuntimeError):
        await query_api.analyze_query(query_api.QueryAnalysisRequest(query="问题"), DEFAULT_ADMIN_USER)
    assert closed == [True]
