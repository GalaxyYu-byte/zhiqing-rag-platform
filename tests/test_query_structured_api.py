"""精确目录与原文回答分流；过滤、空结果、分页和流式响应的公共契约。"""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
import test_query_search_api as search_api
from fastapi.testclient import TestClient
from test_query_answer_api import stream_frames

from zhiqing_rag.modules.chat.schemas import ChatAnswerBody
from zhiqing_rag.modules.chat.service import canonical_request, request_digest
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.query_answering.models import AnswerResult
from zhiqing_rag.query_answering.router import get_answerer
from zhiqing_rag.query_routing.router import get_access_scope
from zhiqing_rag.query_structured.models import StructuredDocument, StructuredResult


class Catalog:
    def __init__(self):
        self.calls = []
        self.valid = True
        self.total = 3

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        plan, scope = kwargs["plan"], kwargs["scope"]
        documents = tuple(
            StructuredDocument(
                knowledge_base_id="11",
                document_id=str(100 + index),
                revision_id=str(150 + index),
                generation_id=str(200 + index),
                title=f"制度{index}",
                document_name=f"制度{index}.pdf",
                file_type="application/pdf",
                business_status="EFFECTIVE",
                department_id=None,
                event_date=None,
                uploaded_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
            for index in range(self.total)
        )
        return StructuredResult(
            operation=plan.params.operation,
            total=self.total,
            offset=plan.params.offset,
            limit=plan.params.limit,
            documents=documents[plan.params.offset : plan.params.offset + plan.params.limit]
            if plan.params.operation == "list_documents"
            else (),
            applied_filters=scope.resolved_filters,
            executed_at=datetime.now(UTC),
        )

    async def revalidate(self, **kwargs):
        return self.valid


@pytest.fixture
def api():
    for app, analyzer, retriever in search_api.api.__wrapped__():
        catalog = Catalog()
        retriever.structured_service = catalog
        analyzer.intent = "statistics"
        app.dependency_overrides[require_identity] = lambda: search_api.IDENTITY

        class NoModelAnswer:
            async def execute(self, **kwargs):
                if kwargs["plan"].route == "clarify":
                    return kwargs["retrieval"], AnswerResult(status="skipped")
                pytest.fail("精确统计不能调用原文生成模型")

        app.dependency_overrides[get_answerer] = NoModelAnswer
        yield app, analyzer, retriever, catalog


@pytest.mark.parametrize("endpoint", ["/api/query/search", "/api/query/answer"])
def test_catalog_applies_explicit_filters_before_execution(api, endpoint):
    app, _, retriever, catalog = api
    with TestClient(app) as client:
        response = client.post(
            endpoint,
            json={
                "query": "文档有多少份？",
                "knowledge_base_ids": ["11"],
                "filters": [{"field": "file_type", "operator": "eq", "value": "PDF"}],
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["structured"]["total"] == 3
    assert data["structured"]["applied_filters"] == [
        {"field": "file_type", "operator": "eq", "values": ["application/pdf"]}
    ]
    assert catalog.calls[0]["scope"].resolved_filters == catalog.calls[0]["plan"].resolved_filters
    assert not retriever.calls
    assert "provenance" not in data["structured"]
    if endpoint.endswith("answer"):
        assert data["answer"]["status"] == "structured"
        assert data["answer"]["statements"] == []


def test_empty_authorized_catalog_returns_successful_zero(api):
    app, _, _, catalog = api
    catalog.total = 0
    app.dependency_overrides[get_access_scope] = lambda: replace(
        search_api.SCOPE,
        document_ids=(),
        published_documents=(),
    )
    with TestClient(app) as client:
        response = client.post("/api/query/answer", json={"query": "文档有多少份？"})
    assert response.status_code == 200
    assert response.json()["structured"]["total"] == 0
    assert response.json()["answer"]["status"] == "structured"
    assert "0 份" in response.json()["answer"]["message"]


def test_catalog_page_keeps_independent_total_and_replay_request(api):
    app, _, _, _ = api
    with TestClient(app) as client:
        response = client.post(
            "/api/query/search",
            json={
                "query": "列出所有文档",
                "offset": 2,
                "limit": 1,
                "knowledge_base_ids": ["11"],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["structured"]["total"] == 3
    assert len(body["structured"]["documents"]) == 1
    assert body["structured"]["documents"][0]["document_id"] == "102"
    assert body["request"]["knowledge_base_ids"] == ["11"]
    assert body["request"]["offset"] == 2


def test_unresolvable_filter_does_not_execute_unfiltered_catalog(api):
    app, _, retriever, catalog = api
    with TestClient(app) as client:
        response = client.post(
            "/api/query/answer",
            json={
                "query": "文档有多少份？",
                "filters": [{"field": "uploaded_at", "operator": "eq", "value": "最近"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["plan"]["route"] == "clarify"
    assert not catalog.calls
    # 原文执行器只处理澄清计划；没有任何目录结果。
    assert all(call["plan"].route == "clarify" for call in retriever.calls)


def test_scope_change_rejects_statistics_before_publication(api):
    app, _, _, catalog = api
    catalog.valid = False
    with TestClient(app) as client:
        response = client.post("/api/query/answer", json={"query": "文档有多少份？"})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "STRUCTURED_SCOPE_CHANGED"


def test_statistical_stream_publishes_one_complete_result(api):
    app, _, _, catalog = api
    catalog.total = 0
    with TestClient(app) as client:
        response = client.post("/api/query/answer/stream", json={"query": "文档有多少份？"})
    frames = stream_frames(response)
    assert [name for name, _ in frames] == ["progress", "progress", "answer", "done"]
    assert frames[-2][1] == frames[-1][1]
    assert frames[-1][1]["answer"]["status"] == "structured"


def test_chat_filter_and_page_changes_are_part_of_idempotency_digest():
    first = ChatAnswerBody.model_validate_json(
        '{"query":"列出所有文档","client_request_id":"same",'
        '"filters":[{"field":"file_type","operator":"eq","value":"PDF"}]}'
    )
    original = request_digest(canonical_request(first))
    assert original != request_digest(canonical_request(first.model_copy(update={"offset": 20})))
    other = ChatAnswerBody.model_validate_json(
        '{"query":"列出所有文档","client_request_id":"same",'
        '"filters":[{"field":"file_type","operator":"eq","value":"TXT"}]}'
    )
    assert original != request_digest(canonical_request(other))
