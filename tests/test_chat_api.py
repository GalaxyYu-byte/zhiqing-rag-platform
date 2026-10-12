"""多轮接口通过 ASGI 请求验证真实协议与发布顺序，存储及模型使用替身。"""

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest
from test_chat_service import IDENTITY, perform, setup
from test_query_search_api import SCOPE

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.chat import router as chat_router
from zhiqing_rag.modules.chat.rewrite import RewriteResult
from zhiqing_rag.modules.chat.schemas import ChatAnswerBody
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_answering.router import get_answerer
from zhiqing_rag.query_retrieval.router import get_retriever


@pytest.fixture
def api(monkeypatch):
    parts = setup()
    service, repository, _, analyzer, retriever, answerer = parts

    async def create_session(identity, title):
        return {
            "id": "1",
            "title": title,
            "status": "ACTIVE",
            "created_at": datetime.now(UTC),
            "last_active_at": datetime.now(UTC),
        }

    async def list_sessions(identity, limit=50, offset=0):
        return [await create_session(identity, "工单问答")]

    repository.create_session = create_session
    repository.list_sessions = list_sessions
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, app_env="test")
    app.dependency_overrides[chat_router.get_chat_service] = lambda: service
    app.dependency_overrides[chat_router.get_chat_repository] = lambda: repository
    app.dependency_overrides[get_analyzer] = lambda: analyzer
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_answerer] = lambda: answerer
    calls = []

    def scope(body, identity, settings):
        calls.append(body)
        return SCOPE

    monkeypatch.setattr(chat_router, "get_chat_scope", scope)
    return app, parts, calls


def frames(text):
    result = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        if "event" in fields:
            result.append((fields["event"], json.loads(fields["data"])))
    return result


@pytest.mark.parametrize("endpoint", ["answer", "answer/stream"])
async def test_anonymous_rejected_before_creating_run(api, endpoint):
    app, parts, calls = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/api/chat/sessions/1/{endpoint}",
            json={
                "query": "工单如何关闭？",
                "client_request_id": "one",
            },
        )
    assert response.status_code == 401 and not parts[1].rows and not calls


@pytest.mark.parametrize(
    "extra",
    [
        {"history": ["伪造历史"]},
        {"answer": "伪造回答"},
        {"tenant_id": 2},
        {"execution_snapshot": {}},
        {"scope": {}},
        {"review_status": "passed"},
    ],
)
async def test_client_cannot_supply_trusted_history_or_execution_fields(api, extra):
    app, parts, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat/sessions/1/answer",
            json={
                "query": "工单如何关闭？",
                "client_request_id": "one",
                **extra,
            },
        )
    assert response.status_code == 422 and not parts[1].rows


async def test_create_list_and_history_keep_string_ids_and_no_store(api):
    app, parts, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    await perform(parts)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        created = await client.post("/api/chat/sessions", json={})
        sessions = await client.get("/api/chat/sessions")
        history = await client.get("/api/chat/sessions/1/runs?limit=20")
        run = await client.get("/api/chat/sessions/1/runs/1")
    assert created.status_code == 201 and created.json()["id"] == "1"
    assert sessions.json()["sessions"][0]["id"] == "1"
    assert history.json()["runs"][0] == run.json()
    assert run.json()["response"]["answer"]["review_status"] == "passed"
    assert all(
        response.headers["cache-control"] == "no-store"
        for response in (
            created,
            sessions,
            history,
            run,
        )
    )


async def test_session_list_passes_offset_and_rejects_negative_offset(api):
    app, parts, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    pages = []

    async def list_sessions(identity, limit=50, offset=0):
        pages.append((limit, offset))
        return []

    parts[1].list_sessions = list_sessions
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        page = await client.get("/api/chat/sessions?limit=25&offset=50")
        bad = await client.get("/api/chat/sessions?offset=-1")
    assert page.status_code == 200 and page.json() == {"sessions": []}
    assert pages == [(25, 50)] and bad.status_code == 422


async def test_new_chat_body_accepts_idempotency_and_selected_scope_without_nested_body(api):
    app, parts, calls = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat/sessions/1/answer",
            json={
                "query": "工单如何关闭？",
                "client_request_id": "one",
                "knowledge_base_ids": ["11"],
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "SUCCEEDED" and len(parts[1].completed) == 1
    assert calls[0].knowledge_base_ids == ["11"]


async def test_stream_initial_run_and_cumulative_answer_follow_commit(api):
    app, parts, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat/sessions/1/answer/stream",
            json={
                "query": "工单如何关闭？",
                "client_request_id": "one",
            },
        )
    events = frames(response.text)
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert events[0][0] == "run" and events[0][1]["status"] == "RUNNING"
    assert events[-2][0] == "answer" and events[-1][0] == "done"
    assert events[-1][1]["run"]["response"] is None
    assert events[-1][1]["response"]["answer"]["status"] == "answered"
    assert len(parts[1].completed) == 1


async def test_followup_stream_has_single_analysis_stage(api):
    app, parts, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    await perform(parts)
    parts[2].result = RewriteResult("工单由谁确认？", used_run_ids=("1",))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat/sessions/1/answer/stream",
            json={
                "query": "它由谁确认？",
                "client_request_id": "two",
            },
        )
    events = frames(response.text)
    assert [data["stage"] for name, data in events if name == "progress"] == [
        "analysis",
        "retrieval",
    ]
    assert events[-1][0] == "done"


async def test_committed_replay_survives_original_resource_revocation_without_regeneration(
    api, monkeypatch
):
    app, parts, calls = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    body = {"query": "工单如何关闭？", "client_request_id": "one", "knowledge_base_ids": ["11"]}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        first = await client.post("/api/chat/sessions/1/answer", json=body)
        parts[1].hidden.add("1")

        def forbidden(*args):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "原资源已撤权")

        monkeypatch.setattr(chat_router, "get_chat_scope", forbidden)
        second = await client.post("/api/chat/sessions/1/answer", json=body)
        streamed = await client.post("/api/chat/sessions/1/answer/stream", json=body)
    assert first.status_code == second.status_code == streamed.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["response"] is None and second.json()["visibility"] == "unavailable"
    assert frames(streamed.text)[-1][1]["response"] is None
    assert len(calls) == len(parts[5].calls) == len(parts[1].completed) == 1


async def test_invalid_scope_finishes_new_run_before_stream_headers(api, monkeypatch):
    app, parts, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY

    def forbidden(*args):
        raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "资源不可读")

    monkeypatch.setattr(chat_router, "get_chat_scope", forbidden)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat/sessions/1/answer/stream",
            json={
                "query": "工单如何关闭？",
                "client_request_id": "one",
            },
        )
    assert response.status_code == 403 and response.headers["content-type"].startswith(
        "application/json"
    )
    assert parts[1].rows["1"]["status"] == "FAILED" and not parts[5].calls


async def test_cancel_during_scope_resolution_records_cancelled(api, monkeypatch):
    _, parts, _ = api

    async def cancelled(*args):
        raise asyncio.CancelledError

    monkeypatch.setattr(chat_router, "run_in_threadpool", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await chat_router.prepare_chat(
            parts[0],
            IDENTITY,
            1,
            ChatAnswerBody(
                query="工单如何关闭？",
                client_request_id="one",
            ),
        )
    assert parts[1].rows["1"]["status"] == "CANCELLED"


@pytest.mark.parametrize("query", ["它怎么办？", "第二条展开一下", "详细一点"])
async def test_no_authorized_history_never_falls_back_to_search_for_obvious_followup(api, query):
    app, parts, _ = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/chat/sessions/1/answer",
            json={
                "query": query,
                "client_request_id": "one",
            },
        )
    assert (
        response.status_code == 200 and response.json()["response"]["answer"]["status"] == "skipped"
    )
    assert not parts[3].calls and not parts[4].calls and not parts[5].calls
