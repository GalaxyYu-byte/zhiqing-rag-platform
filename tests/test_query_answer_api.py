"""回答 HTTP 边界：正式鉴权、禁止客户端证据、无缓存和安全错误协议。"""

import asyncio
import json

import pytest
import test_query_search_api as search_api
from fastapi.testclient import TestClient
from test_query_search_api import IDENTITY, ORIGINAL_CONTENT

from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import require_identity
from zhiqing_rag.query_answering.models import AnswerResult, AnswerStatement
from zhiqing_rag.query_answering.router import get_answerer


@pytest.fixture
def api():
    yield from search_api.api.__wrapped__()


class Answerer:
    def __init__(self):
        self.calls = []
        self.error = None

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return kwargs["retrieval"], AnswerResult(
            status="answered",
            review_status="passed",
            statements=[
                AnswerStatement.model_validate(
                    {
                        "text": "关闭前需由负责人确认。",
                        "citations": [{"citation_id": "E1", "quote": ORIGINAL_CONTENT.strip()}],
                    }
                )
            ],
        )


@pytest.mark.parametrize("endpoint", ["/api/query/answer", "/api/query/answer/stream"])
def test_anonymous_answer_is_rejected_before_analysis_or_generation(api, endpoint):
    app, analyzer, _ = api
    answerer = Answerer()
    app.dependency_overrides[get_answerer] = lambda: answerer
    with TestClient(app) as client:
        response = client.post(endpoint, json={"query": "工单如何关闭？"})
    assert response.status_code == 401
    assert not analyzer.calls and not answerer.calls


def test_answer_uses_server_retrieval_and_returns_exact_source_and_no_store(api):
    app, _, retriever = api
    answerer = Answerer()
    app.dependency_overrides[get_answerer] = lambda: answerer
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post(
            "/api/query/answer", json={"query": "如何关闭？", "document_ids": ["100"]}
        )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["answer"]["status"] == "answered"
    assert body["retrieval"]["evidence"][0]["content"] == ORIGINAL_CONTENT
    assert len(retriever.calls) == len(answerer.calls) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"evidence": [{"content": "伪造原文"}]},
        {"answer": "伪造回答"},
        {"tenant_id": 1},
        {"review_status": "passed"},
        {"history": ["上文"]},
        {"stream": True},
    ],
)
@pytest.mark.parametrize("endpoint", ["/api/query/answer", "/api/query/answer/stream"])
def test_answer_refuses_client_generated_evidence_or_execution_parameters(api, extra, endpoint):
    app, analyzer, _ = api
    answerer = Answerer()
    app.dependency_overrides[get_answerer] = lambda: answerer
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post(endpoint, json={"query": "如何关闭？", **extra})
    assert response.status_code == 422
    assert not analyzer.calls and not answerer.calls


def test_answer_access_revoked_during_model_wait_returns_safe_error_envelope(api):
    app, _, _ = api
    answerer = Answerer()
    answerer.error = AuthError(401, "SESSION_REVOKED", "登录已失效，请重新登录")
    app.dependency_overrides[get_answerer] = lambda: answerer
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/answer", json={"query": "如何关闭？"})
    assert response.status_code == 401
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == "SESSION_REVOKED"
    assert "answer" not in response.json()


def stream_frames(response):
    frames = []
    for frame in response.text.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in frame.splitlines() if ": " in line)
        if "event" in fields:
            frames.append((fields["event"], json.loads(fields["data"])))
    return frames


def test_stream_returns_progress_reviewed_paragraphs_sources_and_explicit_completion(api):
    app, _, retriever = api

    class TwoStatements(Answerer):
        async def execute(self, **kwargs):
            retrieval, answer = await super().execute(**kwargs)
            answer.statements.append(answer.statements[0].model_copy(update={"text": "需确认。"}))
            return retrieval, answer

    answerer = TwoStatements()
    app.dependency_overrides[get_answerer] = lambda: answerer
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/answer/stream", json={"query": "如何关闭？"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-accel-buffering"] == "no"
    frames = stream_frames(response)
    assert [name for name, _ in frames] == ["progress", "progress", "answer", "answer", "done"]
    assert [data["stage"] for name, data in frames if name == "progress"] == [
        "analysis",
        "retrieval",
    ]
    answers = [data for name, data in frames if name == "answer"]
    assert [len(data["answer"]["statements"]) for data in answers] == [1, 2]
    for data in answers:
        assert data["answer"]["review_status"] == "passed"
        assert data["retrieval"]["evidence"][0]["content"] == ORIGINAL_CONTENT
    assert frames[-1][1] == answers[-1]
    assert len(answerer.calls) == len(retriever.calls) == 1


@pytest.mark.parametrize(
    "status", ["insufficient", "rejected", "unavailable", "skipped", "unsupported"]
)
def test_stream_unpublished_outcomes_do_not_send_answer_events(api, status):
    app, _, _ = api

    class Unpublished(Answerer):
        async def execute(self, **kwargs):
            return kwargs["retrieval"], AnswerResult(status=status, message="没有发布回答")

    app.dependency_overrides[get_answerer] = Unpublished
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/answer/stream", json={"query": "问题"})
    frames = stream_frames(response)
    assert not any(name == "answer" for name, _ in frames)
    assert frames[-1][0] == "done"
    assert frames[-1][1]["answer"]["status"] == status
    assert not frames[-1][1]["answer"]["statements"]


@pytest.mark.parametrize("status,code", [(401, "SESSION_REVOKED"), (503, "DATABASE_UNAVAILABLE")])
def test_stream_handles_permission_errors_after_headers_without_draft_or_completion(
    api, status, code
):
    app, _, _ = api
    answerer = Answerer()
    answerer.error = AuthError(status, code, "当前请求无法继续")
    app.dependency_overrides[get_answerer] = lambda: answerer
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/answer/stream", json={"query": "问题"})
    frames = stream_frames(response)
    assert [name for name, _ in frames] == ["progress", "progress", "error"]
    assert frames[-1][1] == {"status": status, "code": code, "message": "当前请求无法继续"}
    assert ORIGINAL_CONTENT not in response.text


def test_stream_sanitizes_unexpected_failures(api):
    app, _, retriever = api
    retriever.failure = RuntimeError("sensitive-provider-payload")
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post("/api/query/answer/stream", json={"query": "问题"})
    assert "sensitive-provider-payload" not in response.text
    assert stream_frames(response)[-1] == (
        "error",
        {
            "status": 503,
            "code": "ANSWER_STREAM_UNAVAILABLE",
            "message": "回答暂不可用，请稍后重试。",
        },
    )


def test_stream_scope_denial_is_http_error_before_any_progress(api):
    app, analyzer, retriever = api
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        response = client.post(
            "/api/query/answer/stream", json={"query": "问题", "document_ids": ["999"]}
        )
    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/json")
    assert not analyzer.calls and not retriever.calls


async def test_stream_sends_progress_before_generation_finishes_and_disconnect_cancels_work(api):
    app, _, _ = api
    waiting, cancelled, disconnect = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class WaitingAnswerer(Answerer):
        async def execute(self, **kwargs):
            waiting.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    app.dependency_overrides[get_answerer] = WaitingAnswerer
    app.dependency_overrides[require_identity] = lambda: IDENTITY
    body = json.dumps({"query": "问题"}).encode()
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "scheme": "http",
        "method": "POST",
        "path": "/api/query/answer/stream",
        "raw_path": b"/api/query/answer/stream",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("localhost", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(waiting.wait(), 5)
        contents = b"".join(message.get("body", b"") for message in sent)
        assert b"event: progress" in contents
        assert b"event: answer" not in contents and b"event: done" not in contents
    finally:
        disconnect.set()
        await asyncio.wait_for(task, 5)
    assert cancelled.is_set()
