from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from zq_rag_app.api import chat as chat_api
from zq_rag_app.core.security import DEFAULT_ADMIN_USER
from zq_rag_app.models.chat import ChatMessage
from zq_rag_app.query_routing.models import ExecutionRecord, RoutingDecision
from zq_rag_app.services.rag_service import (
    RagAnswerResult,
    RagSource,
    RagTiming,
)


class _Session:
    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, item):
        self.added.append(item)

    def add_all(self, items):
        self.added.extend(items)
        for item in items:
            if isinstance(item, ChatMessage) and item.role == "ASSISTANT":
                item.id = 99

    async def commit(self):
        self.committed = True

    async def refresh(self, _item):
        return None

    async def rollback(self):
        return None


class _Service:
    closed = False
    kwargs = None

    async def answer(self, *_args, **kwargs):
        self.kwargs = kwargs
        return RagAnswerResult(
            query=kwargs["query"],
            answer="答案。[S1]",
            model="test-model",
            engine="hybrid",
            token_count=12,
            graph_degraded=False,
            reranked=True,
            sources=(
                RagSource(
                    citation_id="S1",
                    rank=1,
                    chunk_id=2,
                    doc_id=1,
                    document="来源.txt",
                    chunk_index=0,
                    section=None,
                    page=None,
                    excerpt="证据",
                    score=0.9,
                    dense_rank=1,
                    bm25_rank=1,
                    graph_rank=1,
                ),
            ),
            timing=RagTiming(retrieval_ms=20, generation_ms=30, total_ms=50),
            routing=RoutingDecision("hybrid", "hybrid", "retrieve", "hybrid_selected", "ok"),
            execution=ExecutionRecord("hybrid", kwargs["query"], (4,), (1,), 3, 1, weights=(0.5, 0.5, 0.0)),
        )

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_chat_answer_persists_session_messages_and_sources(monkeypatch):
    service = _Service()
    monkeypatch.setattr(chat_api, "RagAnswerService", lambda: service)
    async def accessible_documents(*_args, **_kwargs):
        return [1]

    monkeypatch.setattr(
        chat_api, "list_accessible_document_ids", accessible_documents
    )
    session = _Session()

    response = await chat_api.answer_question(
        chat_api.ChatAnswerRequest(
            query="问题",
            kb_ids=[4],
            candidate_k=3,
            top_k=1,
        ),
        DEFAULT_ADMIN_USER,
        session,
    )

    assert response.message_id == 99
    assert response.answer == "答案。[S1]"
    assert response.sources[0].citation_id == "S1"
    assert response.timing.total_ms == 50
    assert response.routing.path == "hybrid"
    assert response.model_dump(mode="json")["routing"]["reason"] == "hybrid_selected"
    assert response.model_dump(mode="json")["execution"]["weights"] == [0.5, 0.5, 0.0]
    assistant = next(item for item in session.added if isinstance(item, ChatMessage) and item.role == "ASSISTANT")
    assert assistant.retrieval_trace["routing"]["path"] == "hybrid"
    assert assistant.retrieval_trace["execution"]["doc_ids"] == [1]
    assert assistant.retrieval_trace["execution"]["weights"] == [0.5, 0.5, 0.0]
    assert service.kwargs["doc_ids"] == [1]
    assert session.committed is True
    assert [item.role for item in session.added if isinstance(item, ChatMessage)] == [
        "USER",
        "ASSISTANT",
    ]
    assert service.closed is True


@pytest.mark.asyncio
async def test_chat_loads_user_history_only_after_session_ownership_and_scope_checks(monkeypatch):
    service = _Service()
    session_id = "00000000-0000-0000-0000-000000000001"
    events = []

    async def accessible(*args, **kwargs):
        events.append("permissions")
        return [1]

    class Session(_Session):
        async def scalar(self, statement):
            events.append("session_owner")
            sql = str(statement)
            assert "kb_chat_session.user_id" in sql and "kb_chat_session.is_deleted" in sql
            return SimpleNamespace(id=session_id, kb_ids="[4]", message_count=4)

        async def execute(self, statement):
            events.append("history")
            sql = str(statement)
            assert "kb_chat_message.session_id" in sql and "kb_chat_message.role" in sql
            assert "DESC" in sql and "LIMIT" in sql
            rows = [SimpleNamespace(content="第二个问题"), SimpleNamespace(content="第一个问题")]
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

    monkeypatch.setattr(chat_api, "list_accessible_document_ids", accessible)
    monkeypatch.setattr(chat_api, "RagAnswerService", lambda: service)
    response = await chat_api.answer_question(chat_api.ChatAnswerRequest(
        query="新的问题", kb_ids=[4], session_id=session_id,
    ), DEFAULT_ADMIN_USER, Session())
    assert events == ["permissions", "session_owner", "history"]
    assert [message.content for message in service.kwargs["history"]] == ["第一个问题", "第二个问题"]
    assert all(message.role == "user" for message in service.kwargs["history"])
    assert response.session_id == session_id


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["documents", "session", "kb_scope"])
async def test_chat_rejects_invalid_scope_before_model_or_history_access(monkeypatch, failure):
    async def accessible(*args, **kwargs):
        return [] if failure == "documents" else [1]

    def forbidden_factory():
        raise AssertionError("权限或会话校验失败后不能调用模型")

    class Session(_Session):
        async def scalar(self, statement):
            return None if failure == "session" else SimpleNamespace(kb_ids="[5]")

        async def execute(self, statement):
            raise AssertionError("校验失败后不能读取历史")

    monkeypatch.setattr(chat_api, "list_accessible_document_ids", accessible)
    monkeypatch.setattr(chat_api, "RagAnswerService", forbidden_factory)
    with pytest.raises(HTTPException) as caught:
        await chat_api.answer_question(chat_api.ChatAnswerRequest(
            query="问题", kb_ids=[4], session_id="00000000-0000-0000-0000-000000000001",
        ), DEFAULT_ADMIN_USER, Session())
    assert caught.value.status_code == {"documents": 403, "session": 404, "kb_scope": 409}[failure]
