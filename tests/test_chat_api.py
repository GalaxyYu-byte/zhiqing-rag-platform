from types import SimpleNamespace

import pytest

from zq_rag_app.api import chat as chat_api
from zq_rag_app.core.security import DEFAULT_ADMIN_USER
from zq_rag_app.models.chat import ChatMessage
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
    assert service.kwargs["doc_ids"] == [1]
    assert session.committed is True
    assert [item.role for item in session.added if isinstance(item, ChatMessage)] == [
        "USER",
        "ASSISTANT",
    ]
    assert service.closed is True
