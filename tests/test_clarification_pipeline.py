from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json

import pytest

from zq_rag_app.query_analysis.clarification import ClarificationCompletion, PendingClarification
from zq_rag_app.query_analysis.models import HistoryMessage, QueryAnalysisRequest
from zq_rag_app.services.query_rewrite_service import QueryRewriteService
from zq_rag_app.services.query_router_service import QueryRouterService
from zq_rag_app.services.rag_service import RagAnswerService
from zq_rag_app.services.retrieval_service import CosineRetrievalResult, RetrievedChunk
from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult


def pending():
    return PendingClarification.create(
        query="它由谁负责？", question="请明确项目。", reason="context_rewrite_required",
        kb_ids=[4], doc_ids=[7, 8],
        history=[HistoryMessage(role="user", content="星河和月海项目"),
                 HistoryMessage(role="assistant", content="无依据的事实")],
    )


def test_state_roundtrip_expiry_and_current_permission_intersection():
    state = PendingClarification.model_validate_json(pending().model_dump_json())
    assert [m.content for m in state.user_context] == ["星河和月海项目"]
    scoped = state.revalidate_scope([4], [8, 9])
    assert scoped.doc_ids == [8]
    assert [candidate.doc_id for candidate in scoped.candidates] == [8]
    assert state.doc_ids == [7, 8]
    assert state.revalidate_scope([5], [7]) is None
    assert state.revalidate_scope([4], [9]) is None
    state.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert state.revalidate_scope([4], [7]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path,intent,kind", [
    ("structured", "statistics", "aggregate"), ("none", "chitchat", "conversational"),
])
async def test_context_guard_precedes_all_terminal_routes(analysis_factory, path, intent, kind):
    analysis = analysis_factory(query="它呢？", path=path, intent=intent, query_type=kind, reference=True)
    result = await QueryRouterService().route(
        object(), analysis=analysis,
        request=QueryAnalysisRequest(query=analysis.original_query, history=[{"role": "user", "content": "项目"}]),
        kb_ids=[4], doc_ids=[7], preprocessing_complete=True,
    )
    assert result.decision.action == "clarify"
    assert result.decision.reason == "context_rewrite_required"


class Client:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False)),
        )])


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_semantic_completion_is_independently_audited(accepted):
    candidate = {"status": "completed", "query": "星河项目由谁负责？", "clarification_question": None}
    audit = dict(equivalent=True, intent_preserved=True, constraints_preserved=True,
                 no_unsupported_additions=True, context_resolved=accepted, reason="语义检查")
    client = Client([candidate, audit])
    service = QueryRewriteService(client=client, retry_attempts=1)
    result = await service.complete_clarification("星河项目", pending())
    assert result.status == ("completed" if accepted else "clarify")
    assert len(client.calls) == 2
    assert "无依据的事实" not in client.calls[0]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_missing_completion_provider_keeps_clarification():
    result = await QueryRewriteService(api_key="").complete_clarification("星河项目", pending())
    assert result.status == "clarify" and result.query is None


@pytest.mark.asyncio
async def test_new_task_cannot_change_user_input():
    client = Client([{"status": "new_task", "query": "模型虚构的新任务", "clarification_question": None}])
    result = await QueryRewriteService(client=client, retry_attempts=1).complete_clarification("星河项目", pending())
    assert result.status == "clarify" and len(client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "clarify", "new_task"])
async def test_supplement_completed_before_analysis_and_routing(analysis_factory, status):
    events = []
    semantic = "差旅报销流程是什么？" if status == "new_task" else "星河项目由谁负责？"
    class Rewriter:
        async def complete_clarification(self, query, state):
            events.append("complete")
            assert state.doc_ids == [8] and len(state.candidates) == 1
            return ClarificationCompletion(status=status, query=None if status == "clarify" else semantic,
                                           clarification_question="哪个部门？" if status == "clarify" else None)

        async def aclose(self):
            events.append("close_rewriter")

    class Analyzer:
        async def analyze(self, request):
            events.append("analyze")
            assert request.query == semantic and not request.history
            return analysis_factory(query=semantic)

        async def aclose(self):
            pass

    async def dense(session, **kwargs):
        events.append("retrieve")
        assert kwargs["query"] == semantic
        assert kwargs["doc_ids"] == ([8, 9] if status == "new_task" else [8])
        return CosineRetrievalResult(semantic, "test", 1, 0, [])

    async def bm25(session, **kwargs):
        return BM25RetrievalResult(semantic, "test", 0, [])

    service = RagAnswerService(client=object(), analyzer_factory=Analyzer, rewrite_factory=Rewriter,
                               dense_search=dense, bm25_search=bm25, multi_query_enabled=False)
    result = await service.answer(object(), query=semantic if status == "new_task" else "星河项目",
                                  kb_ids=[4], doc_ids=[8, 9], pending_clarification=pending(), rerank=False)
    if status == "clarify":
        assert events == ["complete", "close_rewriter"]
        assert result.pending_clarification.original_query == "它由谁负责？"
        assert result.pending_clarification.supplements == ["星河项目"]
        assert result.pending_clarification.doc_ids == [8]
    else:
        assert events == ["complete", "close_rewriter", "analyze", "retrieve"]
        assert result.pending_clarification is None


@pytest.mark.asyncio
async def test_invalid_scope_is_rejected_before_completion():
    service = RagAnswerService(client=object(), rewrite_factory=lambda: pytest.fail("不应调用补全"))
    with pytest.raises(ValueError, match="授权范围失效"):
        await service.answer(object(), query="星河项目", kb_ids=[4], doc_ids=[9], pending_clarification=pending())


@pytest.mark.asyncio
async def test_resumed_generation_receives_original_task_and_completed_question(analysis_factory):
    semantic = "星河项目由谁负责？"
    state, prompts = pending(), []
    class Rewriter:
        async def complete_clarification(self, query, current):
            return ClarificationCompletion(status="completed", query=semantic, clarification_question=None)
        async def aclose(self):
            pass
    class Analyzer:
        async def analyze(self, request):
            return analysis_factory(query=semantic)
        async def aclose(self):
            pass
    async def create(**kwargs):
        prompts.append(kwargs["messages"][1]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="负责人是张三。[S1]"))])
    chunk = RetrievedChunk(1, 1, 7, "项目.txt", 0, None, None, "星河项目由张三负责。", 10, 0.9)
    async def dense(session, **kwargs):
        assert kwargs["query"] == semantic
        return CosineRetrievalResult(semantic, "test", 1, 0, [chunk])
    async def bm25(session, **kwargs):
        return BM25RetrievalResult(semantic, "test", 0, [])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    service = RagAnswerService(client=client, analyzer_factory=Analyzer, rewrite_factory=Rewriter,
                               dense_search=dense, bm25_search=bm25, multi_query_enabled=False)
    result = await service.answer(object(), query="星河项目", kb_ids=[4], doc_ids=[7, 8],
                                  pending_clarification=state, rerank=False)
    assert result.query == "星河项目" and result.pending_clarification is None
    assert f"原问题：{state.original_query}" in prompts[0]
    assert f"经过校验的独立问题：{semantic}" in prompts[0]


@pytest.mark.asyncio
async def test_chat_persists_then_consumes_state_with_fresh_permissions(monkeypatch):
    from zq_rag_app.api import chat as api
    from zq_rag_app.core.security import DEFAULT_ADMIN_USER
    from zq_rag_app.models.chat import ChatMessage, ChatSession
    from zq_rag_app.query_routing.models import RoutingDecision
    from zq_rag_app.services.rag_service import RagAnswerResult, RagTiming

    task = pending()
    calls = []
    class Session:
        stored = None
        messages = []
        async def scalar(self, statement):
            assert "FOR UPDATE" in str(statement)
            return self.stored

        async def execute(self, statement):
            # 不加载助手回答，补全恢复使用专门的待澄清状态。
            assert statement.compile().params["role_1"] == "USER"
            rows = [row for row in reversed(self.messages) if row.role == "USER"]
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))

        def add(self, value):
            if isinstance(value, ChatSession):
                self.stored = value

        def add_all(self, values):
            for value in values:
                value.id = len(self.messages) + 1
                self.messages.append(value)

        async def commit(self):
            pass

        async def refresh(self, value):
            pass

    class Service:
        async def answer(self, session, **kwargs):
            calls.append(kwargs)
            first = len(calls) == 1
            if not first:
                restored = kwargs["pending_clarification"]
                assert restored.original_query == task.original_query
                assert restored.doc_ids == [8]
                assert [candidate.doc_id for candidate in restored.candidates] == [8]
            return RagAnswerResult(
                query=kwargs["query"], answer="请明确项目" if first else "完成",
                model="test", engine="test", token_count=0, graph_degraded=False, reranked=False,
                sources=(), timing=RagTiming(0, 0, 1), pending_clarification=task if first else None,
                routing=RoutingDecision("clarify", "clarify", "clarify", "clarification_required", "ok") if first
                        else RoutingDecision("hybrid", "hybrid", "retrieve", "hybrid_selected", "ok"),
            )

        async def aclose(self):
            pass

    async def accessible(*args, **kwargs):
        return [7, 8] if not calls else [8, 9]

    monkeypatch.setattr(api, "RagAnswerService", Service)
    monkeypatch.setattr(api, "list_accessible_document_ids", accessible)
    session = Session()
    first = await api.answer_question(api.ChatAnswerRequest(query="它由谁负责？", kb_ids=[4]),
                                      DEFAULT_ADMIN_USER, session)
    assert session.stored.pending_clarification["original_query"] == task.original_query
    second = await api.answer_question(api.ChatAnswerRequest(query="星河项目", kb_ids=[4], session_id=first.session_id),
                                       DEFAULT_ADMIN_USER, session)
    assert second.answer == "完成" and session.stored.pending_clarification is None
    assert len(session.messages) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expired", "revoked", "invalid"])
async def test_chat_does_not_resurrect_invalid_pending_from_history(monkeypatch, failure):
    from zq_rag_app.api import chat as api
    from zq_rag_app.core.security import DEFAULT_ADMIN_USER
    from zq_rag_app.services.rag_service import RagAnswerResult, RagTiming

    state = pending()
    if failure == "expired":
        state.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    stored = {"invalid": True} if failure == "invalid" else state.model_dump(mode="json")
    chat_session = SimpleNamespace(id="00000000-0000-0000-0000-000000000001", kb_ids="[4]",
                                   message_count=2, pending_clarification=stored)
    class Session:
        async def scalar(self, statement):
            return chat_session

        async def execute(self, statement):
            pytest.fail("失效澄清状态不能通过普通历史再次恢复")

        def add_all(self, messages):
            messages[-1].id = 1

        async def commit(self):
            pass

        async def refresh(self, value):
            pass

    class Service:
        async def answer(self, session, **kwargs):
            assert kwargs["pending_clarification"] is None and not kwargs["history"]
            return RagAnswerResult(query=kwargs["query"], answer="新问题", model="test", engine="test",
                                   token_count=0, graph_degraded=False, reranked=False, sources=(),
                                   timing=RagTiming(0, 0, 1))

        async def aclose(self):
            pass

    async def accessible(*args, **kwargs):
        return [9] if failure == "revoked" else [7, 8]

    monkeypatch.setattr(api, "RagAnswerService", Service)
    monkeypatch.setattr(api, "list_accessible_document_ids", accessible)
    await api.answer_question(api.ChatAnswerRequest(query="新问题", kb_ids=[4], session_id=chat_session.id),
                              DEFAULT_ADMIN_USER, Session())
    assert chat_session.pending_clarification is None


@pytest.mark.asyncio
async def test_second_partial_supplement_retains_original_task_and_expiry():
    state = pending()
    state.supplements = ["星河项目"]
    class Rewriter:
        async def complete_clarification(self, query, current):
            assert current.supplements == ["星河项目"]
            return ClarificationCompletion(status="clarify", query=None, clarification_question="请明确年份。")

        async def aclose(self):
            pass

    service = RagAnswerService(client=object(), rewrite_factory=Rewriter,
                               analyzer_factory=lambda: pytest.fail("补全失败不能进行分析和路由"))
    result = await service.answer(object(), query="研发部", kb_ids=[4], doc_ids=[7, 8], pending_clarification=state)
    assert result.pending_clarification.supplements == ["星河项目", "研发部"]
    assert result.pending_clarification.original_query == state.original_query
    assert result.pending_clarification.expires_at == state.expires_at
