import asyncio
from types import SimpleNamespace

import pytest

from zq_rag_app.query_routing.models import GraphCoverage
from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult
from zq_rag_app.services.graph_retrieval_service import GraphRetrievalResult
from zq_rag_app.services.query_router_service import QueryRouterService
from zq_rag_app.services.rag_service import RagAnswerService
from zq_rag_app.services.retrieval_service import CosineRetrievalResult, RetrievedChunk


class Client:
    def __init__(self, answer="已找到相关规定。[S1]"):
        self.answer = answer
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))],
            usage=SimpleNamespace(total_tokens=10),
        )


def searches(calls, *, graph_result="ok"):
    chunks = [RetrievedChunk(1, 1, 7, "制度.txt", 0, None, None, "相关规定。", 5, 0.9)]

    async def dense(session, **kwargs):
        calls.append(("dense", kwargs))
        return CosineRetrievalResult(kwargs["query"], "test", 3, 0, chunks)

    async def bm25(session, **kwargs):
        calls.append(("bm25", kwargs))
        return BM25RetrievalResult(kwargs["query"], "test", 0, chunks)

    async def graph(session, **kwargs):
        calls.append(("graph", kwargs))
        if graph_result == "timeout":
            await asyncio.sleep(1)
        if graph_result == "unavailable":
            raise ConnectionError("unavailable")
        return GraphRetrievalResult(kwargs["query"], "test", 0, [] if graph_result == "empty" else chunks, ())

    return dense, bm25, graph


@pytest.mark.asyncio
@pytest.mark.parametrize("path,intent,expected_action", [
    ("clarify", "fact", "clarify"),
    ("structured", "statistics", "unsupported"),
    ("none", "chitchat", "chat"),
])
async def test_non_retrieval_routes_never_search(analysis_factory, analyzer_stub_factory, path, intent, expected_action):
    query = "你好"
    calls, requests, closed = [], [], []
    dense, bm25, graph = searches(calls)
    client = Client(answer="你好！有什么可以帮你？")
    service = RagAnswerService(
        client=client, dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis_factory(query=query, path=path, intent=intent),
                                              requests=requests, closed=closed),
    )
    result = await service.answer(object(), query=query, kb_ids=[4], doc_ids=[7], rerank=False)
    assert not calls and not result.sources and not result.reranked
    assert result.routing.action == expected_action
    assert result.engine == f"query_router_{expected_action}"
    assert result.answer and result.timing.retrieval_ms == 0
    assert len(client.calls) == (1 if path == "none" else 0)
    assert len(requests) == 1 and closed == [True]


@pytest.mark.asyncio
async def test_hybrid_route_uses_scoped_two_way_retrieval(analysis_factory, analyzer_stub_factory):
    calls = []
    dense, bm25, graph = searches(calls)
    service = RagAnswerService(
        client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis_factory()),
    )
    result = await service.answer(object(), query="Aurora-KB 谁负责？", kb_ids=[4], doc_ids=[7],
                                  rerank=False, top_k=1)
    assert [name for name, _ in calls] == ["dense", "bm25"]
    assert all(kwargs["kb_ids"] == [4] and kwargs["doc_ids"] == [7] for _, kwargs in calls)
    assert result.routing.path == "hybrid" and not result.graph_degraded
    assert result.sources[0].graph_rank is None
    assert "graph" not in result.engine


@pytest.mark.asyncio
@pytest.mark.parametrize("graph_result,reason", [
    ("ok", "graph_covered"),
    ("empty", "graph_execution_empty"),
    ("unavailable", "graph_execution_unavailable"),
    ("timeout", "graph_execution_timeout"),
])
async def test_graph_uses_resolved_uids_and_handles_late_failures(
    analysis_factory, analyzer_stub_factory, graph_result, reason,
):
    calls = []
    dense, bm25, graph = searches(calls, graph_result=graph_result)

    async def coverage(*args, **kwargs):
        return GraphCoverage("covered", ("resolved-uid",))

    service = RagAnswerService(
        client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis_factory(path="hybrid_graph", names=["Aurora-KB"])),
        query_router=QueryRouterService(graph_coverage=coverage, graph_timeout_seconds=0.02),
    )
    result = await service.answer(object(), query="Aurora-KB 谁负责？", kb_ids=[4], doc_ids=[7], rerank=False)
    assert [name for name, _ in calls] == ["dense", "bm25", "graph"]
    assert calls[2][1]["seed_entity_uids"] == ["resolved-uid"]
    assert calls[2][1]["max_hops"] == 1 and calls[2][1]["doc_ids"] == [7]
    assert result.routing.reason == reason
    assert result.graph_degraded == (graph_result != "ok")
    assert result.routing.path == ("hybrid_graph" if graph_result == "ok" else "hybrid")
    assert result.sources[0].graph_rank == (1 if graph_result == "ok" else None)


@pytest.mark.asyncio
async def test_preflight_failure_skips_actual_graph(analysis_factory, analyzer_stub_factory):
    calls = []
    dense, bm25, graph = searches(calls)
    async def coverage(*args, **kwargs):
        return GraphCoverage("unmatched")
    service = RagAnswerService(
        client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis_factory(path="hybrid_graph", names=["Aurora-KB"])),
        query_router=QueryRouterService(graph_coverage=coverage),
    )
    result = await service.answer(object(), query="Aurora-KB 谁负责？", kb_ids=[4], doc_ids=[7], rerank=False)
    assert [name for name, _ in calls] == ["dense", "bm25"]
    assert result.graph_degraded and result.routing.reason == "graph_not_covered"


@pytest.mark.asyncio
async def test_analyzer_degraded_without_history_still_retrieves(analysis_factory, analyzer_stub_factory):
    calls = []
    dense, bm25, graph = searches(calls)
    service = RagAnswerService(
        client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis_factory(status="degraded")),
    )
    result = await service.answer(object(), query="Aurora-KB 谁负责？", kb_ids=[4], doc_ids=[7], rerank=False)
    assert [name for name, _ in calls] == ["dense", "bm25"]
    assert result.routing.analyzer_status == "degraded" and result.routing.reason == "analysis_degraded"


@pytest.mark.asyncio
async def test_analyzer_closed_even_on_failure(analyzer_stub_factory):
    closed = []
    class BrokenAnalyzer:
        async def analyze(self, request):
            raise RuntimeError("bug")
        async def aclose(self):
            closed.append(True)
    service = RagAnswerService(client=Client(), analyzer_factory=BrokenAnalyzer)
    with pytest.raises(RuntimeError):
        await service.answer(object(), query="问题", kb_ids=[4], doc_ids=[7])
    assert closed == [True]
