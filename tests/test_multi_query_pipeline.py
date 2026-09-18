import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from zq_rag_app.query_analysis.multi_query import MultiQuerySummary
from zq_rag_app.query_routing.models import GraphCoverage
from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult
from zq_rag_app.services.graph_retrieval_service import GraphRetrievalResult
from zq_rag_app.services.multi_query_retrieval_service import fuse_query_rankings, retrieve_expansions
from zq_rag_app.services.query_router_service import QueryRouterService
from zq_rag_app.services.rag_service import RagAnswerService
from zq_rag_app.services.retrieval_service import CosineRetrievalResult, RetrievedChunk


QUERY = "Aurora-KB 的使用流程是什么？"
VARIANTS = ("Aurora-KB 的操作流程是什么？", "如何使用 Aurora-KB？")


def chunk(chunk_id, rank=1, doc_id=7):
    return RetrievedChunk(rank, chunk_id, doc_id, "操作说明.txt", chunk_id, None, None,
                          f"操作说明片段 {chunk_id}。", 5, 0.9)


class Client:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="按说明操作。[S1]"))],
                               usage=SimpleNamespace(total_tokens=10))


def expander(calls, *, status="expanded"):
    class Service:
        async def expand(self, request, analysis, **kwargs):
            calls.append(("expand", {**kwargs, "query": request.query}))
            return MultiQuerySummary(status, (request.query, *VARIANTS) if status == "expanded" else (request.query,),
                                     trigger=kwargs["trigger"], attempts=1,
                                     degradation_reason="invalid_output" if status == "degraded" else None)
        async def aclose(self):
            calls.append(("close_expander", {}))
    return Service


def searches(calls, *, original_empty=False, fail_variant=None, graph_enabled=False, score=0.9):
    def chunks(query):
        if query == QUERY:
            return [] if original_empty else [replace(chunk(1), score=score)]
        return [chunk(2 + VARIANTS.index(query)), chunk(1, rank=2)]
    async def dense(session, **kwargs):
        calls.append(("dense", kwargs))
        if kwargs["query"] == fail_variant:
            raise ConnectionError("do-not-log-provider-content")
        return CosineRetrievalResult(kwargs["query"], "test", 3, 0, chunks(kwargs["query"]))
    async def bm25(session, **kwargs):
        calls.append(("bm25", kwargs))
        return BM25RetrievalResult(kwargs["query"], "test", 0, chunks(kwargs["query"]))
    async def graph(session, **kwargs):
        calls.append(("graph", kwargs))
        return GraphRetrievalResult(kwargs["query"], "test", 0, [chunk(4)] if graph_enabled else [], ())
    return dense, bm25, graph


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["analyzer", "empty_retrieval"])
async def test_both_triggers_fixed_scope_and_one_rerank(analysis_factory, analyzer_stub_factory, trigger):
    calls, ranked = [], []
    dense, bm25, graph = searches(calls, original_empty=trigger == "empty_retrieval")
    class Reranker:
        async def rerank(self, result, *, top_n):
            ranked.append((result.query, len(result.results)))
            return result
        async def aclose(self):
            pass
    client = Client()
    analysis = analysis_factory(query=QUERY, intent="procedure", names=["Aurora-KB"],
                                multi_query=trigger == "analyzer")
    service = RagAnswerService(client=client, dense_search=dense, bm25_search=bm25, graph_search=graph,
                               analyzer_factory=analyzer_stub_factory(analysis), multi_query_factory=expander(calls),
                               multi_query_enabled=True, reranker_factory=Reranker)
    result = await service.answer(object(), query=QUERY, kb_ids=[4], doc_ids=[7], candidate_k=3, top_k=2)
    assert len([c for c, _ in calls if c == "expand"]) == 1
    assert next(k for c, k in calls if c == "expand")["trigger"] == trigger
    if trigger == "analyzer":
        assert calls[0][0] == "expand"
    else:
        assert [c for c, _ in calls[:3]] == ["dense", "bm25", "expand"]
    for name in ("dense", "bm25"):
        assert [k["query"] for c, k in calls if c == name] == [QUERY, *VARIANTS]
    assert all(k["kb_ids"] == [4] and k["doc_ids"] == [7] for c, k in calls if c in {"dense", "bm25"})
    assert ranked == [(QUERY, 3)]  # 候选预算保持固定，仅重排一次。
    assert result.multi_query.applied_queries == (QUERY, *VARIANTS)
    assert result.multi_query.trigger == trigger and result.multi_query.status == "expanded"
    assert len({s.chunk_id for s in result.sources}) == len(result.sources)
    assert "问题：" + QUERY in client.calls[0]["messages"][1]["content"]
    assert result.query == QUERY and result.routing.deferred_features == ()


@pytest.mark.asyncio
async def test_graph_coverage_and_graph_retrieval_run_once(analysis_factory, analyzer_stub_factory):
    calls = []
    dense, bm25, graph = searches(calls, graph_enabled=True)
    async def coverage(*args, **kwargs):
        calls.append(("coverage", kwargs))
        return GraphCoverage("covered", ("confirmed-uid",))
    class Router(QueryRouterService):
        async def route(self, *args, **kwargs):
            calls.append(("route", {}))
            return await super().route(*args, **kwargs)
    analysis = analysis_factory(query=QUERY, names=["Aurora-KB"], path="hybrid_graph", multi_query=True)
    service = RagAnswerService(client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
                               analyzer_factory=analyzer_stub_factory(analysis), multi_query_factory=expander(calls),
                               multi_query_enabled=True, query_router=Router(graph_coverage=coverage))
    result = await service.answer(object(), query=QUERY, kb_ids=[4], doc_ids=[7], rerank=False)
    assert [c for c, _ in calls].count("route") == 1
    assert [c for c, _ in calls].count("coverage") == 1
    assert [c for c, _ in calls].count("graph") == 1
    assert next(k for c, k in calls if c == "graph")["seed_entity_uids"] == ["confirmed-uid"]
    assert result.routing.path == "hybrid_graph"
    assert "multi_query_rrf" in result.engine


@pytest.mark.asyncio
async def test_nonempty_low_scores_do_not_trigger_expansion(analysis_factory, analyzer_stub_factory):
    calls = []
    dense, bm25, graph = searches(calls, score=0.01)
    service = RagAnswerService(client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
                               analyzer_factory=analyzer_stub_factory(analysis_factory(query=QUERY)),
                               multi_query_factory=expander(calls), multi_query_enabled=True)
    result = await service.answer(object(), query=QUERY, kb_ids=[4], doc_ids=[7], rerank=False)
    assert not any(c == "expand" for c, _ in calls)
    assert result.multi_query.status == "skipped" and result.multi_query.reason == "no_trigger"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_empty_result_rescue_switches(analysis_factory, analyzer_stub_factory, enabled):
    calls = []
    dense, bm25, graph = searches(calls, original_empty=True)
    service = RagAnswerService(client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
                               analyzer_factory=analyzer_stub_factory(analysis_factory(query=QUERY)),
                               multi_query_factory=expander(calls), multi_query_enabled=enabled,
                               empty_retrieval_expansion=False)
    result = await service.answer(object(), query=QUERY, kb_ids=[4], doc_ids=[7], rerank=False)
    assert not any(c == "expand" for c, _ in calls)
    assert not result.sources and result.timing.expansion_ms == 0
    assert result.multi_query.status == ("skipped" if enabled else "disabled")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["degraded", "unchanged"])
async def test_analyzer_attempt_is_not_retried_by_empty_result_trigger(analysis_factory, analyzer_stub_factory, status):
    calls = []
    dense, bm25, graph = searches(calls, original_empty=True)
    service = RagAnswerService(client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
                               analyzer_factory=analyzer_stub_factory(analysis_factory(query=QUERY, multi_query=True)),
                               multi_query_factory=expander(calls, status=status), multi_query_enabled=True)
    result = await service.answer(object(), query=QUERY, kb_ids=[4], doc_ids=[7], rerank=False)
    assert len([c for c, _ in calls if c == "expand"]) == 1
    assert result.multi_query.status == status and not result.sources


@pytest.mark.asyncio
async def test_partial_failure_preserves_original_and_successful_expansion(analysis_factory, analyzer_stub_factory, caplog):
    calls = []
    dense, bm25, graph = searches(calls, fail_variant=VARIANTS[0])
    service = RagAnswerService(client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
                               analyzer_factory=analyzer_stub_factory(analysis_factory(query=QUERY, multi_query=True)),
                               multi_query_factory=expander(calls), multi_query_enabled=True)
    result = await service.answer(object(), query=QUERY, kb_ids=[4], doc_ids=[7], rerank=False)
    assert result.multi_query.status == "degraded"
    assert result.multi_query.applied_queries == (QUERY, VARIANTS[1])
    assert result.multi_query.failed_queries == (VARIANTS[0],)
    assert result.execution.applied_queries == (QUERY, VARIANTS[1])
    assert result.execution.degradation_reasons == ("retrieval_error",)
    assert {1, 3}.issubset({s.chunk_id for s in result.sources})
    assert "do-not-log-provider-content" not in caplog.text


def test_rrf_original_prior_duplicates_and_top_k():
    original = [chunk(1), chunk(2, 2)]
    extra = [chunk(3), chunk(3, 2)]
    fused = fuse_query_rankings([original, extra], top_k=3)
    assert [c.chunk_id for c in fused] == [1, 2, 3]
    assert [c.rank for c in fused] == [1, 2, 3]
    assert fused[-1].score == pytest.approx(0.4)  # 重复 ID 未重复投票。
    assert len(fuse_query_rankings([original, extra], top_k=2)) == 2


@pytest.mark.asyncio
async def test_timeout_keeps_completed_queries_and_rolls_back_savepoint():
    scopes = []
    class Session:
        @asynccontextmanager
        async def begin_nested(self):
            try:
                yield
                scopes.append("commit")
            except BaseException:
                scopes.append("rollback")
                raise
    async def dense(session, **kwargs):
        if kwargs["query"] == VARIANTS[1]:
            await asyncio.Event().wait()
        return CosineRetrievalResult(kwargs["query"], "test", 3, 0, [chunk(2)])
    async def bm25(session, **kwargs):
        return BM25RetrievalResult(kwargs["query"], "test", 0, [chunk(2)])
    original_dense = CosineRetrievalResult(QUERY, "test", 3, 0, [chunk(1)])
    original_bm25 = BM25RetrievalResult(QUERY, "test", 0, [chunk(1)])
    result_dense, _, summary = await retrieve_expansions(
        Session(), dense=original_dense, bm25=original_bm25,
        expansion=MultiQuerySummary("expanded", (QUERY, *VARIANTS), trigger="analyzer"),
        dense_search=dense, bm25_search=bm25, kb_ids=[4], doc_ids=[7], candidate_k=3, timeout_seconds=0.2,
    )
    assert summary.degradation_reason == "retrieval_timeout" and summary.applied_queries == (QUERY, VARIANTS[0])
    assert summary.failed_queries == (VARIANTS[1],) and scopes == ["commit", "rollback"]
    assert {c.chunk_id for c in result_dense.results} == {1, 2}


@pytest.mark.asyncio
async def test_unauthorized_variant_results_are_discarded():
    async def dense(session, **kwargs):
        return CosineRetrievalResult(kwargs["query"], "test", 3, 0, [chunk(999, doc_id=999)])
    async def bm25(session, **kwargs):
        return BM25RetrievalResult(kwargs["query"], "test", 0, [])
    original_dense = CosineRetrievalResult(QUERY, "test", 3, 0, [chunk(1)])
    result_dense, _, summary = await retrieve_expansions(
        object(), dense=original_dense, bm25=BM25RetrievalResult(QUERY, "test", 0, [chunk(1)]),
        expansion=MultiQuerySummary("expanded", (QUERY, VARIANTS[0]), trigger="analyzer"),
        dense_search=dense, bm25_search=bm25, kb_ids=[4], doc_ids=[7], candidate_k=3, timeout_seconds=1,
    )
    assert result_dense is original_dense and summary.applied_queries == (QUERY,)
    assert summary.degradation_reason == "retrieval_error"


@pytest.mark.asyncio
async def test_retrieval_cancellation_propagates_and_exits_savepoint():
    scopes = []
    class Session:
        @asynccontextmanager
        async def begin_nested(self):
            try:
                yield
            finally:
                scopes.append("exited")
    started = asyncio.Event()
    async def blocked(session, **kwargs):
        started.set()
        await asyncio.Event().wait()
    task = asyncio.create_task(retrieve_expansions(
        Session(), dense=CosineRetrievalResult(QUERY, "test", 3, 0, []),
        bm25=BM25RetrievalResult(QUERY, "test", 0, []),
        expansion=MultiQuerySummary("expanded", (QUERY, VARIANTS[0]), trigger="analyzer"),
        dense_search=blocked, bm25_search=blocked, kb_ids=[4], doc_ids=[7], candidate_k=3, timeout_seconds=1,
    ))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scopes == ["exited"]
