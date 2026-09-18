import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from zq_rag_app.query_routing.models import GraphCoverage
from zq_rag_app.query_analysis.models import QueryKeyword
from zq_rag_app.query_analysis.rewrite import RewriteResult, RewriteSummary
from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult
from zq_rag_app.services.graph_retrieval_service import GraphRetrievalResult
from zq_rag_app.services.hybrid_retrieval_service import fuse_dense_bm25_rrf
from zq_rag_app.services.query_router_service import QueryRouterService
from zq_rag_app.services.query_analyzer_service import QueryAnalyzerService
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
@pytest.mark.parametrize("graph_result", ["empty", "unavailable", "timeout"])
@pytest.mark.parametrize("exact_mode,weights", [
    ("semantic", (0.5, 0.5)),
    ("exact_type", (0.3, 0.7)),
    ("exact_keyword", (0.3, 0.7)),
])
async def test_graph_runtime_failure_restores_hybrid_scores_and_order(
    analysis_factory, analyzer_stub_factory, graph_result, exact_mode, weights,
):
    query = "Aurora-KB 谁负责？"
    analysis = analysis_factory(
        path="hybrid_graph", names=["Aurora-KB"],
        query_type="exact" if exact_mode == "exact_type" else "semantic",
    )
    if exact_mode == "exact_keyword":
        analysis.keywords = [QueryKeyword(
            text="Aurora-KB", kind="exact", source="query", history_index=None,
            evidence_quote="Aurora-KB", confidence=0.9,
        )]
    original_analysis = analysis.model_dump()
    first = RetrievedChunk(1, 1, 7, "制度.txt", 0, None, None, "规定一。", 5, 0.9)
    second = replace(first, rank=2, chunk_id=2, chunk_index=1, content="规定二。", score=0.4)
    dense_result = CosineRetrievalResult(query, "test", 3, 0, [first, second])
    bm25_result = BM25RetrievalResult(query, "test", 0, [
        replace(second, rank=1, score=0.9), replace(first, rank=2, score=0.4),
    ])

    async def dense(*args, **kwargs):
        return dense_result

    async def bm25(*args, **kwargs):
        return bm25_result

    async def coverage(*args, **kwargs):
        return GraphCoverage("covered", ("resolved-uid",))

    _, _, graph = searches([], graph_result=graph_result)
    service = RagAnswerService(
        client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis), multi_query_enabled=False,
        query_router=QueryRouterService(graph_coverage=coverage, graph_timeout_seconds=0.02),
    )
    result = await service.answer(object(), query=query, kb_ids=[4], doc_ids=[7],
                                  candidate_k=2, top_k=2, rerank=False)
    expected = fuse_dense_bm25_rrf(
        dense_result, bm25_result, top_k=2, dense_weight=weights[0], bm25_weight=weights[1],
    )
    assert result.routing.path == "hybrid" and result.graph_degraded
    assert [source.chunk_id for source in result.sources] == [chunk.chunk_id for chunk in expected.results]
    assert [source.score for source in result.sources] == pytest.approx([chunk.score for chunk in expected.results])
    assert all(source.graph_rank is None for source in result.sources)
    assert result.engine == f"{expected.engine}+graph_fallback"
    assert analysis.model_dump() == original_analysis
    assert result.routing.model_suggestion["path"] == "hybrid_graph"
    assert result.routing.policy_path == "hybrid_graph"
    assert result.routing.rule_path == "hybrid_graph"
    assert result.routing.rule_reason == "graph_covered"
    assert result.routing.rule_weights == (0.4, 0.3, 0.3)
    assert result.execution.path == "hybrid"
    assert result.execution.weights == (*weights, 0.0)
    assert result.execution.graph_attempted
    assert result.execution.graph_seed_entity_uids == ("resolved-uid",)
    assert result.execution.graph_max_hops == 1
    assert result.execution.graph_timeout_seconds == 0.02
    assert result.execution.kb_ids == (4,) and result.execution.doc_ids == (7,)
    assert (result.execution.candidate_k, result.execution.top_k) == (2, 2)
    assert result.execution.applied_queries == (query,)
    assert result.execution.degradation_reasons == (f"graph_execution_{graph_result}",)


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
async def test_analyzer_outage_does_not_trap_existing_session_after_clarification():
    calls = []
    dense, bm25, graph = searches(calls)
    client = Client()
    def forbidden_preprocessing():
        raise AssertionError("Analyzer 故障时不应调用改写、HyDE 或扩展")
    service = RagAnswerService(
        client=client, dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=lambda: QueryAnalyzerService(api_key=""), rewrite_factory=forbidden_preprocessing,
        multi_query_factory=forbidden_preprocessing, multi_query_enabled=True,
    )
    history = [{"role": "user", "content": "介绍星河项目。"}]
    first = await service.answer(object(), query="它是谁负责？", kb_ids=[4], doc_ids=[7], history=history, rerank=False)
    assert first.routing.action == "clarify" and not calls and not client.calls
    history += [{"role": "user", "content": first.query}, {"role": "assistant", "content": first.answer}]
    second = await service.answer(object(), query="星河项目是谁负责？", kb_ids=[4], doc_ids=[7], history=history, rerank=False)
    assert second.routing.action == "retrieve" and second.routing.analyzer_status == "degraded"
    assert [c for c, _ in calls] == ["dense", "bm25"]
    assert all(k["query"] == second.query and k["kb_ids"] == [4] and k["doc_ids"] == [7] for _, k in calls)
    assert second.sources and second.rewrite.status == "skipped" and second.multi_query.reason == "analysis_degraded"
    history += [{"role": "user", "content": second.query}, {"role": "assistant", "content": second.answer}]
    third = await service.answer(object(), query="员工差旅报销审批步骤是什么？", kb_ids=[4], doc_ids=[7], history=history, rerank=False)
    assert third.routing.action == "retrieve" and third.sources
    assert len(calls) == 4 and len(client.calls) == 2
    assert "星河项目" not in client.calls[-1]["messages"][1]["content"]


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


@pytest.mark.asyncio
async def test_router_runs_once_after_rewrite_and_reanalysis(analysis_factory):
    original, updated = "它是谁负责？", "Aurora-KB 是谁负责？"
    events, calls = [], []
    dense, bm25, graph = searches(calls)
    analyses = iter([
        analysis_factory(query=original, reference=True, rewrite=True),
        analysis_factory(query=updated, path="hybrid_graph", names=["Aurora-KB"]),
    ])
    class Analyzer:
        async def analyze(self, request):
            events.append("analyze")
            return next(analyses)
        async def aclose(self):
            pass
    class Rewriter:
        async def rewrite(self, request, analysis):
            events.append("rewrite")
            return RewriteResult(RewriteSummary("query_rewrite", "rewritten", updated, ("context_completion",)))
        async def aclose(self):
            pass
    async def coverage(*args, **kwargs):
        events.append("coverage")
        assert kwargs["entity_names"] == ["Aurora-KB"]
        return GraphCoverage("covered", ("resolved-uid",))
    class Router(QueryRouterService):
        async def route(self, *args, **kwargs):
            events.append("route")
            return await super().route(*args, **kwargs)
    client = Client()
    service = RagAnswerService(
        client=client, dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=Analyzer, rewrite_factory=Rewriter, query_router=Router(graph_coverage=coverage),
    )
    result = await service.answer(object(), query=original, kb_ids=[4], doc_ids=[7], rerank=False,
                                  history=[{"role": "user", "content": "Aurora-KB"}])
    assert events == ["analyze", "rewrite", "analyze", "route", "coverage"]
    assert all(kwargs["query"] == updated for _, kwargs in calls)
    assert all(kwargs["kb_ids"] == [4] and kwargs["doc_ids"] == [7] for _, kwargs in calls)
    assert result.query == original and result.rewrite.route_changed and result.rewrite.reanalyzed
    prompt = client.calls[0]["messages"][1]["content"]
    assert "原问题：" + original in prompt
    assert "经过校验的独立问题：" + updated in prompt
    assert not result.routing.deferred_features


@pytest.mark.asyncio
async def test_hyde_used_only_for_dense_embedding_and_not_reranking_or_evidence(analysis_factory, analyzer_stub_factory):
    query = "知识零散导致答案偏离重点该怎么改善？"
    document = "HYDE_ONLY 假想语义检索机制。"
    analysis = analysis_factory(query=query, intent="recommendation")
    analysis.retrieval_strategy.rewrite_method = "hyde"
    analysis.retrieval_strategy.use_hyde = True
    calls, ranked_queries, events = [], [], []
    dense, bm25, graph = searches(calls)
    class Rewriter:
        async def rewrite(self, request, analysis):
            events.append("hyde")
            return RewriteResult(RewriteSummary("hyde", "rewritten", query), hypothetical_document=document)
        async def aclose(self):
            pass
    class Reranker:
        async def rerank(self, result, *, top_n):
            ranked_queries.append(result.query)
            return result
        async def aclose(self):
            pass
    client = Client()
    class Router(QueryRouterService):
        async def route(self, *args, **kwargs):
            events.append("route")
            return await super().route(*args, **kwargs)
    service = RagAnswerService(
        client=client, dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis), rewrite_factory=Rewriter, reranker_factory=Reranker,
        query_router=Router(),
    )
    result = await service.answer(object(), query=query, kb_ids=[4], doc_ids=[7])
    assert [name for name, _ in calls] == ["dense", "bm25"]
    assert events == ["route", "hyde"]
    assert calls[0][1]["embedding_text"] == document
    assert "embedding_text" not in calls[1][1] and calls[1][1]["query"] == query
    assert ranked_queries == [query]
    assert "HYDE_ONLY" not in str(client.calls) and "HYDE_ONLY" not in str(result)
    assert result.rewrite.method == "hyde" and not result.rewrite.reanalyzed


@pytest.mark.asyncio
async def test_normalized_question_reaches_router_retrieval_and_generation_without_reanalysis(
    analysis_factory, analyzer_stub_factory,
):
    original, updated = "Aurora-KB 谁负责？", "Aurora-KB 的负责人是谁？"
    requests, calls = [], []
    dense, bm25, graph = searches(calls)
    class Rewriter:
        async def rewrite(self, request, analysis):
            return RewriteResult(RewriteSummary("query_rewrite", "rewritten", updated,
                                                 ("retrieval_normalization",)))
        async def aclose(self):
            pass
    class Router(QueryRouterService):
        async def route(self, *args, **kwargs):
            assert kwargs["request"].query == kwargs["analysis"].original_query == updated
            return await super().route(*args, **kwargs)
    client = Client()
    service = RagAnswerService(
        client=client, dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis_factory(query=original, rewrite=True,
                                                               names=["Aurora-KB"]), requests=requests),
        rewrite_factory=Rewriter, query_router=Router(), multi_query_enabled=False,
    )
    result = await service.answer(object(), query=original, kb_ids=[4], doc_ids=[7], rerank=False)
    assert len(requests) == 1 and not result.rewrite.reanalyzed
    assert all(kwargs["query"] == updated for _, kwargs in calls)
    assert result.execution.query == updated and result.query == original
    prompt = client.calls[0]["messages"][1]["content"]
    assert f"原问题：{original}" in prompt and f"经过校验的独立问题：{updated}" in prompt


@pytest.mark.asyncio
async def test_terminal_router_decision_skips_hyde(analysis_factory, analyzer_stub_factory):
    query = "知识零散导致答案偏离重点该怎么改善？"
    analysis = analysis_factory(query=query, intent="recommendation")
    analysis.retrieval_strategy.rewrite_method = "hyde"
    analysis.retrieval_strategy.use_hyde = True
    class Router(QueryRouterService):
        async def route(self, *args, **kwargs):
            plan = await super().route(*args, **kwargs)
            return replace(plan, decision=replace(plan.decision, path="clarify", action="clarify"),
                           message="请补充场景。")
    def forbidden():
        pytest.fail("终止路由不能生成 HyDE 文档")
    service = RagAnswerService(client=Client(), query_router=Router(), rewrite_factory=forbidden,
                               analyzer_factory=analyzer_stub_factory(analysis))
    result = await service.answer(object(), query=query, kb_ids=[4], doc_ids=[7])
    assert result.answer == "请补充场景。" and result.rewrite.status == "skipped"


@pytest.mark.asyncio
async def test_failed_context_rewrite_clarifies_without_retrieval(analysis_factory, analyzer_stub_factory):
    query = "它是谁负责？"
    calls = []
    dense, bm25, graph = searches(calls)
    class Rewriter:
        async def rewrite(self, request, analysis):
            return RewriteResult(RewriteSummary("query_rewrite", "degraded", query,
                                               degradation_reason="timeout"),
                                 clarification_question="请明确主体。")
        async def aclose(self):
            pass
    service = RagAnswerService(
        client=Client(), dense_search=dense, bm25_search=bm25, graph_search=graph,
        analyzer_factory=analyzer_stub_factory(analysis_factory(query=query, reference=True, rewrite=True)),
        rewrite_factory=Rewriter,
    )
    result = await service.answer(object(), query=query, kb_ids=[4], doc_ids=[7],
                                  history=[{"role": "user", "content": "Aurora-KB"}])
    assert result.routing.action == "clarify" and not calls and not result.sources
    assert result.answer == "请明确主体。"


@pytest.mark.asyncio
async def test_final_exhaustive_analysis_selects_unsupported_without_search(analysis_factory):
    original, updated = "该项目的完整合同清单呢？", "星河项目的完整合同清单是什么？"
    analyses = iter([analysis_factory(query=original, reference=True, rewrite=True),
                     analysis_factory(query=updated, exhaustive=True)])
    calls = []
    class Analyzer:
        async def analyze(self, request):
            return next(analyses)
        async def aclose(self):
            pass
    class Rewriter:
        async def rewrite(self, request, analysis):
            return RewriteResult(RewriteSummary("query_rewrite", "rewritten", updated, ("context_completion",)))
        async def aclose(self):
            pass
    dense, bm25, graph = searches(calls)
    service = RagAnswerService(client=Client(), analyzer_factory=Analyzer, rewrite_factory=Rewriter,
                               dense_search=dense, bm25_search=bm25, graph_search=graph)
    result = await service.answer(object(), query=original, kb_ids=[4], doc_ids=[7],
                                  history=[{"role": "user", "content": "星河项目"}])
    assert result.routing.path == "structured" and result.routing.action == "unsupported"
    assert result.rewrite.route_changed and not calls
