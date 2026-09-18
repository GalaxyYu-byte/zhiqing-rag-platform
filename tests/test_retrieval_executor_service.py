from dataclasses import replace

import pytest

from zq_rag_app.query_analysis.models import QueryAnalysisRequest
from zq_rag_app.query_analysis.rewrite import RewriteSummary
from zq_rag_app.query_routing.models import GraphCoverage
from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult
from zq_rag_app.services.hybrid_retrieval_service import fuse_dense_bm25_rrf
from zq_rag_app.services.query_preprocessing_service import PreparedQuery
from zq_rag_app.services.query_router_service import QueryRouterService
from zq_rag_app.services.retrieval_executor_service import RetrievalExecutor
from zq_rag_app.services.retrieval_service import CosineRetrievalResult, RetrievedChunk


@pytest.mark.asyncio
async def test_executor_uses_plan_fallback_without_overwriting_router_plan(analysis_factory):
    analysis = analysis_factory(path="hybrid_graph", names=["Aurora-KB"])
    request = QueryAnalysisRequest(query=analysis.original_query)

    async def coverage(*args, **kwargs):
        return GraphCoverage("covered", ("resolved-uid",))

    plan = await QueryRouterService(graph_coverage=coverage).route(
        object(), analysis=analysis, request=request, kb_ids=[4], doc_ids=[7],
        candidate_k=2, top_k=1, rerank=False,
    )
    # 使用与默认策略不同的回退值，确认 Executor 执行计划而不自行决策。
    plan = replace(plan, hybrid_fallback_weights=(0.2, 0.8))
    first = RetrievedChunk(1, 1, 7, "制度.txt", 0, None, None, "规定一。", 5, 0.9)
    second = replace(first, rank=2, chunk_id=2, content="规定二。", score=0.4)
    dense_result = CosineRetrievalResult(plan.query, "test", 3, 0, [first, second])
    bm25_result = BM25RetrievalResult(plan.query, "test", 0, [
        replace(second, rank=1, score=0.9), replace(first, rank=2, score=0.4),
    ])
    calls = []

    async def dense(*args, **kwargs):
        calls.append(("dense", kwargs))
        return dense_result

    async def bm25(*args, **kwargs):
        calls.append(("bm25", kwargs))
        return bm25_result

    async def graph(*args, **kwargs):
        calls.append(("graph", kwargs))
        raise ConnectionError("unavailable")

    def forbidden():
        raise AssertionError("计划未要求扩展或重排")

    executor = RetrievalExecutor(
        dense_search=dense, bm25_search=bm25, graph_search=graph,
        reranker_factory=forbidden, multi_query_factory=forbidden,
        multi_query_enabled=False, empty_retrieval_expansion=False,
        expansion_retrieval_timeout_seconds=1,
    )
    executed = await executor.execute(
        object(), plan=plan,
        prepared=PreparedQuery(request, analysis, RewriteSummary("none", "skipped", plan.query), None, 0, 0),
    )
    expected = fuse_dense_bm25_rrf(dense_result, bm25_result, top_k=2, dense_weight=0.2, bm25_weight=0.8)
    assert executed.chunks == expected.results[:1]
    assert plan.decision.path == plan.decision.rule_path == "hybrid_graph"
    assert (plan.dense_weight, plan.bm25_weight, plan.graph_weight) == (0.4, 0.3, 0.3)
    assert executed.routing.rule_weights == (0.4, 0.3, 0.3)
    assert executed.execution.path == "hybrid"
    assert executed.execution.weights == (0.2, 0.8, 0.0)
    assert executed.execution.degradation_reasons == ("graph_execution_unavailable",)
    assert [name for name, _ in calls] == ["dense", "bm25", "graph"]
    assert all(kwargs["kb_ids"] == [4] and kwargs["doc_ids"] == [7] and kwargs["top_k"] == 2
               for _, kwargs in calls)
