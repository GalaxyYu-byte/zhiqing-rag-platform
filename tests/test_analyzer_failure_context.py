import pytest

from zq_rag_app.query_analysis.context_fallback import requires_context_on_failure
from zq_rag_app.query_analysis.models import QueryAnalysisRequest
from zq_rag_app.services.query_analyzer_service import QueryAnalyzerService
from zq_rag_app.services.query_router_service import QueryRouterService


@pytest.mark.parametrize("query", [
    "他负责什么？", "她还负责天河项目吗？", "它去年未验收的原因是什么？",
    "请问它的负责人是谁？", "那他是否也负责天河项目？", "该项目的完整合同清单呢？",
    "这个项目谁负责？", "这份文档的审批流程是什么？", "前者的负责人是谁？",
    "刚才提到的项目什么时候验收？", "那去年呢？", "还有呢？", "继续", "再详细一点。",
    "What about it?", "Who owns it?", "What does the previous answer mean?",
])
def test_obvious_context_reference_requires_completion(query):
    assert requires_context_on_failure(query)


@pytest.mark.parametrize("query", [
    "星河项目的负责人是谁？", "HT-2025-001 的金额是多少？", "员工差旅报销需要哪些审批步骤？",
    "其他项目的验收流程是什么？", "吉他如何调音？", "利他行为是什么意思？",
    "张三负责星河项目，他还负责天河项目吗？", "这个项目（星河项目）的负责人是谁？",
    "这个合同 HT-2025-001 的金额是多少？", "该模型 DeepSeek 的调用方式是什么？",
    "Who owns Aurora-KB?", "How do employees apply for annual leave?", "What is IT?", "IT 是什么？",
])
def test_complete_questions_are_not_blocked_by_history(query):
    assert not requires_context_on_failure(query)


async def forbidden_coverage(*args, **kwargs):
    raise AssertionError("降级查询不应执行图谱检查")


@pytest.mark.asyncio
@pytest.mark.parametrize("history", [[], [{"role": "user", "content": "张三负责星河项目。"}]])
async def test_degraded_dependent_query_clarifies_even_without_history(analysis_factory, history):
    query = "他负责什么？"
    analysis = analysis_factory(query=query, status="degraded")  # 故障结果声称没有指代。
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=query, history=history), kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.action == "clarify" and plan.decision.reason == "analysis_degraded_context_required"
    assert not plan.seed_entity_uids


@pytest.mark.asyncio
async def test_degraded_complete_question_ignores_model_flags_and_old_topic(analysis_factory):
    query = "员工差旅报销需要哪些审批步骤？"
    analysis = analysis_factory(query=query, status="degraded", reference=True, path="hybrid_graph",
                                names=["无依据的历史实体"], rewrite=True, multi_query=True)
    original = analysis.model_dump()
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis,
        request=QueryAnalysisRequest(query=query, history=[{"role": "user", "content": "张三负责星河项目。"}]),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == "hybrid" and plan.decision.action == "retrieve"
    assert plan.query == query and plan.decision.reason == "analysis_degraded"
    assert plan.dense_weight == plan.bm25_weight == 0.5 and plan.graph_weight == 0
    assert not plan.seed_entity_uids and not plan.decision.deferred_features
    assert analysis.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["missing_api_key", "timeout", "provider_error", "invalid_output"])
async def test_real_analyzer_fallback_with_history_can_retrieve_complete_query(reason):
    request = QueryAnalysisRequest(query="星河项目的负责人是谁？", history=[
        {"role": "user", "content": "他负责什么？"},
        {"role": "assistant", "content": "请把对象写成完整问题。"},
    ])
    # 使用生产 fallback 构造结果，确认不是只有测试模拟的状态能恢复。
    import time
    from datetime import date
    service = QueryAnalyzerService(api_key="")
    analysis = service._fallback(request, date(2026, 9, 18), time.perf_counter(), 0, reason)
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=request, kb_ids=[4], doc_ids=[7],
    )
    assert analysis.status == "degraded" and analysis.degradation_reason == reason
    assert plan.decision.action == "retrieve" and plan.query == request.query
