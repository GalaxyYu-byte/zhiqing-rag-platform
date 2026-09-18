import pytest

from zq_rag_app.query_analysis.models import QueryAnalysisRequest
from zq_rag_app.query_analysis.rewrite import RewriteResult, RewriteSummary
from zq_rag_app.services.query_preprocessing_service import prepare_query


def factories(analyses, rewrite_result, calls):
    analyses = iter(analyses)

    class Analyzer:
        async def analyze(self, request):
            calls.append(("analyze", request))
            return next(analyses)

        async def aclose(self):
            calls.append(("close_analyzer", None))

    class Rewriter:
        async def rewrite(self, request, analysis):
            calls.append(("rewrite", request))
            return rewrite_result

        async def aclose(self):
            calls.append(("close_rewriter", None))

    return Analyzer, Rewriter


@pytest.mark.asyncio
async def test_changed_question_reanalyzed_without_history_and_without_loop(analysis_factory):
    original, updated = "它是谁负责？", "Aurora-KB 是谁负责？"
    initial = analysis_factory(query=original, reference=True, rewrite=True)
    final = analysis_factory(query=updated, path="hybrid_graph", names=["Aurora-KB"], rewrite=True)
    result = RewriteResult(RewriteSummary("query_rewrite", "rewritten", updated, ("context_completion",)))
    calls = []
    analyzer, rewriter = factories([initial, final], result, calls)
    prepared = await prepare_query(
        QueryAnalysisRequest(query=original, history=[{"role": "user", "content": "Aurora-KB"}]),
        analyzer_factory=analyzer, rewrite_factory=rewriter,
    )
    assert [c for c, _ in calls] == ["analyze", "close_analyzer", "rewrite", "close_rewriter", "analyze", "close_analyzer"]
    assert prepared.request.query == updated and not prepared.request.history
    assert prepared.request.reference_date.isoformat() == initial.reference_date
    assert prepared.rewrite.route_changed and prepared.rewrite.reanalyzed
    assert initial.query_type.has_context_reference  # 未污染原对象


@pytest.mark.asyncio
async def test_normalization_reuses_grounded_route_features(analysis_factory):
    original, updated = "Aurora-KB 谁负责？", "Aurora-KB 的负责人是谁？"
    initial = analysis_factory(query=original, rewrite=True, names=["Aurora-KB"])
    calls = []
    analyzer, rewriter = factories([initial], RewriteResult(
        RewriteSummary("query_rewrite", "rewritten", updated, ("retrieval_normalization",)),
    ), calls)
    prepared = await prepare_query(QueryAnalysisRequest(query=original), analyzer_factory=analyzer, rewrite_factory=rewriter)
    assert not prepared.rewrite.reanalyzed and not prepared.rewrite.route_changed
    assert prepared.request.query == updated
    assert prepared.analysis.original_query == updated
    prepared.analysis.validate_grounding(prepared.request)
    assert len([c for c, _ in calls if c == "analyze"]) == 1
    assert initial.original_query == original


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [False, True])
async def test_reanalysis_failure_restores_original_or_clarifies(analysis_factory, reference):
    original, updated = "原问题", "补全问题"
    initial = analysis_factory(query=original, rewrite=True, reference=reference)
    failed = analysis_factory(query=updated, status="degraded")
    calls = []
    analyzer, rewriter = factories([initial, failed], RewriteResult(
        RewriteSummary("query_rewrite", "rewritten", updated, ("constraint_explicitization",)),
    ), calls)
    request = QueryAnalysisRequest(query=original, history=[{"role": "user", "content": "主体"}] if reference else [])
    prepared = await prepare_query(request, analyzer_factory=analyzer, rewrite_factory=rewriter)
    assert prepared.request.query == original and prepared.rewrite.retrieval_query == original
    assert prepared.rewrite.degradation_reason == "reanalysis_failed"
    assert prepared.analysis.retrieval_strategy.path == ("clarify" if reference else "hybrid")


@pytest.mark.asyncio
async def test_hyde_is_deferred_until_after_routing(analysis_factory):
    query = "知识零散导致答案偏离重点该怎么改善？"
    initial = analysis_factory(query=query, intent="recommendation")
    initial.retrieval_strategy.rewrite_method = "hyde"
    initial.retrieval_strategy.use_hyde = True
    calls = []
    analyzer, rewriter = factories([initial], RewriteResult(
        RewriteSummary("hyde", "rewritten", query), hypothetical_document="假想机制文档。",
    ), calls)
    prepared = await prepare_query(QueryAnalysisRequest(query=query), analyzer_factory=analyzer, rewrite_factory=rewriter)
    assert not prepared.rewrite.reanalyzed and not prepared.rewrite.route_changed
    assert prepared.request.query == query and prepared.hypothetical_document is None
    assert prepared.analysis.retrieval_strategy.use_hyde
    assert not any(c == "rewrite" for c, _ in calls)
    assert len([c for c, _ in calls if c == "analyze"]) == 1


@pytest.mark.asyncio
async def test_normalization_with_changed_extraction_evidence_reanalyzes(analysis_factory):
    from zq_rag_app.query_analysis.models import QueryKeyword
    original, updated = "Aurora-KB 谁负责？", "Aurora-KB 的负责人是谁？"
    initial = analysis_factory(query=original, rewrite=True, names=["Aurora-KB"])
    initial.keywords = [QueryKeyword(text="负责", kind="relation", source="query",
                                    history_index=None, evidence_quote="谁负责", confidence=0.9)]
    calls = []
    analyzer, rewriter = factories([initial, analysis_factory(query=updated, names=["Aurora-KB"])],
                                  RewriteResult(RewriteSummary("query_rewrite", "rewritten", updated,
                                                               ("retrieval_normalization",))), calls)
    prepared = await prepare_query(QueryAnalysisRequest(query=original), analyzer_factory=analyzer,
                                   rewrite_factory=rewriter)
    assert prepared.rewrite.reanalyzed and prepared.request.query == updated
    assert len([c for c, _ in calls if c == "analyze"]) == 2


@pytest.mark.asyncio
async def test_known_subject_completion_updates_features_without_analyzer(analysis_factory):
    from zq_rag_app.query_analysis.models import QueryEntity
    original, updated = "他负责什么？", "张三负责什么？"
    initial = analysis_factory(query=original, reference=True, rewrite=True)
    initial.entities = [QueryEntity(name="张三", entity_type="PERSON", source="history",
                                   history_index=0, evidence_quote="张三是负责人", confidence=0.9)]
    calls = []
    analyzer, rewriter = factories([initial], RewriteResult(RewriteSummary(
        "query_rewrite", "rewritten", updated, ("context_completion",),
    )), calls)
    prepared = await prepare_query(
        QueryAnalysisRequest(query=original, history=[{"role": "user", "content": "张三是负责人"}]),
        analyzer_factory=analyzer, rewrite_factory=rewriter,
    )
    assert not prepared.rewrite.reanalyzed and prepared.rewrite.route_changed
    assert not prepared.analysis.query_type.has_context_reference and not prepared.request.history
    assert prepared.analysis.entities[0].source == "query"
    assert prepared.analysis.entities[0].history_index is None
    prepared.analysis.validate_grounding(prepared.request)
    assert initial.entities[0].source == "history" and initial.query_type.has_context_reference
    assert len([c for c, _ in calls if c == "analyze"]) == 1


@pytest.mark.asyncio
async def test_no_rewrite_never_constructs_rewriter(analysis_factory, analyzer_stub_factory):
    analysis = analysis_factory()
    def forbidden():
        raise AssertionError("不应初始化 LLM")
    prepared = await prepare_query(QueryAnalysisRequest(query=analysis.original_query),
                                   analyzer_factory=analyzer_stub_factory(analysis), rewrite_factory=forbidden)
    assert prepared.rewrite.status == "skipped" and prepared.rewrite_ms == 0
