import asyncio
from dataclasses import asdict

import pytest

from zq_rag_app.query_analysis.models import QueryAnalysisRequest, QueryEntity
from zq_rag_app.query_routing.models import GraphCoverage, GraphEntityCandidate
from zq_rag_app.services.query_router_service import QueryRouterService


async def forbidden_coverage(*args, **kwargs):
    raise AssertionError("此分支不应访问图谱")


@pytest.mark.asyncio
@pytest.mark.parametrize("path,intent,query_type,action", [
    ("hybrid", "fact", "semantic", "retrieve"),
    ("clarify", "fact", "ambiguous", "clarify"),
    ("structured", "statistics", "aggregate", "unsupported"),
    ("none", "chitchat", "semantic", "chat"),
])
async def test_layers_without_graph_calls(analysis_factory, path, intent, query_type, action):
    query = "当前问题"
    analysis = analysis_factory(query=query, path=path, intent=intent, query_type=query_type)
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=query), kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == path and plan.decision.action == action
    if action == "retrieve":
        assert (plan.dense_weight, plan.bm25_weight, plan.graph_weight) == (0.5, 0.5, 0)
    else:
        assert (plan.dense_weight, plan.bm25_weight, plan.graph_weight) == (0, 0, 0)
    assert plan.message if action in {"clarify", "unsupported"} else True
    assert plan.query == query


@pytest.mark.asyncio
async def test_graph_coverage_and_scope_passed_to_plan(analysis_factory):
    calls = []

    async def coverage(session, **kwargs):
        calls.append(kwargs)
        return GraphCoverage("covered", ("entity-1",))

    analysis = analysis_factory(path="hybrid_graph", query_type="multi_hop", names=["Aurora-KB"], multi_hop=True)
    plan = await QueryRouterService(graph_coverage=coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == "hybrid_graph"
    assert plan.seed_entity_uids == ("entity-1",) and plan.max_hops == 2
    assert calls == [{"entity_names": ["Aurora-KB"], "kb_ids": [4], "doc_ids": [7],
                      "entity_candidates": [GraphEntityCandidate("Aurora-KB", "SYSTEM")]}]
    assert plan.graph_weight == 0.3
    assert not plan.decision.graph_degraded
    assert plan.kb_ids == (4,) and plan.doc_ids == (7,)
    assert plan.decision.rule_path == "hybrid_graph"


@pytest.mark.asyncio
async def test_router_keeps_model_suggestion_separate_from_rule_decision(analysis_factory):
    analysis = analysis_factory(path="hybrid_graph", names=["Aurora-KB"], query_type="exact")
    analysis.query_type.relation_required = False
    original = analysis.model_dump()
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7], candidate_k=12, top_k=3, rerank=False,
    )
    assert plan.decision.requested_path == "hybrid_graph"
    assert plan.decision.model_suggestion["path"] == "hybrid_graph"
    assert "dense_weight" not in plan.decision.model_suggestion
    assert plan.decision.policy_path == plan.decision.rule_path == plan.decision.path == "hybrid"
    assert plan.decision.rule_reason == "hybrid_selected"
    assert (plan.dense_weight, plan.bm25_weight, plan.graph_weight) == (0.3, 0.7, 0.0)
    assert (plan.candidate_k, plan.top_k, plan.rerank) == (12, 3, False)
    assert analysis.model_dump() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("coverage_result,expected_path,reason", [
    (GraphCoverage("unmatched"), "hybrid", "graph_not_covered"),
    (GraphCoverage("covered"), "hybrid", "graph_not_covered"),
    (GraphCoverage("ambiguous"), "clarify", "graph_entity_ambiguous"),
])
async def test_missing_and_ambiguous_entities(analysis_factory, coverage_result, expected_path, reason):
    async def coverage(*args, **kwargs):
        return coverage_result
    analysis = analysis_factory(path="hybrid_graph", names=["Aurora-KB"])
    plan = await QueryRouterService(graph_coverage=coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == expected_path and plan.decision.reason == reason
    assert plan.decision.graph_degraded == (expected_path == "hybrid")


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_graph_check_unavailable_or_timeout(analysis_factory, timeout):
    async def coverage(*args, **kwargs):
        if timeout:
            await asyncio.sleep(1)
        raise ConnectionError("unavailable")
    analysis = analysis_factory(path="hybrid_graph", names=["Aurora-KB"])
    plan = await QueryRouterService(graph_coverage=coverage, graph_timeout_seconds=0.01).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == "hybrid" and plan.decision.graph_degraded
    assert plan.decision.reason == ("graph_check_timeout" if timeout else "graph_check_unavailable")


@pytest.mark.asyncio
async def test_degraded_without_history_ignores_all_extractions(analysis_factory):
    analysis = analysis_factory(path="hybrid_graph", names=["unsupported"], status="degraded")
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == "hybrid" and plan.decision.reason == "analysis_degraded"
    assert not plan.seed_entity_uids and plan.query == analysis.original_query


@pytest.mark.asyncio
@pytest.mark.parametrize("status,reason", [
    ("degraded", "analysis_degraded_context_required"), ("ok", "context_rewrite_required"),
])
async def test_context_dependency_is_not_silently_retrieved(analysis_factory, status, reason):
    query = "他负责什么？"
    analysis = analysis_factory(query=query, status=status, reference=True)
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis,
        request=QueryAnalysisRequest(query=query, history=[{"role": "user", "content": "张三负责项目。"}]),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == "clarify" and plan.decision.reason == reason
    assert plan.message


@pytest.mark.asyncio
async def test_graph_requires_explicit_accessible_documents(analysis_factory):
    analysis = analysis_factory(path="hybrid_graph", names=["Aurora-KB"])
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=None,
    )
    assert plan.decision.path == "hybrid" and plan.decision.reason == "graph_scope_required"


@pytest.mark.asyncio
async def test_rewrite_and_multi_query_are_reported_as_deferred(analysis_factory):
    analysis = analysis_factory(rewrite=True, multi_query=True)
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.deferred_features == ("query_rewrite", "multi_query")
    assert plan.query == analysis.original_query


@pytest.mark.asyncio
async def test_router_recomputes_weights_without_mutating_analysis(analysis_factory):
    analysis = analysis_factory(query_type="exact", reference=True)
    original = analysis.model_dump()
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == "clarify"
    assert analysis.model_dump() == original
    analysis.query_type.has_context_reference = False
    plan = await QueryRouterService(graph_coverage=forbidden_coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=analysis.original_query),
        kb_ids=[4], doc_ids=[7],
    )
    assert (plan.dense_weight, plan.bm25_weight, plan.graph_weight) == (0.3, 0.7, 0)
    assert asdict(plan)["decision"]["router_version"] == "query-router-v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("kb_ids,doc_ids", [([], [7]), ([0], [7]), ([4], []), ([4], [0])])
async def test_invalid_scope_is_rejected(analysis_factory, kb_ids, doc_ids):
    analysis = analysis_factory()
    with pytest.raises(ValueError):
        await QueryRouterService().route(object(), analysis=analysis,
            request=QueryAnalysisRequest(query=analysis.original_query), kb_ids=kb_ids, doc_ids=doc_ids)


@pytest.mark.asyncio
async def test_query_mismatch_is_rejected(analysis_factory):
    with pytest.raises(ValueError, match="当前问题"):
        await QueryRouterService().route(object(), analysis=analysis_factory(),
            request=QueryAnalysisRequest(query="其他问题"), kb_ids=[4], doc_ids=[7])


@pytest.mark.asyncio
async def test_router_does_not_swallow_cancellation(analysis_factory):
    async def coverage(*args, **kwargs):
        raise asyncio.CancelledError()
    analysis = analysis_factory(path="hybrid_graph", names=["Aurora-KB"])
    with pytest.raises(asyncio.CancelledError):
        await QueryRouterService(graph_coverage=coverage).route(object(), analysis=analysis,
            request=QueryAnalysisRequest(query=analysis.original_query), kb_ids=[4], doc_ids=[7])


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_roles", [True, False])
async def test_project_relation_keeps_graph_when_incidental_technology_is_missing(analysis_factory, explicit_roles):
    query = "采用Neo4j的星河项目由谁负责？"
    analysis = analysis_factory(query=query, path="hybrid_graph")
    analysis.entities = [QueryEntity(
        name=name, entity_type=kind, source="query", history_index=None,
        evidence_quote=query, confidence=0.9, graph_role=role if explicit_roles else None,
    ) for name, kind, role in [("星河项目", "PROJECT", "subject"), ("Neo4j", "TECHNOLOGY", "auxiliary")]]
    async def coverage(session, **kwargs):
        assert kwargs["entity_candidates"] == [GraphEntityCandidate("星河项目"),
                                                GraphEntityCandidate("Neo4j", "TECHNOLOGY", False)]
        return GraphCoverage("partial", ("project",), ("Neo4j",))

    plan = await QueryRouterService(graph_coverage=coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=query), kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.path == "hybrid_graph" and plan.decision.action == "retrieve"
    assert plan.decision.reason == "graph_partial_coverage" and not plan.decision.graph_degraded
    assert plan.seed_entity_uids == ("project",)
    assert any("部分证据" in message for message in plan.decision.warnings)


@pytest.mark.asyncio
async def test_named_technology_can_be_required_relationship_subject(analysis_factory):
    query = "Neo4j依赖哪些服务？"
    analysis = analysis_factory(query=query, path="hybrid_graph")
    analysis.entities = [QueryEntity(name="Neo4j", entity_type="TECHNOLOGY", graph_role="subject",
                                    source="query", history_index=None, evidence_quote=query, confidence=0.9)]
    async def coverage(session, **kwargs):
        assert kwargs["entity_candidates"] == [GraphEntityCandidate("Neo4j", "TECHNOLOGY")]
        return GraphCoverage("ambiguous")

    plan = await QueryRouterService(graph_coverage=coverage).route(
        object(), analysis=analysis, request=QueryAnalysisRequest(query=query), kb_ids=[4], doc_ids=[7],
    )
    assert plan.decision.action == "clarify"


def test_identity_qualifier_requires_verbatim_user_evidence(analysis_factory):
    query = "研发部的张三负责什么？"
    analysis = analysis_factory(query=query, path="hybrid_graph")
    entity = QueryEntity(name="张三", entity_type="PERSON", graph_role="subject", qualifiers=["研发部"],
                         source="query", history_index=None, evidence_quote=query, confidence=0.9)
    analysis.entities = [entity]
    analysis.validate_grounding(QueryAnalysisRequest(query=query))
    entity.qualifiers = ["财务部"]
    with pytest.raises(ValueError, match="限定信息"):
        analysis.validate_grounding(QueryAnalysisRequest(query=query))
