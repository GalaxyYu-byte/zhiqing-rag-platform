"""查询计划契约、确定性分流及可信范围边界；不调用模型或数据库。"""

import json
from dataclasses import replace

import pytest
from pydantic import TypeAdapter, ValidationError

from zhiqing_rag.query_analysis.models import (
    GroundingError,
    QueryAnalysisRequest,
    QueryAnalysisResult,
    QueryConstraint,
    QueryFeatures,
)
from zhiqing_rag.query_routing.models import (
    ChatPlan,
    ClarifyParams,
    ClarifyPlan,
    GraphParams,
    GraphPlan,
    GraphProjectionRef,
    HybridPlan,
    QueryPlan,
    RetrievalParams,
    StructuredParams,
    StructuredPlan,
)
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument
from zhiqing_rag.query_routing.service import GraphReadiness, QueryRouterService

QUERY = "工单关闭前需要谁确认？"
PLAN_ADAPTER = TypeAdapter(QueryPlan)


def term(text, *, source="query", index=None, quote=None):
    return {
        "text": text,
        "evidence": {"source": source, "history_index": index, "quote": quote or text},
    }


def features(**changes):
    return QueryFeatures.model_validate(
        {
            "intent": "fact",
            "entities": [],
            "keywords": [],
            "constraints": [],
            "relation_required": False,
            "needs_context": False,
            "clarification_question": None,
            **changes,
        }
    )


def analysis(query, **changes):
    return QueryAnalysisResult(
        status="ok",
        original_query=query,
        analysis=features(**changes),
        fallback_hint=None,
        message=None,
        degradation_reason=None,
        model="test",
        latency_ms=0,
    )


def scope(*, published=True):
    return AccessScope(
        tenant_id=1,
        user_id=2,
        knowledge_base_ids=(3,),
        document_ids=(4,),
        published_documents=(PublishedDocument(4, 3, 5),) if published else (),
    )


def graph_params(**changes):
    return GraphParams(
        seed_entity_uids=("工单_UID",),
        projections=(
            GraphProjectionRef(
                tenant_id="1",
                document_id="4",
                generation_id="5",
                projection_id="6",
                namespace="tenant_1_document_4_generation_5",
            ),
        ),
        **changes,
    )


def graph_readiness(query=QUERY, *, entities=("工单",), params=None):
    return GraphReadiness(
        query=query,
        matched_entities=entities,
        params=params or graph_params(),
    )


def constraint(*, field="other", operator="unspecified", value="关闭前", quote=None):
    return QueryConstraint.model_validate(
        {
            "field": field,
            "operator": operator,
            "value": value,
            "evidence": {"source": "query", "history_index": None, "quote": quote or value},
        }
    )


def route(query, *, router=None, access=None, history=(), graph=None, **changes):
    return (router or QueryRouterService()).route(
        request=QueryAnalysisRequest(query=query, history=list(history)),
        analysis=analysis(query, **changes),
        scope=access or scope(),
        graph=graph,
    )


def plans():
    common = {"original_query": QUERY, "query": QUERY, "reason_code": "TEST"}
    return [
        HybridPlan(**common, requested_route="hybrid"),
        GraphPlan(**common, requested_route="hybrid_graph", params=graph_params()),
        StructuredPlan(
            **common,
            requested_route="structured",
            params=StructuredParams(operation="count_documents"),
        ),
        ClarifyPlan(
            **common,
            requested_route="clarify",
            params=ClarifyParams(question="请指定对象。"),
        ),
        ChatPlan(**common, requested_route="none"),
    ]


@pytest.mark.parametrize("plan", plans(), ids=lambda plan: plan.route)
def test_all_five_plan_variants_roundtrip_with_discriminator(plan):
    restored = PLAN_ADAPTER.validate_json(plan.model_dump_json())
    assert type(restored) is type(plan)
    assert restored == plan
    assert restored.version == 1


@pytest.mark.parametrize(
    "route_name,params",
    [
        ("hybrid", {"question": "对象是谁？"}),
        ("hybrid_graph", {"candidate_k": 30, "top_k": 5, "rerank": True}),
        ("structured", {"operation": "execute_sql", "sql": "SELECT 1"}),
        ("clarify", {"operation": "count_documents"}),
        ("none", {"sql": "SELECT 1"}),
        ("invented", {}),
    ],
)
def test_plan_rejects_unknown_routes_and_mixed_or_arbitrary_params(route_name, params):
    payload = {
        "original_query": QUERY,
        "query": QUERY,
        "requested_route": "hybrid",
        "reason_code": "TEST",
        "route": route_name,
        "params": params,
    }
    with pytest.raises(ValidationError):
        PLAN_ADAPTER.validate_json(json.dumps(payload, ensure_ascii=False))


@pytest.mark.parametrize(
    "values",
    [
        {"candidate_k": 0},
        {"candidate_k": 101},
        {"top_k": 0},
        {"candidate_k": 10, "top_k": 11},
        {"rerank": "true"},
        {"top_k": True},
    ],
)
def test_retrieval_budget_is_bounded_and_strict(values):
    with pytest.raises(ValidationError):
        RetrievalParams(**values)


@pytest.mark.parametrize("values", [{"max_hops": 0}, {"max_hops": 3}])
def test_graph_hop_budget_is_bounded(values):
    with pytest.raises(ValidationError):
        graph_params(**values)


def test_graph_params_require_confirmed_seeds_and_projections():
    values = graph_params().model_dump()
    for field in ("seed_entity_uids", "projections"):
        with pytest.raises(ValidationError):
            GraphParams.model_validate({**values, field: ()})


@pytest.mark.parametrize(
    "values",
    [
        {"offset": -1},
        {"offset": 100001},
        {"limit": 0},
        {"limit": 101},
        {"operation": "sum_amount"},
        {"scope": "all_tenants"},
    ],
)
def test_structured_params_only_allow_whitelisted_operations_and_safe_pagination(values):
    with pytest.raises(ValidationError):
        StructuredParams(**{"operation": "list_documents", **values})


def test_unsupported_or_empty_plans_require_explicit_message():
    common = {
        "original_query": QUERY,
        "query": QUERY,
        "requested_route": "hybrid",
        "reason_code": "TEST",
    }
    for status in ("unsupported", "empty"):
        with pytest.raises(ValidationError):
            HybridPlan(**common, planning_status=status)


def test_operationless_structured_plan_must_be_explicitly_unsupported():
    common = {
        "original_query": QUERY,
        "query": QUERY,
        "requested_route": "structured",
        "reason_code": "TEST",
        "params": StructuredParams(operation=None),
    }
    with pytest.raises(ValidationError):
        StructuredPlan(**common)
    plan = StructuredPlan(**common, planning_status="unsupported", message="业务统计尚未实现。")
    assert plan.params.operation is None


def test_complete_question_preserves_raw_query_and_server_retrieval_budget():
    budget = RetrievalParams(candidate_k=40, top_k=8, rerank=False)
    plan = route(
        "制度 ABC-001 的适用范围是什么？",
        router=QueryRouterService(retrieval_params=budget),
    )
    assert plan.route == plan.requested_route == "hybrid"
    assert plan.original_query == plan.query == "制度 ABC-001 的适用范围是什么？"
    assert plan.params == budget
    assert plan.planning_status == "planned"
    assert plan.reason_code == "BASE_RETRIEVAL"


@pytest.mark.parametrize(
    "query,operation",
    [
        ("知识库有多少份文档？", "count_documents"),
        ("当前知识库中文档有几篇？", "count_documents"),
        ("文档有多少份？", "count_documents"),
        ("统计知识库文档总数。", "count_documents"),
        ("列出所有文档", "list_documents"),
        ("查看知识库中全部文件！", "list_documents"),
    ],
)
def test_catalog_whitelist_is_identified_but_current_executor_is_not_ready(query, operation):
    plan = route(query, intent="statistics")
    assert plan.route == "structured"
    assert plan.params.operation == operation
    assert plan.planning_status == "unsupported"
    assert plan.reason_code == "CATALOG_EXECUTOR_NOT_READY"
    assert plan.message


@pytest.mark.parametrize(
    "query",
    [
        "统计所有合同的金额总额",
        "去年发生了多少次事故？",
        "按部门统计文档数量",
        "列出所有工单",
        "列出所有文档并按名称排序",
        "列出所有文档然后总结它们的内容",
        "PDF文档有多少份？",
        "知识库有多少份文档以及多少个用户？",
    ],
)
def test_business_statistics_and_unsupported_full_list_require_exact_capability(query):
    plan = route(query, intent="statistics")
    assert plan.route == "structured"
    assert plan.params.operation is None
    assert plan.planning_status == "unsupported"
    assert plan.reason_code == "STRUCTURED_QUERY_UNSUPPORTED"


def test_statistical_method_question_keeps_knowledge_retrieval():
    plan = route("如何统计知识库中的文档数量？", intent="explanation")
    assert plan.route == "hybrid"
    assert plan.reason_code == "BASE_RETRIEVAL"


def test_catalog_condition_is_never_silently_dropped_even_when_executor_is_available():
    query = "知识库有多少份文档？"
    item = constraint(field="document_name", operator="contains", value="文档")
    plan = route(
        query,
        intent="statistics",
        constraints=[item],
        router=QueryRouterService(catalog_available=True),
    )
    assert plan.planning_status == "unsupported"
    assert plan.reason_code == "STRUCTURED_QUERY_UNSUPPORTED"
    assert plan.constraints == (item,)


def test_catalog_available_is_a_trusted_service_capability():
    plan = route(
        "列出所有文档",
        intent="statistics",
        router=QueryRouterService(catalog_available=True),
    )
    assert plan.route == "structured"
    assert plan.params.operation == "list_documents"
    assert plan.planning_status == "planned"
    assert plan.reason_code == "CATALOG_SELECTED"


def test_relation_question_defaults_to_hybrid_until_graph_has_been_checked():
    plan = route(QUERY, intent="procedure", entities=[term("工单")], relation_required=True)
    assert plan.route == "hybrid"
    assert plan.requested_route == "hybrid_graph"
    assert plan.reason_code == "GRAPH_NOT_READY"
    assert plan.warnings


def test_no_graph_entities_keeps_basic_retrieval_without_inventing_seeds():
    plan = route(QUERY, intent="procedure", relation_required=True)
    assert plan.route == plan.requested_route == "hybrid"
    assert plan.reason_code == "BASE_RETRIEVAL"


def test_trusted_graph_readiness_approves_authorized_current_generation():
    ready = graph_readiness()
    plan = route(
        QUERY,
        intent="procedure",
        entities=[term("工单")],
        relation_required=True,
        graph=ready,
    )
    assert plan.route == plan.requested_route == "hybrid_graph"
    assert plan.reason_code == "GRAPH_APPROVED"
    assert plan.params == ready.params


@pytest.mark.parametrize(
    "query,entities",
    [
        ("另一个问题", ("工单",)),
        (QUERY, ()),
        (QUERY, ("合同",)),
        (QUERY, ("工单", "合同")),
    ],
)
def test_graph_readiness_must_be_bound_to_current_query_and_all_entities(query, entities):
    with pytest.raises(ValueError, match="当前问题|全部实体"):
        route(
            QUERY,
            entities=[term("工单")],
            relation_required=True,
            graph=graph_readiness(query, entities=entities),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"tenant_id": "9"},
        {"document_id": "9"},
        {"generation_id": "9"},
    ],
)
def test_graph_projection_cannot_cross_tenant_document_or_active_generation(changes):
    values = graph_params().model_dump()
    reference = values["projections"][0]
    reference.update(changes)
    ready = graph_readiness(params=GraphParams.model_validate(values))
    with pytest.raises(ValueError, match="授权范围|活动批次"):
        route(QUERY, entities=[term("工单")], relation_required=True, graph=ready)


def test_every_graph_projection_must_be_authorized():
    values = graph_params().model_dump()
    values["projections"] = (
        *values["projections"],
        {
            "tenant_id": "1",
            "document_id": "99",
            "generation_id": "5",
            "projection_id": "100",
            "namespace": "unapproved",
        },
    )
    ready = graph_readiness(params=GraphParams.model_validate(values))
    with pytest.raises(ValueError, match="授权范围|活动批次"):
        route(QUERY, entities=[term("工单")], relation_required=True, graph=ready)


@pytest.mark.parametrize(
    "field,operator,value,query",
    [
        ("uploaded_at", "range", "去年", "去年上传的文档是什么？"),
        ("file_type", "neq", "PDF", "不是PDF的资料有哪些？"),
        ("department", "eq", "财务部", "财务部的职责是什么？"),
        ("event_time", "unspecified", "去年", "去年工单处理流程是什么？"),
    ],
)
def test_metadata_constraints_are_preserved_and_blocked_until_filtering_is_supported(
    field,
    operator,
    value,
    query,
):
    item = constraint(field=field, operator=operator, value=value)
    plan = route(query, constraints=[item])
    assert plan.route == "hybrid"
    assert plan.planning_status == "unsupported"
    assert plan.reason_code == "METADATA_FILTERS_NOT_READY"
    assert plan.constraints == (item,)
    assert plan.query == query


def test_other_unspecified_business_condition_stays_in_query_and_plan():
    item = constraint()
    plan = route(
        QUERY,
        intent="procedure",
        entities=[term("工单")],
        relation_required=True,
        constraints=[item],
    )
    assert plan.planning_status == "planned"
    assert plan.constraints == (item,)
    assert plan.constraints[0].field == "other"
    assert plan.constraints[0].operator == "unspecified"
    assert plan.constraints[0].value == "关闭前"
    assert plan.original_query == plan.query == QUERY
    assert any("语义证据" in warning for warning in plan.warnings)


def test_history_entity_does_not_mean_independent_query_has_been_rewritten():
    plan = route(
        "它由谁确认？",
        history=(QUERY,),
        intent="procedure",
        entities=[term("工单", source="history", index=0)],
        relation_required=True,
        needs_context=True,
    )
    assert plan.route == "clarify"
    assert plan.reason_code == "CONTEXT_REWRITE_REQUIRED"
    assert plan.query == "它由谁确认？"
    assert "对象名称" in plan.params.question


def test_explicit_clarification_question_is_preserved_with_conditions():
    query = "去年这两份文档哪个有效？"
    item = constraint(value="去年")
    plan = route(
        query,
        constraints=[item],
        needs_context=True,
        clarification_question="请指定要比较的两份文档。",
    )
    assert plan.route == "clarify"
    assert plan.reason_code == "CLARIFICATION_REQUIRED"
    assert plan.params.question == "请指定要比较的两份文档。"
    assert plan.constraints == (item,)


def test_old_context_dependency_does_not_affect_new_complete_question():
    plan = route("工单关闭流程是什么？", intent="procedure", history=("它怎么办？",))
    assert plan.route == "hybrid"


@pytest.mark.parametrize(
    "hint,expected_route,expected_status",
    [
        ("basic_retrieval", "hybrid", "planned"),
        ("clarify", "clarify", "planned"),
        ("unsupported", "structured", "unsupported"),
    ],
)
def test_three_analysis_degradation_hints_are_respected(hint, expected_route, expected_status):
    request = QueryAnalysisRequest(query=QUERY, history=["不应再使用的历史问题"])
    result = QueryAnalysisResult(
        status="degraded",
        original_query=QUERY,
        analysis=None,
        fallback_hint=hint,
        message="分析暂不可用，请补充对象。",
        degradation_reason="timeout",
        model="test",
        latency_ms=0,
    )
    plan = QueryRouterService().route(request=request, analysis=result, scope=scope())
    assert plan.route == expected_route
    assert plan.planning_status == expected_status
    assert plan.original_query == plan.query == QUERY
    assert plan.constraints == ()
    if hint == "unsupported":
        assert plan.params.operation is None
    elif hint == "clarify":
        assert plan.params.question == result.message
    else:
        assert plan.reason_code == "ANALYSIS_DEGRADED"


def test_unbound_analysis_is_rejected_instead_of_routing_another_question():
    with pytest.raises(ValueError, match="不属于当前问题"):
        QueryRouterService().route(
            request=QueryAnalysisRequest(query=QUERY),
            analysis=analysis("另一个问题"),
            scope=scope(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"entities": [term("合同")]},
        {"keywords": [term("关闭", quote="不存在的引文")]},
        {"constraints": [constraint(value="不存在的条件")]},
        {"entities": [term("工单", source="history", index=0)]},
    ],
)
def test_router_rechecks_grounding_even_for_constructed_success_result(changes):
    with pytest.raises(GroundingError):
        route(QUERY, **changes)


@pytest.mark.parametrize(
    "access",
    [
        AccessScope(1, 2, (), (), ()),
        scope(published=False),
    ],
)
def test_no_published_scope_returns_empty_retrieval_plan(access):
    plan = route(QUERY, access=access)
    assert plan.route == "hybrid"
    assert plan.planning_status == "empty"
    assert plan.message


def test_empty_scope_catalog_does_not_claim_query_was_executed():
    plan = route(
        "文档有多少份？",
        intent="statistics",
        access=scope(published=False),
        router=QueryRouterService(catalog_available=True),
    )
    assert plan.route == "structured"
    assert plan.planning_status == "empty"
    assert plan.reason_code == "NO_PUBLISHED_DOCUMENTS"


def test_chat_does_not_require_any_published_documents():
    plan = route("你好", intent="chitchat", access=AccessScope(1, 2, (), (), ()))
    assert plan.route == "none"
    assert plan.planning_status == "planned"


@pytest.mark.parametrize(
    "query,changes",
    [
        ("工单是什么？", {}),
        ("这个document如何使用？", {}),
        ("业务对象是什么？", {"entities": [term("业务对象")]}),
    ],
)
def test_document_or_entity_question_misclassified_as_chat_still_uses_retrieval(query, changes):
    plan = route(query, intent="chitchat", **changes)
    assert plan.route == "hybrid"


def test_access_scope_is_independent_and_does_not_leak_into_plan_contract():
    plan = route(QUERY)
    dumped = plan.model_dump()
    scope_keys = ("tenant_id", "user_id", "knowledge_base_ids", "document_ids")
    assert all(key not in dumped for key in scope_keys)
    with pytest.raises(ValidationError):
        PLAN_ADAPTER.validate_json(json.dumps({**dumped, "tenant_id": "99"}, ensure_ascii=False))


def test_scope_cannot_contain_unapproved_or_duplicate_active_document():
    original = scope()
    for changes in (
        {"published_documents": (PublishedDocument(99, 3, 5),)},
        {"published_documents": (PublishedDocument(4, 99, 5),)},
        {"published_documents": (PublishedDocument(4, 3, 0),)},
        {"published_documents": original.published_documents * 2},
    ):
        with pytest.raises(ValueError):
            replace(original, **changes)
