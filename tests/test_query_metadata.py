"""过滤的日期边界、租户解析、SQL 参数化与多路召回一致性。"""

import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from types import SimpleNamespace

import psycopg
import pytest
from pydantic import ValidationError
from test_query_retrieval_repository import (
    VECTOR,
    FakeConnection,
    chunk,
    identity,
    repository,
    scope,
)
from test_query_retrieval_repository import fake_database as retrieval_database
from test_query_retrieval_service import FakeRepository, execute, hit, plan, service
from test_query_router import analysis, constraint, term

from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_analysis.models import QueryAnalysisRequest
from zhiqing_rag.query_metadata import resolver as resolver_module
from zhiqing_rag.query_metadata.compiler import compile_filters
from zhiqing_rag.query_metadata.models import MetadataFilter, MetadataFilterError, ResolvedFilter
from zhiqing_rag.query_metadata.resolver import MetadataFilterResolver, resolve_date
from zhiqing_rag.query_retrieval import graph as graph_module
from zhiqing_rag.query_retrieval.graph import OnlineGraphRepository
from zhiqing_rag.query_routing.models import HybridPlan, PlanQueryBody, StructuredPlan
from zhiqing_rag.query_routing.planning import plan_with_graph
from zhiqing_rag.query_routing.service import QueryRouterService
from zhiqing_rag.query_structured.models import StructuredResult

NOW = datetime(2026, 10, 1, 17, tzinfo=UTC)  # 上海已是 10 月 2 日。
fake_database = retrieval_database


def raw(field, value, operator="eq"):
    return MetadataFilter(field=field, operator=operator, value=value)


@pytest.mark.parametrize(
    ("field", "value", "operator", "expected"),
    [
        ("uploaded_at", "2025年", "eq", (
            "2024-12-31T16:00:00+00:00", "2025-12-31T16:00:00+00:00",
        )),
        ("uploaded_at", "2024年2月", "range", (
            "2024-01-31T16:00:00+00:00", "2024-02-29T16:00:00+00:00",
        )),
        ("event_time", "2024-02-28 至 2024-02-29", "range", (
            "2024-02-28", "2024-03-01",
        )),
        ("uploaded_at", "今天", "eq", (
            "2026-10-01T16:00:00+00:00", "2026-10-02T16:00:00+00:00",
        )),
        ("event_time", "上月", "eq", ("2026-09-01", "2026-10-01")),
        ("event_time", "2025年", "gt", ("2026-01-01",)),
        ("uploaded_at", "2025-01-01T10:00:00+08:00", "gte", (
            "2025-01-01T02:00:00+00:00",
        )),
    ],
)
def test_calendar_bounds_are_shanghai_local_and_end_exclusive(field, value, operator, expected):
    result = resolve_date(raw(field, value, operator), NOW)
    assert result.values == expected


@pytest.mark.parametrize("value", ["2025-02-30", "最近", "2025年一季度", "2025-01-01T00:00:00"])
def test_ambiguous_invalid_or_timezone_less_dates_require_clarification(value):
    with pytest.raises(MetadataFilterError):
        resolve_date(raw("uploaded_at", value), NOW)


async def test_department_lookup_is_tenant_bound_and_ambiguous_names_fail():
    calls = []

    async def lookup(tenant_id, name):
        calls.append((tenant_id, name))
        return [{"id": 8}] if name == "客服部" else [{"id": 8}, {"id": 9}]

    resolver = MetadataFilterResolver(clock=lambda: NOW, department_lookup=lookup)
    result = await resolver.resolve(SimpleNamespace(tenant_id=1), [raw("department", "客服部")])
    assert result == (ResolvedFilter(field="department_id", operator="eq", values=("8",)),)
    assert calls == [(1, "客服部")]
    with pytest.raises(MetadataFilterError):
        await resolver.resolve(SimpleNamespace(tenant_id=2), [raw("department", "运营")])


async def test_enum_mime_values_are_normalized_and_conflicting_sources_are_not_overwritten():
    resolver = MetadataFilterResolver(clock=lambda: NOW)
    filters = await resolver.resolve(identity(), [
        raw("file_type", "Excel"), raw("business_status", "生效"),
    ])
    statement, parameters = compile_filters(filters)
    assert "r.mime_type = ANY(%s::text[])" in statement
    assert parameters[0] == [
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ]
    assert parameters[1] == "EFFECTIVE"
    with pytest.raises(MetadataFilterError):
        await resolver.resolve(identity(), [raw("file_type", "PDF"), raw("file_type", "TXT")])


def test_user_text_is_literal_parameter_and_unknown_aliases_are_rejected():
    text = "_%'; DROP TABLE document; --"
    item = ResolvedFilter(field="document_name", operator="contains", values=(text,))
    statement, parameters = compile_filters((item,))
    assert text not in statement
    assert parameters == (text,)
    assert "strpos(lower(r.original_filename), lower(%s))" in statement
    with pytest.raises(ValueError):
        compile_filters((item,), document_alias="doc; DROP TABLE document")
    assert compile_filters(()) == ("TRUE", ())


@pytest.mark.parametrize("values", [
    {"field": "event_date", "operator": "range", "values": ("2026-01-01", "2025-01-01")},
    {"field": "uploaded_at", "operator": "eq", "values": ("2025-01-01T00:00:00",)},
    {"field": "department_id", "operator": "contains", "values": (1,)},
    {"field": "metadata.secret", "operator": "eq", "values": ("value",)},
])
def test_resolved_contract_cannot_accept_unknown_fields_or_wrong_types(values):
    with pytest.raises(ValidationError):
        ResolvedFilter(**values)


async def test_metadata_is_applied_before_dense_bm25_limits_and_on_revalidation(fake_database):
    template, connections, _ = fake_database
    template.rows = [{**asdict(chunk()), "score": 0.75}]
    condition = ResolvedFilter(field="file_type", operator="eq", values=("application/pdf",))
    filtered_scope = replace(scope(), resolved_filters=(condition,))
    repo = repository()
    await repo.dense(identity(), filtered_scope, VECTOR, 5)
    await repo.bm25(identity(), filtered_scope, "原文", 5)
    await repo.revalidate(identity(), filtered_scope, (chunk(),))
    dense_statement, dense_parameters = connections[0].calls[-1]
    assert dense_statement.index("r.mime_type = %s") < dense_statement.index("ORDER BY")
    assert "application/pdf" in dense_parameters
    bm25_authorized, bm25_parameters = connections[1].calls[2]
    assert "r.mime_type = %s" in bm25_authorized
    assert "LIMIT" not in bm25_authorized
    assert "application/pdf" in bm25_parameters
    final_statement, final_parameters = connections[2].calls[-1]
    assert "r.mime_type = %s" in final_statement
    assert "doc.active_generation_id = c.generation_id" in final_statement
    assert "application/pdf" in final_parameters


async def test_graph_projection_scope_filters_the_active_revision_before_neo4j(monkeypatch):
    connection = FakeConnection()

    @contextmanager
    def database(settings):
        yield connection

    monkeypatch.setattr(graph_module, "database_connection", database)
    date_filter = resolve_date(raw("uploaded_at", "2025年"), NOW)
    await OnlineGraphRepository(repository().settings).projections(
        identity(), replace(scope(), resolved_filters=(date_filter,)),
    )
    statement, parameters = connection.calls[-1]
    assert "r.id = gp.revision_id" in statement
    assert "doc.active_generation_id = gp.generation_id" in statement
    assert "r.created_at >= %s::timestamptz" in statement
    assert date_filter.values[0] in parameters
    assert "gp.status = 'READY'" in statement


async def test_graph_readiness_and_its_fallback_keep_the_resolved_filter_scope():
    class Graph:
        settings = SimpleNamespace(query_graph_timeout_seconds=1)
        scope = None

        async def readiness(self, **kwargs):
            self.scope = kwargs["scope"]
            return None

    graph = Graph()
    query = "PDF工单由谁确认"
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query),
        analysis=analysis(query, intent="relationship", entities=[term("工单")],
                          relation_required=True, constraints=[
                              constraint(field="file_type", operator="eq", value="PDF"),
                          ]),
        identity=identity(), scope=scope(), planner=QueryRouterService(metadata_available=True),
        graph_service=graph, metadata_resolver=MetadataFilterResolver(clock=lambda: NOW),
    )
    assert result.route == "hybrid"
    assert result.reason_code == "GRAPH_NOT_READY"
    assert graph.scope.resolved_filters == result.resolved_filters
    assert result.resolved_filters[0].values == ("application/pdf",)
    assert not any("不转换为目录筛选" in warning for warning in result.warnings)


async def test_retrieval_service_binds_plan_filters_on_every_recall_and_final_check():
    condition = ResolvedFilter(field="file_type", operator="eq", values=("application/pdf",))
    repo = FakeRepository(dense=(hit(),))
    result = await execute(service(repository=repo), query_plan=plan(resolved_filters=(condition,)))
    assert result.status == "ok"
    assert all(call[2].resolved_filters == (condition,) for call in repo.calls)
    assert all(call[1].resolved_filters == (condition,) for call in repo.validation_calls)


async def test_filter_resolution_enables_a_fully_covered_catalog_request_and_pagination():
    query = "统计2025年PDF文档数量"
    features = [
        constraint(field="uploaded_at", operator="range", value="2025年"),
        constraint(field="file_type", operator="eq", value="PDF"),
    ]
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query),
        analysis=analysis(query, intent="statistics", constraints=features),
        identity=identity(), scope=scope(),
        planner=QueryRouterService(metadata_available=True, catalog_available=True),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW), offset=40, limit=10,
    )
    assert isinstance(result, StructuredPlan)
    assert result.planning_status == "planned"
    assert result.params.operation == "count_documents"
    assert (result.params.offset, result.params.limit) == (40, 10)
    assert len(result.resolved_filters) == 2
    assert result.constraints == tuple(features)


async def test_uncaptured_business_or_grouping_request_is_not_removed_from_statistics():
    query = "按部门统计2025年PDF文档数量"
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query),
        analysis=analysis(query, intent="statistics", constraints=[
            constraint(field="uploaded_at", operator="eq", value="2025年"),
            constraint(field="file_type", operator="eq", value="PDF"),
        ]),
        identity=identity(), scope=scope(),
        planner=QueryRouterService(metadata_available=True, catalog_available=True),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW),
    )
    assert result.planning_status == "unsupported"


async def test_invalid_filter_never_falls_back_to_unfiltered_retrieval():
    query = "最近的文档说明"
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query),
        analysis=analysis(query, constraints=[
            constraint(field="uploaded_at", operator="eq", value="最近"),
        ]),
        identity=identity(), scope=scope(), planner=QueryRouterService(metadata_available=True),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW),
    )
    assert result.route == "clarify"
    assert result.resolved_filters == ()


async def test_explicit_filters_survive_degraded_analyzer():
    query = "说明制度"
    degraded = analysis(query).model_copy(update={
        "status": "degraded", "analysis": None, "fallback_hint": "basic_retrieval",
        "message": "暂不可用", "degradation_reason": "disabled",
    })
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query), analysis=degraded,
        identity=identity(), scope=scope(), planner=QueryRouterService(metadata_available=True),
        filters=(raw("file_type", "PDF"),),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW),
    )
    assert isinstance(result, HybridPlan)
    assert result.resolved_filters == (
        ResolvedFilter(field="file_type", operator="eq", values=("application/pdf",)),
    )


@pytest.mark.parametrize(("query", "conditions"), [
    ("业务状态为生效的文档有多少份", [
        constraint(field="business_status", operator="eq", value="生效"),
    ]),
    ("去年上传的财务部PDF有多少份文档", [
        constraint(field="uploaded_at", operator="range", value="去年"),
        constraint(field="department", operator="eq", value="财务部"),
        constraint(field="file_type", operator="eq", value="PDF"),
    ]),
])
async def test_supported_natural_catalog_templates_do_not_leave_metadata_labels(query, conditions):
    async def lookup(tenant_id, value):
        return [{"id": 7}]

    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query),
        analysis=analysis(query, intent="statistics", constraints=conditions),
        identity=identity(), scope=scope(),
        planner=QueryRouterService(catalog_available=True, metadata_available=True),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW, department_lookup=lookup),
    )
    assert result.planning_status == "planned"
    assert result.params.operation == "count_documents"


@pytest.mark.parametrize("query", [
    "不是PDF的文档有多少份", "统计PDF文档数量并按名称排序",
    "统计PDF或TXT文档数量", "统计PDF合同文档数量",
    "统计2025年之后PDF文档数量",
])
async def test_negation_sort_topics_or_and_time_comparisons_cannot_be_silently_removed(query):
    conditions = [constraint(field="file_type", operator="eq", value="PDF")]
    if "2025年" in query:
        conditions.append(constraint(field="uploaded_at", operator="eq", value="2025年"))
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query),
        analysis=analysis(query, intent="statistics", constraints=conditions),
        identity=identity(), scope=scope(),
        planner=QueryRouterService(catalog_available=True, metadata_available=True),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW),
    )
    assert result.planning_status == "unsupported"


async def test_misbound_analysis_is_rejected_before_any_department_lookup():
    async def unexpected_lookup(tenant_id, value):
        pytest.fail("不属于当前问题的分析不能触发查部门")

    with pytest.raises(ValueError, match="分析结果不属于当前问题"):
        await plan_with_graph(
            request=QueryAnalysisRequest(query="当前问题"),
            analysis=analysis("财务部文档", constraints=[
                constraint(field="department", operator="eq", value="财务部"),
            ]),
            identity=identity(), scope=scope(), planner=QueryRouterService(metadata_available=True),
            metadata_resolver=MetadataFilterResolver(department_lookup=unexpected_lookup),
        )


def test_department_database_errors_are_redacted_and_do_not_become_no_filter(monkeypatch):
    @contextmanager
    def failing_database(settings):
        raise psycopg.OperationalError("password=secret internal host unreachable")
        yield

    monkeypatch.setattr(resolver_module, "database_connection", failing_database)
    with pytest.raises(AuthError) as error:
        MetadataFilterResolver(repository().settings)._departments(1, "财务部")
    assert error.value.status_code == 503
    assert "secret" not in str(error.value)


def test_explicit_http_contract_caps_filters_and_pagination():
    body = PlanQueryBody(query="多少份文档", filters=[raw("file_type", "PDF")])
    assert (body.offset, body.limit) == (0, 20)
    with pytest.raises(ValidationError):
        PlanQueryBody(query="多少份文档", limit=101)
    with pytest.raises(ValidationError):
        PlanQueryBody(query="多少份文档", filters=[raw("file_type", "PDF")] * 21)


@pytest.mark.parametrize("filters", [
    [raw("uploaded_at", "2026年", "gte"), raw("uploaded_at", "2025年", "lte")],
    [raw("event_time", "2025年"), raw("event_time", "2026-01-01 至 2026-12-31", "range")],
    [raw("uploaded_at", "2025-01-01T00:00:00+08:00", "gt"),
     raw("uploaded_at", "2024-12-31T16:00:00+00:00", "lte")],
    [raw("uploaded_at", "2025-01-01T00:00:00+08:00", "eq"),
     raw("uploaded_at", "2024-12-31T16:00:00+00:00", "neq")],
    [raw("file_type", "PDF"), raw("file_type", "PDF", "neq")],
    [raw("business_status", "生效"), raw("business_status", "有效", "neq")],
    [raw("document_name", "合同.pdf"), raw("document_name", "合同.pdf", "neq")],
])
async def test_empty_date_intersections_and_eq_neq_conflicts_require_clarification(filters):
    query = "说明制度"
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query), analysis=analysis(query),
        identity=identity(), scope=scope(), planner=QueryRouterService(metadata_available=True),
        filters=filters, metadata_resolver=MetadataFilterResolver(clock=lambda: NOW),
    )
    assert result.route == "clarify"
    assert "条件" in result.params.question
    assert result.resolved_filters == ()


@pytest.mark.parametrize("filters", [
    [raw("file_type", "Excel"), raw("file_type", "XLSX")],
    [raw("file_type", "Excel"), raw("file_type", "XLSX"), raw("file_type", "XLS", "neq")],
    [raw("file_type", "Excel"), raw("file_type", "XLSX", "neq")],
    [raw("uploaded_at", "2025年"), raw("uploaded_at", "2025-07-01 至 2025-12-31", "range")],
    [raw("event_time", "2025-01-01", "gte"), raw("event_time", "2025-01-01", "lte")],
    [raw("event_time", "2025年"), raw("uploaded_at", "2026年")],
])
async def test_overlapping_sets_and_date_intervals_are_retained_as_conjunctions(filters):
    result = await MetadataFilterResolver(clock=lambda: NOW).resolve(identity(), filters)
    assert len(result) == len(filters)
    statement, parameters = compile_filters(result)
    assert statement != "TRUE"
    assert parameters


@pytest.mark.parametrize("connector", ["或", "或者", " OR ", " or "])
async def test_metadata_disjunction_never_becomes_an_and_filter(connector):
    query = f"财务部{connector}PDF中的报销规定"

    async def unexpected_lookup(tenant_id, value):
        pytest.fail("已知不支持的 OR 条件不应查部门或继续执行")

    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query),
        analysis=analysis(query, constraints=[
            constraint(field="department", operator="eq", value="财务部"),
            constraint(field="file_type", operator="eq", value="PDF"),
        ]),
        identity=identity(), scope=scope(), planner=QueryRouterService(metadata_available=True),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW,
                                                 department_lookup=unexpected_lookup),
    )
    assert result.route == "clarify"
    assert "或" in result.params.question
    assert result.resolved_filters == ()


@pytest.mark.parametrize("query", [
    "财务部PDF文档里申请或审批怎么办", "申请 OR 审批在财务部PDF资料中怎么办",
    "PDF文档里申请或审批怎么办",
])
async def test_body_disjunction_outside_metadata_spans_keeps_valid_filters(query):
    async def lookup(tenant_id, value):
        return [{"id": 7}]

    conditions = [constraint(field="file_type", operator="eq", value="PDF")]
    if "财务部" in query:
        conditions.append(constraint(field="department", operator="eq", value="财务部"))
    result = await plan_with_graph(
        request=QueryAnalysisRequest(query=query), analysis=analysis(query, constraints=conditions),
        identity=identity(), scope=scope(), planner=QueryRouterService(metadata_available=True),
        metadata_resolver=MetadataFilterResolver(clock=lambda: NOW, department_lookup=lookup),
    )
    assert result.route == "hybrid"
    assert result.planning_status == "planned"
    assert len(result.resolved_filters) == len(conditions)


async def test_large_department_id_is_a_lossless_string_in_json_and_bigint_sql_parameter():
    department_id = 2**63 - 1

    async def lookup(tenant_id, value):
        return [{"id": department_id}]

    resolved = await MetadataFilterResolver(department_lookup=lookup).resolve(
        identity(), [raw("department", "财务部")],
    )
    assert resolved[0].values == (str(department_id),)
    encoded = resolved[0].model_dump_json()
    assert json.loads(encoded)["values"] == [str(department_id)]
    assert ResolvedFilter.model_validate_json(encoded) == resolved[0]
    statement, parameters = compile_filters(resolved)
    assert "doc.department_id = %s" in statement
    assert parameters == (department_id,)
    assert type(parameters[0]) is int


def test_legacy_integer_filters_normalize_in_public_plan_and_private_catalog_snapshot():
    department_id = 9007199254740993
    legacy_filter = {"field": "department_id", "operator": "eq", "values": [department_id]}
    legacy_plan = {**plan().model_dump(mode="json"), "resolved_filters": [legacy_filter]}
    restored_plan = HybridPlan.model_validate_json(json.dumps(legacy_plan))
    assert restored_plan.resolved_filters[0].values == (str(department_id),)
    legacy_result = {
        "operation": "count_documents", "total": 0, "documents": [],
        "applied_filters": [legacy_filter], "executed_at": NOW.isoformat(),
        "provenance": {
            "tenant_id": "1", "user_id": "2", "knowledge_base_ids": ["3"],
            "operation": "count_documents", "applied_filters": [legacy_filter],
            "executed_at": NOW.isoformat(), "sources": [],
        },
    }
    restored = StructuredResult.model_validate_json(json.dumps(legacy_result))
    assert restored.applied_filters[0].values == (str(department_id),)
    assert restored.provenance.applied_filters == restored.applied_filters
    assert json.loads(restored.model_dump_json())["applied_filters"][0]["values"] == [
        str(department_id),
    ]


@pytest.mark.parametrize("value", [
    True, False, 0, -1, 2**63, 1.0, 1.5,
    "0", "-1", "+1", "01", "1.0", "1e3", str(2**63), None,
])
def test_department_ids_reject_boolean_nonpositive_fractional_and_overflow_values(value):
    with pytest.raises(ValidationError):
        ResolvedFilter(field="department_id", operator="eq", values=(value,))
    with pytest.raises(ValidationError):
        ResolvedFilter.model_validate_json(json.dumps({
            "field": "department_id", "operator": "eq", "values": [value],
        }))
