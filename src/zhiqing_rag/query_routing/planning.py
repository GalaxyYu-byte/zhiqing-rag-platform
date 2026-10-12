"""统一服务端规划：图谱能力由当前身份和授权范围确认，故障降级到混合检索。"""

import asyncio
import logging
import re
from dataclasses import replace
from typing import Annotated

from fastapi import Depends

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.query_analysis.models import QueryAnalysisRequest, QueryAnalysisResult
from zhiqing_rag.query_metadata.models import MetadataFilter, MetadataFilterError
from zhiqing_rag.query_metadata.resolver import MetadataFilterResolver
from zhiqing_rag.query_retrieval.graph import OnlineGraphService

from .models import ClarifyParams, ClarifyPlan, HybridPlan, QueryPlan, StructuredPlan
from .scope import AccessScope
from .service import QueryRouterService

logger = logging.getLogger(__name__)


def _metadata_has_disjunction(query, constraints) -> bool:
    """仅查看有原文依据的元数据值之间的连接词，不检查正文其他位置。"""
    spans = set()
    for item in constraints:
        quote_start = query.find(item.evidence.quote)
        value_start = item.evidence.quote.find(item.value)
        if quote_start >= 0 and value_start >= 0:
            start = quote_start + value_start
            spans.add((start, start + len(item.value)))
    ordered = sorted(spans)
    return any(
        first[1] <= second[0] and re.search(
            r"或者|或|(?<![A-Za-z])or(?![A-Za-z])", query[first[1]:second[0]], re.I,
        ) is not None
        for first, second in zip(ordered, ordered[1:])
    )


def get_graph_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> OnlineGraphService:
    # 构造本身不建立数据库或 Neo4j 连接；普通检索不执行图谱检查。
    return OnlineGraphService(settings)


async def plan_with_graph(
    *,
    request: QueryAnalysisRequest,
    analysis: QueryAnalysisResult,
    identity: AuthIdentity,
    scope: AccessScope,
    planner: QueryRouterService,
    graph_service: OnlineGraphService | None = None,
    filters=(),
    offset: int = 0,
    limit: int = 20,
    metadata_resolver=None,
) -> QueryPlan:
    if request.query != analysis.original_query:
        raise ValueError("分析结果不属于当前问题")
    raw_constraints = ()
    if analysis.status == "ok":
        analysis.analysis.validate_grounding(request)
        raw_constraints = tuple(analysis.analysis.constraints)
    metadata_constraints = tuple(item for item in raw_constraints if item.field != "other")
    if filters or (metadata_constraints and planner.metadata_available):
        try:
            if _metadata_has_disjunction(request.query, metadata_constraints):
                raise MetadataFilterError(
                    "元数据筛选暂不支持“或”组合，请明确使用同时满足的条件或分别查询。",
                )
            normalized = tuple(filters) + tuple(
                MetadataFilter(field=item.field, operator=item.operator, value=item.value)
                for item in metadata_constraints
                if item.operator != "unspecified"
            )
            if any(item.operator == "unspecified" for item in metadata_constraints):
                raise MetadataFilterError("元数据条件缺少比较方式，请明确等于、包含或时间范围。")
            resolver = metadata_resolver or MetadataFilterResolver()
            resolved = await resolver.resolve(identity, normalized)
            if resolved:
                scope = replace(scope, resolved_filters=resolved)
        except MetadataFilterError as error:
            return ClarifyPlan(
                original_query=request.query, query=request.query, requested_route="clarify",
                reason_code="METADATA_FILTER_CLARIFICATION", constraints=raw_constraints,
                params=ClarifyParams(question=str(error)),
            )
    preliminary = planner.route(request=request, analysis=analysis, scope=scope)
    if isinstance(preliminary, StructuredPlan):
        preliminary = preliminary.model_copy(
            update={"params": preliminary.params.model_copy(
                update={"offset": offset, "limit": limit},
            )},
        )
    if not (
        isinstance(preliminary, HybridPlan)
        and preliminary.requested_route == "hybrid_graph"
        and preliminary.planning_status == "planned"
    ):
        return preliminary

    def fallback(reason: str, warning: str) -> HybridPlan:
        # Router 的最后一条 warning 是图谱占位说明，替换成此次检查的实际结果。
        return preliminary.model_copy(
            update={"reason_code": reason, "warnings": (*preliminary.warnings[:-1], warning)}
        )

    if graph_service is None:
        return fallback("GRAPH_NOT_READY", "当前图谱能力不可用，使用基础混合检索。")

    try:
        settings = getattr(graph_service, "settings", None)
        timeout = getattr(settings, "query_graph_timeout_seconds", 3)
        async with asyncio.timeout(timeout):
            graph = await graph_service.readiness(
                identity=identity,
                scope=scope,
                query=request.query,
                entities=tuple(entity.text for entity in analysis.analysis.entities),
                retrieval_params=preliminary.params,
            )
        if graph is None:
            return fallback(
                "GRAPH_NOT_READY",
                "当前授权范围内没有就绪图谱或无法确认全部问题实体，使用基础混合检索。",
            )
        return planner.route(request=request, analysis=analysis, scope=scope, graph=graph)
    except AuthError:
        # 会话和权限错误不能被可选召回分支掩盖。
        raise
    except Exception as error:
        logger.warning("图谱规划检查失败: %s", type(error).__name__)
        return fallback("GRAPH_UNAVAILABLE", "图谱检查暂不可用，已降级到基础混合检索。")
