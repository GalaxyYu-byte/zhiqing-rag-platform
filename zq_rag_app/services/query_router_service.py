"""确定性 Query Router：不调用 LLM，按分析与运行时能力生成执行计划。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..core.config import settings
from ..query_analysis.models import QueryAnalysisRequest, QueryAnalysisResult
from ..query_analysis.policy import select_strategy
from ..query_routing.models import GraphCoverage, RetrievalPlan, RoutingDecision
from .graph_routing_service import check_graph_coverage


class QueryRouterService:
    def __init__(
        self, *, graph_coverage: Callable[..., Any] = check_graph_coverage,
        graph_timeout_seconds: float | None = None,
    ) -> None:
        self.graph_coverage = graph_coverage
        self.graph_timeout_seconds = (
            settings.query_router_graph_timeout_seconds
            if graph_timeout_seconds is None else graph_timeout_seconds
        )
        if not 0 < self.graph_timeout_seconds <= 30:
            raise ValueError("graph_timeout_seconds 必须在 (0, 30] 秒之间")

    async def route(
        self, session: Any, *, analysis: QueryAnalysisResult,
        request: QueryAnalysisRequest, kb_ids: list[int], doc_ids: list[int] | None,
    ) -> RetrievalPlan:
        if not kb_ids or any(kb_id <= 0 for kb_id in kb_ids):
            raise ValueError("路由必须指定有效的知识库范围")
        if doc_ids is not None and (not doc_ids or any(doc_id <= 0 for doc_id in doc_ids)):
            raise ValueError("授权文档范围不能为空或包含无效 ID")
        if analysis.original_query != request.query:
            raise ValueError("Analyzer 结果不属于当前问题")
        # 不信任外部构造的建议权重或路径；从分析特征重算，并禁止改写共享分析对象。
        analysis = analysis.model_copy(deep=True)
        warnings = list(analysis.warnings)
        deferred = []

        def plan(path, action, reason, *, message=None, query=None, graph_degraded=False,
                 weights=(0.0, 0.0, 0.0), seeds=(), hops=1):
            return RetrievalPlan(
                decision=RoutingDecision(
                    requested_path=analysis.retrieval_strategy.path,
                    path=path, action=action, reason=reason,
                    analyzer_status=analysis.status, graph_degraded=graph_degraded,
                    warnings=tuple(dict.fromkeys(warnings)),
                    deferred_features=tuple(deferred),
                ),
                query=query or request.query, message=message,
                dense_weight=weights[0], bm25_weight=weights[1], graph_weight=weights[2],
                seed_entity_uids=tuple(seeds), max_hops=hops,
            )

        if analysis.status == "degraded":
            if request.history:
                return plan("clarify", "clarify", "analysis_degraded_with_history", message=(
                    "暂时无法可靠分析历史指代，请把对象名称和条件写成一个完整问题。"
                ))
            return plan("hybrid", "retrieve", "analysis_degraded", weights=(0.5, 0.5, 0.0))

        analysis.validate_grounding(request)
        strategy, policy_warnings = select_strategy(analysis, request)
        warnings.extend(policy_warnings)
        if strategy.use_query_rewrite:
            deferred.append("query_rewrite")
        if strategy.use_multi_query:
            deferred.append("multi_query")
        path = strategy.path
        if path == "clarify":
            return plan(path, "clarify", "clarification_required", message=analysis.query_type.clarification_question)
        if path == "structured":
            return plan(path, "unsupported", "structured_not_supported", message=(
                "当前暂不支持精确统计或保证完整的全量查询，无法仅依据部分检索结果给出可靠总数、"
                "总额或完整清单。请缩小到具体文档或事实问题。"
            ))
        if path == "none":
            return plan(path, "chat", "no_retrieval_required")

        # Rewrite 尚未实现，不能把需要语义补全的指代问题伪装成独立检索问题。
        if analysis.query_type.has_context_reference:
            return plan("clarify", "clarify", "context_rewrite_required", message=(
                "请把上文中的对象名称和本次条件写进完整问题，当前尚未启用上下文问题改写。"
            ))
        if deferred:
            warnings.append("改写或多查询执行器尚未接入，本次使用原问题单次检索。")

        exact = analysis.query_type.primary == "exact" or any(k.kind == "exact" for k in analysis.keywords)
        hybrid_weights = (0.3, 0.7, 0.0) if exact else (0.5, 0.5, 0.0)
        if path == "hybrid":
            return plan(path, "retrieve", "hybrid_selected", weights=hybrid_weights)

        if doc_ids is None:
            # 图谱预检查必须使用经过权限服务求交集的具体文档 ID，不能扩大到整个 KB。
            warnings.append("缺少明确授权文档范围，跳过图谱分支。")
            return plan("hybrid", "retrieve", "graph_scope_required", weights=hybrid_weights, graph_degraded=True)
        names = list(dict.fromkeys(entity.name for entity in analysis.entities))
        try:
            async with asyncio.timeout(self.graph_timeout_seconds):
                coverage: GraphCoverage = await self.graph_coverage(
                    session, entity_names=names, kb_ids=kb_ids, doc_ids=doc_ids,
                )
        except TimeoutError:
            warnings.append("图谱实体匹配或覆盖检查超时，降级到 Dense + BM25。")
            return plan("hybrid", "retrieve", "graph_check_timeout", weights=hybrid_weights, graph_degraded=True)
        except Exception:
            warnings.append("图谱实体匹配或覆盖检查不可用，降级到 Dense + BM25。")
            return plan("hybrid", "retrieve", "graph_check_unavailable", weights=hybrid_weights, graph_degraded=True)
        if coverage.status == "ambiguous":
            return plan("clarify", "clarify", "graph_entity_ambiguous", message=(
                "当前范围内存在同名实体，请补充项目、部门或其他限定信息以明确对象。"
            ))
        if coverage.status != "covered" or not coverage.seed_entity_uids:
            warnings.append("实体未完整匹配到当前授权文档的活动图谱事实，使用基础混合检索。")
            return plan("hybrid", "retrieve", "graph_not_covered", weights=hybrid_weights, graph_degraded=True)
        warnings.append("活动图谱证据存在，不代表已证明问题的全部关系；最终回答仍以原文证据为准。")
        return plan("hybrid_graph", "retrieve", "graph_covered", weights=(0.4, 0.3, 0.3),
                    seeds=coverage.seed_entity_uids, hops=2 if analysis.query_type.potential_multi_hop else 1)
