"""确定性 Query Router：不调用 LLM，按分析与运行时能力生成执行计划。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from ..core.config import settings
from ..query_analysis.models import QueryAnalysisRequest, QueryAnalysisResult, RetrievalProposal
from ..query_analysis.context_fallback import requires_context_on_failure
from ..query_analysis.policy import NO_RETRIEVAL_WEIGHTS, SEMANTIC_HYBRID_WEIGHTS, hybrid_weights_for, select_strategy
from ..query_routing.models import GraphCoverage, GraphEntityCandidate, RetrievalPlan, RoutingDecision
from ..graph.models import EntityType
from .graph_routing_service import check_graph_coverage


_GRAPH_ENTITY_TYPES = frozenset(entity_type.value for entity_type in EntityType)


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
        preprocessing_complete: bool = False,
        multi_query_available: bool = False,
        candidate_k: int = 30, top_k: int = 5, rerank: bool = True,
    ) -> RetrievalPlan:
        if not kb_ids or any(kb_id <= 0 for kb_id in kb_ids):
            raise ValueError("路由必须指定有效的知识库范围")
        if doc_ids is not None and (not doc_ids or any(doc_id <= 0 for doc_id in doc_ids)):
            raise ValueError("授权文档范围不能为空或包含无效 ID")
        if analysis.original_query != request.query:
            raise ValueError("Analyzer 结果不属于当前问题")
        if not 1 <= top_k <= candidate_k <= 100:
            raise ValueError("必须满足 1 <= top_k <= candidate_k <= 100")
        # 不信任外部构造的建议权重或路径；从分析特征重算，并禁止改写共享分析对象。
        analysis = analysis.model_copy(deep=True)
        warnings = list(analysis.warnings)
        deferred = []
        suggestion = None
        if analysis.status == "ok":
            suggestion = analysis.model_suggestion or RetrievalProposal.model_validate(
                analysis.retrieval_strategy.model_dump(include=set(RetrievalProposal.model_fields)),
            )
        policy_path = None
        multi_query_selected = False
        hyde_selected = False

        def plan(path, action, reason, *, message=None, query=None, graph_degraded=False,
                 weights=NO_RETRIEVAL_WEIGHTS, seeds=(), hops=1,
                 fallback_weights=SEMANTIC_HYBRID_WEIGHTS[:2]):
            return RetrievalPlan(
                decision=RoutingDecision(
                    requested_path=suggestion.path if suggestion else analysis.retrieval_strategy.path,
                    path=path, action=action, reason=reason,
                    analyzer_status=analysis.status, graph_degraded=graph_degraded,
                    warnings=tuple(dict.fromkeys(warnings)),
                    deferred_features=tuple(deferred),
                    model_suggestion=suggestion.model_dump(mode="json") if suggestion else None,
                    policy_path=policy_path, rule_path=path, rule_reason=reason,
                    rule_weights=weights,
                ),
                query=query or request.query, message=message,
                dense_weight=weights[0], bm25_weight=weights[1], graph_weight=weights[2],
                seed_entity_uids=tuple(seeds), max_hops=hops,
                hybrid_fallback_weights=fallback_weights,
                kb_ids=tuple(kb_ids), doc_ids=tuple(doc_ids) if doc_ids is not None else None,
                candidate_k=candidate_k, top_k=top_k, rerank=rerank,
                use_multi_query=multi_query_selected and action == "retrieve",
                use_hyde=hyde_selected and path == "hybrid" and action == "retrieve",
                graph_timeout_seconds=self.graph_timeout_seconds,
            )

        if analysis.status == "degraded":
            # 有会话历史不代表当前问题依赖上文；否则补充完整问题仍会永久澄清。
            # 故障模型的抽取/标记全部忽略，仅用当前文本的明显指代保护降级检索。
            if requires_context_on_failure(request.query):
                return plan("clarify", "clarify", "analysis_degraded_context_required", message=(
                    "暂时无法可靠分析历史指代，请把对象名称和条件写成一个完整问题。"
                ))
            if request.history:
                warnings.append("分析不可用，忽略会话历史，仅以当前问题执行基础检索。")
            return plan("hybrid", "retrieve", "analysis_degraded", weights=SEMANTIC_HYBRID_WEIGHTS)

        analysis.validate_grounding(request)
        strategy, policy_warnings = select_strategy(analysis, request)
        policy_path = strategy.path
        multi_query_selected = strategy.use_multi_query and multi_query_available
        hyde_selected = strategy.use_hyde
        warnings.extend(policy_warnings)
        if strategy.use_query_rewrite and not preprocessing_complete:
            deferred.append("query_rewrite")
        if strategy.use_hyde and not preprocessing_complete:
            deferred.append("hyde")
        if strategy.use_multi_query and not multi_query_available:
            deferred.append("multi_query")
        path = strategy.path
        if path == "clarify":
            return plan(path, "clarify", "clarification_required", message=analysis.query_type.clarification_question)
        # 独立调用 Router 或补全未成功时，仍禁止检索未解析的指代。
        if analysis.query_type.has_context_reference:
            return plan("clarify", "clarify", "context_rewrite_required", message=(
                "请把上文中的对象名称和本次条件写进完整问题，当前指代尚未可靠解析。"
            ))
        if path == "structured":
            return plan(path, "unsupported", "structured_not_supported", message=(
                "当前暂不支持精确统计或保证完整的全量查询，无法仅依据部分检索结果给出可靠总数、"
                "总额或完整清单。请缩小到具体文档或事实问题。"
            ))
        if path == "none":
            return plan(path, "chat", "no_retrieval_required")
        if deferred:
            warnings.append("部分预处理功能未执行，本次使用当前问题单次检索。")

        hybrid_weights = hybrid_weights_for(analysis)
        if path == "hybrid":
            return plan(path, "retrieve", "hybrid_selected", weights=hybrid_weights)

        if doc_ids is None:
            # 图谱预检查必须使用经过权限服务求交集的具体文档 ID，不能扩大到整个 KB。
            warnings.append("缺少明确授权文档范围，跳过图谱分支。")
            return plan("hybrid", "retrieve", "graph_scope_required", weights=hybrid_weights, graph_degraded=True)
        # 新协议使用语义角色；旧协议不把业务主体旁的技术/概念当作必需种子。
        entities = analysis.entities
        has_subject = any(entity.graph_role == "subject" for entity in entities)
        business_types = {"PERSON", "ORGANIZATION", "DEPARTMENT", "PROJECT", "CONTRACT",
                          "SYSTEM", "SERVICE", "PRODUCT", "POLICY", "EVENT", "IDENTIFIER"}
        has_business = any(entity.entity_type in business_types and entity.graph_role is None for entity in entities)
        candidates = []
        for entity in entities:
            required = (entity.graph_role == "subject" if entity.graph_role is not None else
                        not has_subject and (not has_business or entity.entity_type in business_types))
            candidates.append(GraphEntityCandidate(
                name=entity.name,
                entity_type=entity.entity_type if entity.entity_type in _GRAPH_ENTITY_TYPES else None,
                required=required, qualifiers=tuple(entity.qualifiers),
            ))
        if not any(candidate.required for candidate in candidates):
            warnings.append("未确认图谱关系查询的必需主体，使用基础混合检索。")
            return plan("hybrid", "retrieve", "graph_subject_required", weights=hybrid_weights, graph_degraded=True)
        names = list(dict.fromkeys(candidate.name for candidate in candidates))
        try:
            async with asyncio.timeout(self.graph_timeout_seconds):
                coverage: GraphCoverage = await self.graph_coverage(
                    session, entity_names=names, kb_ids=kb_ids, doc_ids=doc_ids,
                    entity_candidates=candidates,
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
        if coverage.status not in {"covered", "partial"} or not coverage.seed_entity_uids:
            warnings.append("必需主体未匹配到当前授权文档的活动图谱事实，使用基础混合检索。")
            return plan("hybrid", "retrieve", "graph_not_covered", weights=hybrid_weights, graph_degraded=True)
        warnings.append("活动图谱证据存在，不代表已证明问题的全部关系；最终回答仍以原文证据为准。")
        if coverage.status == "partial":
            warnings.append("辅助实体缺失或歧义，已忽略未确认对象；图谱仅提供部分证据，不能证明全部限定条件。")
        return plan("hybrid_graph", "retrieve", "graph_partial_coverage" if coverage.status == "partial" else "graph_covered",
                    weights=(strategy.dense_weight, strategy.bm25_weight, strategy.graph_weight),
                    fallback_weights=hybrid_weights[:2],
                    seeds=coverage.seed_entity_uids, hops=2 if analysis.query_type.potential_multi_hop else 1)
