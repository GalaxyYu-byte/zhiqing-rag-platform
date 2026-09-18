"""Analyzer → 独立问题 → 按需更新分析。HyDE 在 Router 确定后执行。"""

from __future__ import annotations

import time
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from ..query_analysis.models import QueryAnalysisRequest, QueryAnalysisResult, RetrievalProposal
from ..query_analysis.policy import select_strategy
from ..query_analysis.rewrite import RewriteSummary
from ..query_routing.models import RetrievalPlan
from .query_analyzer_service import QueryAnalyzerService
from .query_rewrite_service import QueryRewriteService


def routing_signature(analysis: QueryAnalysisResult, request: QueryAnalysisRequest) -> tuple:
    """比较决策输入而非预执行路由，图谱覆盖仍由最终 Router 查询一次。"""
    value = analysis.model_copy(deep=True)
    strategy, _ = select_strategy(value, request)
    kind = value.query_type
    return (
        strategy.path, strategy.dense_weight, strategy.bm25_weight, strategy.graph_weight,
        kind.primary, kind.has_context_reference, kind.needs_clarification,
        kind.relation_required, kind.potential_multi_hop, kind.requires_exhaustive,
        value.intent.primary, tuple(sorted(value.intent.secondary)),
        tuple(sorted((e.entity_type, e.name, e.graph_role or "", tuple(e.qualifiers)) for e in value.entities)),
        tuple(sorted(k.text for k in value.keywords if k.kind == "exact")),
        tuple(sorted((m.field, m.operator, m.value) for m in value.metadata)),
    )


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    request: QueryAnalysisRequest
    analysis: QueryAnalysisResult
    rewrite: RewriteSummary
    hypothetical_document: str | None
    analysis_ms: int
    rewrite_ms: int


def reuse_grounded_analysis(
    initial: QueryAnalysisResult, request: QueryAnalysisRequest,
    candidate_request: QueryAnalysisRequest, summary: RewriteSummary,
) -> QueryAnalysisResult | None:
    """已通过独立语义审核的表达整理可复用特征；补全或条件变化仍需分析。"""
    if initial.query_type.needs_clarification:
        return None
    candidate = initial.model_copy(deep=True)
    baseline = initial.model_copy(deep=True)
    if initial.query_type.has_context_reference:
        # 已抽取且唯一的主体，纯代词替换可直接更新来源与指代标记。
        # 新主体、额外条件或更复杂的补全需要 Analyzer 提供新特征。
        subjects = [entity for entity in initial.entities if entity.source == "history"]
        if set(summary.operations) != {"context_completion"} or len(subjects) != 1:
            return None
        replaced, count = re.subn(r"[他她它]", lambda _: subjects[0].name, request.query)
        def expression(text: str) -> str:
            return re.sub(r"\s+|[？?。.]", "", unicodedata.normalize("NFKC", text))
        if count != 1 or expression(replaced) != expression(candidate_request.query):
            return None
        candidate.query_type.has_context_reference = False
        baseline.query_type.has_context_reference = False
        for feature in [*candidate.entities, *candidate.keywords, *candidate.metadata]:
            if feature.source == "history":
                feature.source, feature.history_index = "query", None
                if feature.evidence_quote not in candidate_request.query:
                    feature.evidence_quote = getattr(feature, "name", None) or getattr(feature, "text", None) or feature.value
    elif set(summary.operations) != {"retrieval_normalization"}:
        return None
    candidate.original_query = candidate_request.query
    try:
        # 抽取依据也必须在新问题中成立，不能仅修改 original_query 标签。
        candidate.validate_grounding(candidate_request)
    except ValueError:
        return None
    candidate.retrieval_strategy, _ = select_strategy(candidate, candidate_request)
    if routing_signature(baseline, request) != routing_signature(candidate, candidate_request):
        return None
    return candidate


async def prepare_hyde(
    prepared: PreparedQuery, plan: RetrievalPlan, *,
    rewrite_factory: Callable[[], Any] = QueryRewriteService,
) -> PreparedQuery:
    """仅为已确定的 Hybrid 检索计划构造向量辅助文本。"""
    if (
        plan.decision.action != "retrieve" or plan.decision.path != "hybrid"
        or prepared.analysis.status != "ok" or not plan.use_hyde
    ):
        return prepared
    if plan.query != prepared.request.query:
        raise ValueError("HyDE 计划不属于经过校验的问题")
    started = time.perf_counter()
    rewriter = rewrite_factory()
    try:
        result = await rewriter.rewrite(prepared.request, prepared.analysis)
    finally:
        await rewriter.aclose()
    return replace(
        prepared, rewrite=result.summary,
        hypothetical_document=result.hypothetical_document if result.summary.status == "rewritten" else None,
        rewrite_ms=prepared.rewrite_ms + round((time.perf_counter() - started) * 1000),
    )


async def prepare_query(
    request: QueryAnalysisRequest, *, analyzer_factory: Callable[[], Any] = QueryAnalyzerService,
    rewrite_factory: Callable[[], Any] = QueryRewriteService,
) -> PreparedQuery:
    analysis_ms = 0

    async def analyze(current: QueryAnalysisRequest) -> QueryAnalysisResult:
        nonlocal analysis_ms
        started = time.perf_counter()
        analyzer = analyzer_factory()
        try:
            result = await analyzer.analyze(current)
            if result.original_query != current.query:
                raise ValueError("分析结果不属于当前输入")
            if result.status == "ok":
                result.validate_grounding(current)
                result = result.model_copy(deep=True)
                if result.model_suggestion is None:
                    result.model_suggestion = RetrievalProposal.model_validate(
                        result.retrieval_strategy.model_dump(include=set(RetrievalProposal.model_fields)),
                    )
            return result
        finally:
            await analyzer.aclose()
            analysis_ms += round((time.perf_counter() - started) * 1000)

    initial = await analyze(request)
    initial = initial.model_copy(deep=True)
    if initial.status == "ok":
        initial.retrieval_strategy, warnings = select_strategy(initial, request)
        initial.warnings = list(dict.fromkeys([*initial.warnings, *warnings]))
    method = initial.retrieval_strategy.rewrite_method
    summary = RewriteSummary("none", "skipped", request.query)
    if initial.status != "ok" or method != "query_rewrite":
        return PreparedQuery(request, initial, summary, None, analysis_ms, 0)
    rewrite_started = time.perf_counter()
    rewriter = rewrite_factory()
    try:
        result = await rewriter.rewrite(request, initial)
    finally:
        await rewriter.aclose()
    rewrite_ms = round((time.perf_counter() - rewrite_started) * 1000)
    summary = result.summary
    final_request, final = request, initial
    clarification = result.clarification_question
    if method == "query_rewrite" and summary.status == "rewritten":
        # 独立完整问题不再携带旧历史；只在路由特征不能可靠复用时重新分析。
        candidate_request = QueryAnalysisRequest(
            query=summary.retrieval_query, reference_date=date.fromisoformat(initial.reference_date),
        )
        candidate_analysis = reuse_grounded_analysis(initial, request, candidate_request, summary)
        reanalyzed = candidate_analysis is None
        if reanalyzed:
            candidate_analysis = await analyze(candidate_request)
        if candidate_analysis.status == "ok" and not candidate_analysis.query_type.has_context_reference:
            changed = routing_signature(initial, request) != routing_signature(candidate_analysis, candidate_request)
            final_request, final = candidate_request, candidate_analysis
            summary = replace(summary, reanalyzed=reanalyzed, route_changed=changed)
        else:
            # 分析失败不能把未确认的候选传入 Router。
            summary = replace(
                summary, status="degraded", retrieval_query=request.query, operations=(),
                reanalyzed=True, degradation_reason="reanalysis_failed",
                warnings=(*summary.warnings, "改写后分析未确认完整问题，已丢弃候选。"),
            )
            if initial.query_type.has_context_reference:
                clarification = "请把对象名称和本次条件写成完整问题。"
    elif initial.query_type.has_context_reference and summary.status != "rewritten":
        clarification = clarification or "请把对象名称和本次条件写成完整问题。"
    if clarification:
        final = initial.model_copy(deep=True)
        final.query_type.needs_clarification = True
        final.query_type.clarification_question = clarification
        final.retrieval_strategy, _ = select_strategy(final, request)
    final.warnings = list(dict.fromkeys([*final.warnings, *summary.warnings]))
    return PreparedQuery(final_request, final, summary, None, analysis_ms, rewrite_ms)
