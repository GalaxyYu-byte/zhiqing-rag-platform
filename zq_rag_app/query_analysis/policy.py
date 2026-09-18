"""约束 LLM 的策略建议，防止实体数量、模型权重或过滤条件直接驱动检索。"""

import re

from .models import QueryAnalysis, QueryAnalysisRequest, RetrievalStrategy


SEMANTIC_HYBRID_WEIGHTS = (0.5, 0.5, 0.0)
EXACT_HYBRID_WEIGHTS = (0.3, 0.7, 0.0)
GRAPH_WEIGHTS = (0.4, 0.3, 0.3)
NO_RETRIEVAL_WEIGHTS = (0.0, 0.0, 0.0)


def hybrid_weights_for(analysis: QueryAnalysis) -> tuple[float, float, float]:
    exact = analysis.query_type.primary == "exact" or any(k.kind == "exact" for k in analysis.keywords)
    return EXACT_HYBRID_WEIGHTS if exact else SEMANTIC_HYBRID_WEIGHTS


def select_strategy(
    analysis: QueryAnalysis, request: QueryAnalysisRequest,
) -> tuple[RetrievalStrategy, list[str]]:
    warnings: list[str] = []
    query_type = analysis.query_type
    proposal = analysis.retrieval_strategy
    if query_type.has_context_reference and not request.history:
        query_type.needs_clarification = True
        query_type.clarification_question = "请明确问题中指代的对象，或提供相关上文。"
        warnings.append("缺少指代所需的历史上下文。")

    if query_type.needs_clarification:
        path = "clarify"
        reason = "必要信息或指代未明确，先澄清问题。"
    elif (
        analysis.intent.primary == "statistics"
        or "statistics" in analysis.intent.secondary
        or query_type.requires_exhaustive
        or query_type.primary == "aggregate"
    ):
        path = "structured"
        reason = "需要完整证据或结构化统计，Top-K 检索不能保证结果完整。"
        warnings.append("structured 仅为能力需求标记，当前未执行统计或全量检索。")
    elif (
        analysis.intent.primary == "chitchat"
        and not analysis.entities and not analysis.metadata
        and not query_type.relation_required
        and query_type.primary in {"semantic", "conversational"}
    ):
        path = "none"
        reason = "纯闲聊，无知识检索需求。"
    elif query_type.relation_required and analysis.entities:
        path = "hybrid_graph"
        reason = "存在命名实体及关系需求，实体匹配和图谱覆盖确认后可增加图谱召回。"
    else:
        path = "hybrid"
        reason = "结合语义与词法召回；缺少图谱适用依据时保留基础检索。"

    if path != proposal.path:
        warnings.append("模型建议的检索路径已由服务端规则调整。")
    if analysis.metadata:
        warnings.append("元数据保留原文，仅作为软约束；执行过滤前需解析、字段映射和权限校验。")
    graph = path == "hybrid_graph"
    exact = query_type.primary == "exact" or any(k.kind == "exact" for k in analysis.keywords)
    if graph:
        weights = GRAPH_WEIGHTS
    elif path == "hybrid":
        weights = hybrid_weights_for(analysis)
    else:
        weights = NO_RETRIEVAL_WEIGHTS
    retrieval = path in {"hybrid", "hybrid_graph"}
    method = proposal.rewrite_method if retrieval else "none"
    operations = list(proposal.rewrite_operations) if method == "query_rewrite" else []
    if retrieval and query_type.has_context_reference:
        method = "query_rewrite"
        if "context_completion" not in operations:
            operations.insert(0, "context_completion")
    elif method == "hyde" and (
        path != "hybrid" or exact or analysis.metadata
        or query_type.relation_required or query_type.primary != "semantic"
        or analysis.intent.primary not in {"explanation", "procedure", "recommendation"}
        or any(e.entity_type not in {"CONCEPT", "TECHNOLOGY", "PROCESS"} for e in analysis.entities)
        or re.search(r"\d|不|未|无|除外|仅|只|全部|所有|之前|之后|至少|至多|去年|今年|最近|最新", request.query)
    ):
        method = "none"
        warnings.append("精确条件、实体关系或查询形态不适合 HyDE，已保留原问题。")
    if method == "query_rewrite" and not query_type.has_context_reference:
        operations = [op for op in operations if op != "context_completion"]
        if not operations:
            method = "none"
    return RetrievalStrategy(
        path=path,
        rewrite_method=method, rewrite_operations=operations,
        use_query_rewrite=method == "query_rewrite",
        use_multi_query=(retrieval and proposal.use_multi_query and method != "hyde" and not exact
                         and analysis.intent.primary != "other"
                         and not any(e.entity_type == "IDENTIFIER" for e in analysis.entities)
                         and not any(m.field == "document_code" for m in analysis.metadata)),
        use_hyde=method == "hyde",
        reason=reason,
        dense_weight=weights[0], bm25_weight=weights[1], graph_weight=weights[2],
        graph_requires_resolution=graph,
        requires_coverage_check=graph or path == "structured",
    ), warnings
