"""约束 LLM 的策略建议，防止实体数量、模型权重或过滤条件直接驱动检索。"""

from .models import QueryAnalysis, QueryAnalysisRequest, RetrievalStrategy


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
    if proposal.use_hyde:
        warnings.append("第一版未启用 HyDE，已关闭模型的 HyDE 建议。")
    if analysis.metadata:
        warnings.append("元数据保留原文，仅作为软约束；执行过滤前需解析、字段映射和权限校验。")
    graph = path == "hybrid_graph"
    exact = query_type.primary == "exact" or any(k.kind == "exact" for k in analysis.keywords)
    if graph:
        weights = (0.4, 0.3, 0.3)
    elif path == "hybrid":
        weights = (0.3, 0.7, 0.0) if exact else (0.5, 0.5, 0.0)
    else:
        weights = (0.0, 0.0, 0.0)
    retrieval = path in {"hybrid", "hybrid_graph"}
    return RetrievalStrategy(
        path=path,
        use_query_rewrite=retrieval and (
            proposal.use_query_rewrite or query_type.has_context_reference
        ),
        use_multi_query=retrieval and proposal.use_multi_query,
        use_hyde=False,
        reason=reason,
        dense_weight=weights[0], bm25_weight=weights[1], graph_weight=weights[2],
        graph_requires_resolution=graph,
        requires_coverage_check=graph or path == "structured",
    ), warnings
