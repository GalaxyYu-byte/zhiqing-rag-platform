"""确定性 Router：绑定分析输入、保留条件、检查范围和可信运行时能力。"""

import re
from dataclasses import dataclass

from zhiqing_rag.query_analysis.models import QueryAnalysisRequest, QueryAnalysisResult

from .models import (
    ChatPlan,
    ClarifyParams,
    ClarifyPlan,
    GraphParams,
    GraphPlan,
    HybridPlan,
    QueryPlan,
    RetrievalParams,
    StructuredParams,
    StructuredPlan,
)
from .scope import AccessScope

# 全句白名单：出现额外主题、排序、业务条件或复合要求时不能删掉它们再统计。
_COUNT_DOCUMENTS = re.compile(
    r"(?:(?:当前|这个|指定)?知识库(?:中|里|内|的)?)?(?:有)?(?:多少|几)(?:份|篇|个)?(?:文档|文件|资料)"
    r"|(?:(?:当前|这个|指定)?知识库(?:中|里|内|的)?)?(?:文档|文件|资料)(?:有)?(?:多少|几)(?:份|篇|个)?"
    r"|统计(?:(?:当前|这个|指定)?知识库(?:中|里|内|的)?)?(?:文档|文件|资料)(?:数量|总数)"
)
_LIST_DOCUMENTS = re.compile(
    r"(?:列出|查看)(?:(?:当前|这个|指定)?知识库(?:中|里|内|的)?)?(?:所有|全部)(?:的)?(?:文档|文件|资料)"
)
_KNOWLEDGE_SIGNALS = re.compile(
    r"知识库|文档|文件|资料|工单|权限|流程|制度|合同"
    r"|(?<![A-Za-z])(?:document|file)(?![A-Za-z])",
    re.I,
)


@dataclass(frozen=True, slots=True)
class GraphReadiness:
    """未来图谱预检查器的可信结果；HTTP 和 Analyzer 都不能直接提供。"""

    query: str
    matched_entities: tuple[str, ...]
    params: GraphParams


class QueryRouterService:
    def __init__(self, *, retrieval_params: RetrievalParams | None = None, catalog_available=False):
        self.retrieval_params = retrieval_params or RetrievalParams()
        # 表示可信目录执行器已接入；当前 HTTP 入口固定关闭。
        self.catalog_available = catalog_available

    def route(
        self,
        *,
        request: QueryAnalysisRequest,
        analysis: QueryAnalysisResult,
        scope: AccessScope,
        graph: GraphReadiness | None = None,
    ) -> QueryPlan:
        request = QueryAnalysisRequest.model_validate(request.model_dump())
        analysis = QueryAnalysisResult.model_validate(analysis.model_dump())
        if request.query != analysis.original_query:
            raise ValueError("分析结果不属于当前问题")
        common = dict(original_query=request.query, query=request.query)

        def clarify(reason, question):
            return ClarifyPlan(
                **common,
                requested_route="clarify",
                reason_code=reason,
                params=ClarifyParams(question=question),
            )

        def hybrid(reason, *, requested="hybrid", warnings=(), unsupported_message=None):
            if unsupported_message is not None:
                status, message = "unsupported", unsupported_message
            elif not scope.published_documents:
                status, message = "empty", "当前授权范围内没有已发布且活动索引就绪的文档。"
            else:
                status, message = "planned", None
            return HybridPlan(
                **common,
                requested_route=requested,
                reason_code=reason,
                planning_status=status,
                message=message,
                warnings=warnings,
                params=self.retrieval_params,
            )

        if analysis.status == "degraded":
            if analysis.fallback_hint == "clarify":
                return clarify("ANALYSIS_DEGRADED_CONTEXT", analysis.message)
            if analysis.fallback_hint == "unsupported":
                return StructuredPlan(
                    **common,
                    requested_route="structured",
                    reason_code="ANALYSIS_DEGRADED_STATISTICS",
                    planning_status="unsupported",
                    message=analysis.message,
                    params=StructuredParams(operation=None),
                )
            return hybrid(
                "ANALYSIS_DEGRADED", warnings=("保留当前原问题，忽略历史与故障模型特征。",)
            )

        features = analysis.analysis
        features.validate_grounding(request)
        common["constraints"] = tuple(features.constraints)
        if features.clarification_question is not None:
            return clarify("CLARIFICATION_REQUIRED", features.clarification_question)
        if features.needs_context:
            return clarify(
                "CONTEXT_REWRITE_REQUIRED",
                "请补充上文中的对象名称和本次条件，当前尚未接入问题改写。",
            )

        query = request.query.rstrip("？?。.!！ ")
        operation = (
            "count_documents"
            if _COUNT_DOCUMENTS.fullmatch(query)
            else "list_documents"
            if _LIST_DOCUMENTS.fullmatch(query)
            else None
        )
        if features.intent == "statistics" or operation is not None:
            if operation is None or features.constraints:
                reason, status, message = (
                    "STRUCTURED_QUERY_UNSUPPORTED",
                    "unsupported",
                    "首期仅定义无额外条件的文档数量和全量清单；业务统计、分组及条件解析尚未接入。",
                )
            elif not self.catalog_available:
                reason, status, message = (
                    "CATALOG_EXECUTOR_NOT_READY",
                    "unsupported",
                    "已识别目录操作，目录查询执行器尚未接入。",
                )
            elif not scope.published_documents:
                reason, status, message = (
                    "NO_PUBLISHED_DOCUMENTS",
                    "empty",
                    "当前授权范围内没有已发布且活动索引就绪的文档。",
                )
            else:
                reason, status, message = "CATALOG_SELECTED", "planned", None
            return StructuredPlan(
                **common,
                requested_route="structured",
                reason_code=reason,
                planning_status=status,
                message=message,
                params=StructuredParams(operation=operation),
            )

        if (
            features.intent == "chitchat"
            and not features.entities
            and not features.constraints
            and not features.relation_required
            and not _KNOWLEDGE_SIGNALS.search(request.query)
        ):
            return ChatPlan(**common, requested_route="none", reason_code="PURE_CHITCHAT")

        metadata_conditions = [item for item in features.constraints if item.field != "other"]
        if metadata_conditions:
            return hybrid(
                "METADATA_FILTERS_NOT_READY",
                unsupported_message="已保留元数据条件，过滤与日期解析尚未接入，不能忽略条件扩大检索。",
            )
        warnings = (
            ("业务条件保留在原问题和 constraints 中，仅作为语义证据要求，不转换为目录筛选。",)
            if features.constraints
            else ()
        )
        if features.relation_required and features.entities:
            if graph is None:
                return hybrid(
                    "GRAPH_NOT_READY",
                    requested="hybrid_graph",
                    warnings=(*warnings, "图谱预检查与在线召回尚未接入，规划基础混合检索。"),
                )
            self._validate_graph(graph, request, features, scope)
            return GraphPlan(
                **common,
                requested_route="hybrid_graph",
                reason_code="GRAPH_APPROVED",
                warnings=warnings,
                params=graph.params,
            )
        return hybrid("BASE_RETRIEVAL", warnings=warnings)

    @staticmethod
    def _validate_graph(graph, request, features, scope):
        expected = {entity.text for entity in features.entities}
        if graph.query != request.query or set(graph.matched_entities) != expected:
            raise ValueError("图谱预检查不属于当前问题或未确认全部实体")
        approved = {
            (document.document_id, document.generation_id) for document in scope.published_documents
        }
        for projection in graph.params.projections:
            if (
                int(projection.tenant_id) != scope.tenant_id
                or (int(projection.document_id), int(projection.generation_id)) not in approved
            ):
                raise ValueError("图谱投影超出授权范围或不是活动批次")
