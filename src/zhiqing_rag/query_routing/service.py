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


def _catalog_operation(query, constraints, *, metadata_available):
    """只移除已验证的元数据片段；剩余要求仍须完整匹配固定目录操作。"""
    if metadata_available:
        for item in constraints:
            if item.field == "other":
                return None
            value = re.escape(item.value)
            # 条件值来自当前问题逐字依据；不能删除整句 evidence.quote。
            if item.operator != "neq" and re.search(r"(?:不属于|不等于|不是|非)" + value, query):
                return None
            comparisons = {
                "eq": r"(?:属于|来自|等于|为|是|在|于)?",
                "neq": r"(?:不属于|不等于|不是|非)",
                "contains": r"(?:包含|含有)",
                "gt": r"(?:晚于|大于|从|自)?", "gte": r"(?:不早于|自|从)?",
                "lt": r"(?:早于|小于)?", "lte": r"(?:不晚于)?", "range": r"(?:在|于)?",
            }
            labels = {
                "department": r"(?:归属部门|所属部门|部门)?",
                "business_status": r"(?:业务状态|状态)?",
                "document_name": r"(?:文档名称|文件名称|文档名|文件名)?",
                "document_code": r"(?:文档编号|业务编号|文号)?",
                "file_type": r"(?:文件类型|文档类型|文件格式|格式)?",
                "uploaded_at": r"(?:上传(?:时间|日期)?)?",
                "event_time": r"(?:(?:生效|事件|发生)(?:时间|日期)?)?",
            }
            before = labels[item.field] + comparisons.get(item.operator, r"(?!)") + r"\s*"
            after = r"\s*(?:格式)?(?:的)?" if item.field == "file_type" else r"\s*(?:的)?"
            if item.field in ("uploaded_at", "event_time"):
                suffixes = {
                    "eq": r"(?:期间|内)?", "range": r"(?:期间|内)?",
                    "gt": r"(?:以后|之后)?", "gte": r"(?:及以后|及之后|起|以来)?",
                    "lt": r"(?:以前|之前)?", "lte": r"(?:及以前|及之前)?", "neq": "",
                }
                action = "上传" if item.field == "uploaded_at" else "生效"
                after = r"\s*" + suffixes.get(item.operator, "") + f"(?:{action})?(?:的)?"
            query = re.sub(before + value + after, "", query, count=1)
        query = re.sub(r"\s+", "", query)
        query = re.sub(r"^(?:满足条件的|符合条件的)", "", query)
    if _COUNT_DOCUMENTS.fullmatch(query):
        return "count_documents"
    if _LIST_DOCUMENTS.fullmatch(query) or (
        metadata_available and constraints
        and re.fullmatch(r"(?:列出|查看)(?:文档|文件|资料)", query)
    ):
        return "list_documents"
    return None


@dataclass(frozen=True, slots=True)
class GraphReadiness:
    """服务器图谱预检查器的可信结果；HTTP 和 Analyzer 都不能直接提供。"""

    query: str
    matched_entities: tuple[str, ...]
    params: GraphParams


class QueryRouterService:
    def __init__(
        self, *, retrieval_params: RetrievalParams | None = None, catalog_available=False,
        metadata_available=False,
    ):
        self.retrieval_params = retrieval_params or RetrievalParams()
        # 能力由服务端依赖注入声明，客户端和 Analyzer 不能自行开启。
        self.catalog_available = catalog_available
        self.metadata_available = metadata_available

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
        common = dict(
            original_query=request.query, query=request.query,
            resolved_filters=scope.resolved_filters,
        )

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
        operation = _catalog_operation(
            query, features.constraints, metadata_available=self.metadata_available,
        )
        if features.intent == "statistics" or operation is not None:
            if operation is None or (
                features.constraints and not self.metadata_available
            ) or (
                any(item.field != "other" for item in features.constraints)
                and not scope.resolved_filters
            ):
                reason, status, message = (
                    "STRUCTURED_QUERY_UNSUPPORTED",
                    "unsupported",
                    "当前支持文档数量和文档清单；额外业务统计、分组或未解析条件不能忽略。",
                )
            elif not self.catalog_available:
                reason, status, message = (
                    "CATALOG_EXECUTOR_NOT_READY",
                    "unsupported",
                    "已识别目录操作，目录查询执行器尚未接入。",
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
            and not scope.resolved_filters
            and not features.relation_required
            and not _KNOWLEDGE_SIGNALS.search(request.query)
        ):
            return ChatPlan(**common, requested_route="none", reason_code="PURE_CHITCHAT")

        metadata_conditions = [item for item in features.constraints if item.field != "other"]
        if metadata_conditions and (not self.metadata_available or not scope.resolved_filters):
            return hybrid(
                "METADATA_FILTERS_NOT_READY",
                unsupported_message="已保留元数据条件，过滤与日期解析尚未接入，不能忽略条件扩大检索。",
            )
        warnings = (
            ("业务条件保留在原问题和 constraints 中，仅作为语义证据要求，不转换为目录筛选。",)
            if any(item.field == "other" for item in features.constraints)
            else ()
        )
        if features.relation_required and features.entities:
            if graph is None:
                return hybrid(
                    "GRAPH_NOT_READY",
                    requested="hybrid_graph",
                    warnings=(*warnings, "当前授权范围内未确认可用图谱实体，规划基础混合检索。"),
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
