"""DeepSeek 输出协议；所有抽取候选必须有输入原文依据。"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


IntentName = Literal[
    "fact", "explanation", "procedure", "comparison", "summary",
    "relationship", "statistics", "recommendation", "chitchat", "other",
]
QueryTypeName = Literal[
    "exact", "semantic", "relational", "multi_hop", "aggregate",
    "compound", "conversational", "ambiguous",
]
RetrievalPath = Literal["hybrid", "hybrid_graph", "structured", "clarify", "none"]
RewriteMethod = Literal["none", "query_rewrite", "hyde"]
RewriteOperation = Literal[
    "context_completion", "retrieval_normalization", "keyword_optimization",
    "constraint_explicitization",
]
ShortText = Annotated[str, Field(min_length=1, max_length=200)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class QueryAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    # 由调用方提供经过会话所有权校验的近期消息；独立接口不读取数据库。
    history: list[HistoryMessage] = Field(default_factory=list, max_length=6)
    reference_date: date | None = None


class IntentClassification(_StrictModel):
    primary: IntentName
    secondary: list[IntentName] = Field(max_length=3)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def unique_intents(self) -> IntentClassification:
        if len(set(self.secondary)) != len(self.secondary) or self.primary in self.secondary:
            raise ValueError("secondary 意图必须去重，且不能包含 primary")
        return self


class QueryType(_StrictModel):
    primary: QueryTypeName
    confidence: float = Field(ge=0, le=1)
    has_context_reference: bool
    relation_required: bool
    potential_multi_hop: bool
    requires_exhaustive: bool
    needs_clarification: bool
    clarification_question: str | None = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def consistent_flags(self) -> QueryType:
        if self.needs_clarification != (self.clarification_question is not None):
            raise ValueError("澄清标记与澄清问题必须一致")
        if self.primary == "ambiguous" and not self.needs_clarification:
            raise ValueError("ambiguous 查询必须请求澄清")
        if self.primary in {"relational", "multi_hop"} and not self.relation_required:
            raise ValueError("关系型查询必须标记 relation_required")
        if self.primary == "multi_hop" and not self.potential_multi_hop:
            raise ValueError("multi_hop 查询必须标记 potential_multi_hop")
        if self.potential_multi_hop and not self.relation_required:
            raise ValueError("潜在多跳查询必须具有关系需求")
        return self


class Evidence(_StrictModel):
    source: Literal["query", "history"]
    history_index: int | None = Field(ge=0, le=5)
    evidence_quote: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def consistent_source(self) -> Evidence:
        if (self.source == "history") != (self.history_index is not None):
            raise ValueError("history 来源必须指定索引；query 来源索引必须为 null")
        return self


class QueryEntity(Evidence):
    # PROJECT / CONTRACT 是查询侧候选类型，不能直接映射为图数据库节点标签。
    entity_type: Literal[
        "PERSON", "ORGANIZATION", "DEPARTMENT", "PROJECT", "CONTRACT",
        "SYSTEM", "SERVICE", "PRODUCT", "TECHNOLOGY", "PROCESS",
        "POLICY", "EVENT", "LOCATION", "CONCEPT", "IDENTIFIER",
    ]
    name: ShortText
    # 老结果未提供角色时由 Router 使用保守的类型规则选择主体。
    graph_role: Literal["subject", "qualifier", "auxiliary"] | None = None
    qualifiers: list[ShortText] = Field(default_factory=list, max_length=5)


class QueryKeyword(Evidence):
    text: ShortText
    kind: Literal["entity", "domain", "exact", "relation", "action", "qualifier"]


class MetadataConstraint(Evidence):
    field: Literal[
        "time", "event_time", "uploaded_at", "effective_time", "publication_time",
        "location", "document_name", "document_code", "file_type",
        "department", "version", "business_status", "amount",
    ]
    operator: Literal[
        "eq", "neq", "contains", "lt", "lte", "gt", "gte", "range", "latest", "relative",
    ]
    # 原样保留“去年”“2025 年之前”“未验收”等，不在这里臆造数据库值或日期。
    value: str = Field(min_length=1, max_length=300)


class RetrievalProposal(_StrictModel):
    path: RetrievalPath
    rewrite_method: RewriteMethod
    rewrite_operations: list[RewriteOperation] = Field(max_length=4)
    use_query_rewrite: bool
    use_multi_query: bool
    use_hyde: bool
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def consistent_rewrite(self) -> RetrievalProposal:
        if self.use_query_rewrite != (self.rewrite_method == "query_rewrite"):
            raise ValueError("查询改写标记与方式不一致")
        if self.use_hyde != (self.rewrite_method == "hyde"):
            raise ValueError("HyDE 标记与方式不一致")
        if bool(self.rewrite_operations) != self.use_query_rewrite:
            raise ValueError("普通改写必须指定操作，其他方式操作必须为空")
        if len(set(self.rewrite_operations)) != len(self.rewrite_operations):
            raise ValueError("改写操作不能重复")
        return self


class QueryAnalysis(_StrictModel):
    """模型必须完整提供六项内容，空抽取以 [] 表示，禁止省略字段。"""

    intent: IntentClassification
    query_type: QueryType
    entities: list[QueryEntity] = Field(max_length=20)
    keywords: list[QueryKeyword] = Field(max_length=20)
    metadata: list[MetadataConstraint] = Field(max_length=15)
    retrieval_strategy: RetrievalProposal

    def validate_grounding(self, request: QueryAnalysisRequest) -> None:
        for candidate in [*self.entities, *self.keywords, *self.metadata]:
            if candidate.source == "query":
                source = request.query
            else:
                index = candidate.history_index
                if index is None or index >= len(request.history):
                    raise ValueError("抽取引用了不存在的历史消息")
                message = request.history[index]
                if message.role != "user":
                    raise ValueError("助手历史不能作为实体或元数据事实依据")
                if not self.query_type.has_context_reference:
                    raise ValueError("非指代问题不能继承历史抽取条件")
                source = message.content
            if candidate.evidence_quote not in source:
                raise ValueError("抽取证据不是输入原文的连续片段")
            value = (
                candidate.name if isinstance(candidate, QueryEntity)
                else candidate.text if isinstance(candidate, QueryKeyword)
                else candidate.value
            )
            if value not in candidate.evidence_quote:
                raise ValueError("抽取值必须逐字出现在证据中，禁止扩写或猜测")
            if isinstance(candidate, QueryEntity) and any(
                qualifier not in candidate.evidence_quote for qualifier in candidate.qualifiers
            ):
                raise ValueError("实体限定信息必须逐字来自该实体的原文证据")


class RetrievalStrategy(RetrievalProposal):
    """由服务端规则约束后的策略建议，尚未执行检索或权限过滤。"""

    dense_weight: float = Field(ge=0, le=1)
    bm25_weight: float = Field(ge=0, le=1)
    graph_weight: float = Field(ge=0, le=1)
    graph_requires_resolution: bool
    requires_coverage_check: bool
    metadata_mode: Literal["soft"] = "soft"


class QueryAnalysisResult(QueryAnalysis):
    # Analyzer 返回原始建议；兼容预处理阶段及旧调用方的规则策略。
    retrieval_strategy: RetrievalProposal | RetrievalStrategy
    model_suggestion: RetrievalProposal | None = None
    original_query: str
    reference_date: str
    model: str
    analyzer_version: Literal["query-analyzer-v2"] = "query-analyzer-v2"
    status: Literal["ok", "degraded"]
    degradation_reason: Literal["missing_api_key", "timeout", "provider_error", "invalid_output"] | None
    attempts: int = Field(ge=0, le=3)
    latency_ms: int = Field(ge=0)
    warnings: list[str]
