"""Analyzer 输入、逐字依据与严格输出协议。"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IntentName = Literal[
    "fact",
    "explanation",
    "procedure",
    "comparison",
    "summary",
    "relationship",
    "statistics",
    "recommendation",
    "chitchat",
    "other",
]
QueryText = Annotated[str, Field(min_length=1, max_length=2000)]
ShortText = Annotated[str, Field(min_length=1, max_length=200)]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )


class AnalyzeQueryBody(StrictModel):
    """独立 HTTP 接口仅接收当前问题；会话历史必须由服务端验证归属后加载。"""

    query: QueryText


class QueryAnalysisRequest(AnalyzeQueryBody):
    """内部调用方可提供最近六条已校验归属的用户问题，不接收助手回答。"""

    history: list[QueryText] = Field(default_factory=list, max_length=6)


class Evidence(StrictModel):
    source: Literal["query", "history"]
    history_index: int | None
    quote: Annotated[str, Field(min_length=1, max_length=2000)]

    @model_validator(mode="after")
    def validate_source(self):
        if self.source == "query" and self.history_index is not None:
            raise ValueError("当前问题的依据不能指定历史索引")
        if self.source == "history" and (
            self.history_index is None or not 0 <= self.history_index < 6
        ):
            raise ValueError("历史依据必须指定 0 到 5 的索引")
        return self


class GroundedTerm(StrictModel):
    text: ShortText
    evidence: Evidence


class QueryConstraint(StrictModel):
    # other 保留首期未支持的业务限制；Router 不能静默丢弃它再扩大查询。
    field: Literal[
        "department",
        "document_name",
        "document_code",
        "file_type",
        "business_status",
        "uploaded_at",
        "event_time",
        "other",
    ]
    operator: Literal["eq", "neq", "contains", "gt", "gte", "lt", "lte", "range", "unspecified"]
    value: ShortText
    evidence: Evidence


class GroundingError(ValueError):
    """抽取结果不能逐字定位到当前请求。"""


class QueryFeatures(StrictModel):
    intent: IntentName
    entities: list[GroundedTerm] = Field(max_length=20)
    keywords: list[GroundedTerm] = Field(max_length=20)
    constraints: list[QueryConstraint] = Field(max_length=20)
    relation_required: bool
    needs_context: bool
    clarification_question: Annotated[str, Field(min_length=1, max_length=300)] | None

    @model_validator(mode="after")
    def validate_consistency(self):
        for terms in (self.entities, self.keywords):
            names = [item.text.casefold() for item in terms]
            if len(names) != len(set(names)):
                raise ValueError("实体或关键词不能重复")
        keys = [(item.field, item.operator, item.value) for item in self.constraints]
        if len(keys) != len(set(keys)):
            raise ValueError("条件不能重复")
        if self.intent == "relationship" and not self.relation_required:
            raise ValueError("关系意图必须标记关系需求")
        if self.intent == "chitchat" and (self.constraints or self.relation_required):
            raise ValueError("带有查询条件或关系需求不能视为纯闲聊")
        return self

    def validate_grounding(self, request: QueryAnalysisRequest) -> None:
        for item in [*self.entities, *self.keywords, *self.constraints]:
            evidence = item.evidence
            if evidence.source == "query":
                source = request.query
            else:
                index = evidence.history_index
                if index is None or index >= len(request.history) or not self.needs_context:
                    raise GroundingError("历史依据越界或问题未标记上下文依赖")
                source = request.history[index]
            value = item.value if isinstance(item, QueryConstraint) else item.text
            if evidence.quote not in source or value not in evidence.quote:
                raise GroundingError("名称或条件值没有逐字原文依据")
            if isinstance(item, QueryConstraint) and evidence.source != "query":
                raise GroundingError("不能自动把历史问题的条件继承为当前查询条件")
        if self.needs_context and not request.history and self.clarification_question is None:
            raise GroundingError("缺少上下文时必须提出澄清问题")


class QueryAnalysisResult(StrictModel):
    status: Literal["ok", "degraded"]
    original_query: QueryText
    analysis: QueryFeatures | None
    fallback_hint: Literal["basic_retrieval", "clarify", "unsupported"] | None
    message: str | None
    degradation_reason: (
        Literal[
            "disabled",
            "not_configured",
            "timeout",
            "model_unavailable",
            "invalid_response",
            "invalid_grounding",
            "incomplete_response",
        ]
        | None
    )
    analyzer_version: str = "query-analyzer-v1"
    model: str
    latency_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "ok":
            if self.analysis is None or any(
                value is not None
                for value in (self.fallback_hint, self.message, self.degradation_reason)
            ):
                raise ValueError("成功结果只能携带校验后的特征")
        elif (
            self.analysis is not None
            or self.fallback_hint is None
            or self.message is None
            or self.degradation_reason is None
        ):
            raise ValueError("降级结果不能保留故障模型的特征，必须给出处理建议")
        return self
