"""五种查询路径的判别联合；权限范围与运行时执行记录独立维护。"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from zhiqing_rag.query_analysis.models import (
    AnalyzeQueryBody,
    QueryAnalysisResult,
    QueryConstraint,
    QueryText,
    ShortText,
    StrictModel,
)

RouteName = Literal["hybrid", "hybrid_graph", "structured", "clarify", "none"]
PublicId = Annotated[str, Field(pattern=r"^[1-9][0-9]{0,18}$")]


class PlanQueryBody(AnalyzeQueryBody):
    knowledge_base_ids: list[PublicId] | None = Field(default=None, min_length=1, max_length=50)
    document_ids: list[PublicId] | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_ids(self):
        for values in (self.knowledge_base_ids, self.document_ids):
            if values is not None and (
                len(values) != len(set(values)) or any(int(value) > 2**63 - 1 for value in values)
            ):
                raise ValueError("资源 ID 必须是无重复的正 bigint 十进制字符串")
        return self


class PlanBase(StrictModel):
    version: Literal[1] = 1
    original_query: QueryText
    query: QueryText
    requested_route: RouteName
    reason_code: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9_]*$")]
    planning_status: Literal["planned", "unsupported", "empty"] = "planned"
    constraints: tuple[QueryConstraint, ...] = Field(default=(), max_length=20)
    warnings: tuple[str, ...] = ()
    message: str | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.planning_status != "planned" and not self.message:
            raise ValueError("不支持或范围为空时必须给出说明")
        return self


class RetrievalParams(StrictModel):
    candidate_k: int = Field(default=30, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=100)
    rerank: bool = True

    @model_validator(mode="after")
    def validate_budget(self):
        if self.top_k > self.candidate_k:
            raise ValueError("最终返回数量不能超过候选数量")
        return self


class HybridPlan(PlanBase):
    route: Literal["hybrid"] = "hybrid"
    params: RetrievalParams = Field(default_factory=RetrievalParams)


class GraphProjectionRef(StrictModel):
    tenant_id: PublicId
    document_id: PublicId
    generation_id: PublicId
    projection_id: PublicId
    namespace: Annotated[str, Field(min_length=1, max_length=100)]


class GraphParams(RetrievalParams):
    seed_entity_uids: tuple[ShortText, ...] = Field(min_length=1, max_length=20)
    projections: tuple[GraphProjectionRef, ...] = Field(min_length=1)
    max_hops: int = Field(default=1, ge=1, le=2)


class GraphPlan(PlanBase):
    route: Literal["hybrid_graph"] = "hybrid_graph"
    params: GraphParams


class StructuredParams(StrictModel):
    operation: Literal["count_documents", "list_documents"] | None
    scope: Literal["authorized_published_documents"] = "authorized_published_documents"
    offset: int = Field(default=0, ge=0, le=100000)
    limit: int = Field(default=20, ge=1, le=100)


class StructuredPlan(PlanBase):
    route: Literal["structured"] = "structured"
    params: StructuredParams

    @model_validator(mode="after")
    def validate_operation(self):
        if self.planning_status != "unsupported" and self.params.operation is None:
            raise ValueError("可规划的目录查询必须指定白名单操作")
        return self


class ClarifyParams(StrictModel):
    question: Annotated[str, Field(min_length=1, max_length=300)]


class ClarifyPlan(PlanBase):
    route: Literal["clarify"] = "clarify"
    planning_status: Literal["planned"] = "planned"
    params: ClarifyParams


class ChatParams(StrictModel):
    pass


class ChatPlan(PlanBase):
    route: Literal["none"] = "none"
    planning_status: Literal["planned"] = "planned"
    params: ChatParams = Field(default_factory=ChatParams)


QueryPlan = Annotated[
    HybridPlan | GraphPlan | StructuredPlan | ClarifyPlan | ChatPlan,
    Field(discriminator="route"),
]


class ScopeSummary(StrictModel):
    knowledge_base_ids: tuple[PublicId, ...]
    authorized_document_count: int = Field(ge=0)
    published_document_count: int = Field(ge=0)


class QueryPlanningResponse(StrictModel):
    analysis: QueryAnalysisResult
    plan: QueryPlan
    scope: ScopeSummary
