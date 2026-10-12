"""模型草稿与经服务端核验的回答协议。"""

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from zhiqing_rag.query_analysis.models import StrictModel
from zhiqing_rag.query_retrieval.models import QuerySearchResponse

AnswerStage = Literal["analysis", "retrieval", "generation", "review"]
AnswerProgress = Callable[[AnswerStage], Awaitable[None]]


class AnswerCitation(StrictModel):
    model_config = ConfigDict(str_strip_whitespace=False)
    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    quote: str = Field(min_length=2, max_length=2000)


class AnswerStatement(StrictModel):
    text: str = Field(min_length=1, max_length=1500)
    citations: list[AnswerCitation] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def unique_citations(self):
        keys = [(item.citation_id, item.quote) for item in self.citations]
        if len(keys) != len(set(keys)):
            raise ValueError("同一陈述不能重复同一引用")
        return self


class DraftAnswer(StrictModel):
    status: Literal["answered", "insufficient"]
    statements: list[AnswerStatement] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "answered") != bool(self.statements):
            raise ValueError("回答必须有带引用的陈述，证据不足时不能保留猜测")
        return self


class RepairCitation(StrictModel):
    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    quote_candidate_index: int = Field(strict=True, ge=-1, le=9)


class RepairStatement(StrictModel):
    text: str = Field(min_length=1, max_length=1500)
    citations: list[RepairCitation] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def unique_selections(self):
        keys = [(item.citation_id, item.quote_candidate_index) for item in self.citations]
        if len(keys) != len(set(keys)):
            raise ValueError("同一陈述不能重复选择同一引用")
        return self


class SelectedRepairAnswer(StrictModel):
    """修复模型只选择服务端编号的当前连续原文，不生成引用或任意偏移。"""

    status: Literal["answered", "insufficient"]
    statements: list[RepairStatement] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent_status(self):
        if (self.status == "answered") != bool(self.statements):
            raise ValueError("回答必须有带引用的陈述，证据不足时不能保留猜测")
        return self


class StatementReview(StrictModel):
    index: int = Field(ge=0, le=7)
    supported: bool
    reason: str = Field(min_length=1, max_length=600)


class AnswerReview(StrictModel):
    sufficient: bool
    missing_aspects: list[str] = Field(max_length=8)
    statements: list[StatementReview] = Field(min_length=1, max_length=8)


class AnswerTrace(StrictModel):
    repair_count: int = Field(default=0, ge=0, le=1)
    generation_ms: int = Field(default=0, ge=0)
    review_ms: int = Field(default=0, ge=0)
    authorization_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(default=0, ge=0)
    context_tokens: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class AnswerResult(StrictModel):
    status: Literal[
        "answered",
        "structured",
        "insufficient",
        "rejected",
        "unavailable",
        "skipped",
        "unsupported",
    ]
    statements: list[AnswerStatement] = Field(default_factory=list, max_length=8)
    message: str | None = None
    codes: tuple[str, ...] = ()
    model: str | None = None
    review_status: Literal["passed", "failed", "skipped"] = "skipped"
    trace: AnswerTrace = Field(default_factory=AnswerTrace)

    @model_validator(mode="after")
    def publish_only_reviewed_answer(self):
        if self.status == "answered":
            if not self.statements or self.review_status != "passed":
                raise ValueError("只能发布经核验的回答")
        elif self.statements:
            raise ValueError("未通过发布检查时不能携带草稿")
        return self


class QueryAnswerResponse(QuerySearchResponse):
    answer: AnswerResult

    @model_validator(mode="after")
    def validate_structured_answer(self):
        if self.answer.status == "structured":
            from zhiqing_rag.query_execution import structured_answer_text

            if (
                self.structured is None
                or self.plan.route != "structured"
                or self.plan.planning_status != "planned"
                or self.answer.review_status != "passed"
                or self.answer.message != structured_answer_text(self.structured)
            ):
                raise ValueError("目录回答必须对应服务端执行的精确统计结果")
        return self
