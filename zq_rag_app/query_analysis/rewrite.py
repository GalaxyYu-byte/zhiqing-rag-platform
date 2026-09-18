"""改写协议与本地约束。假想文档不得成为路由特征或回答证据。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from .models import (
    QueryAnalysisRequest, QueryAnalysisResult, RewriteMethod, RewriteOperation, _StrictModel,
)


class ContextSource(_StrictModel):
    history_index: int = Field(ge=0, le=5)
    evidence_quote: str = Field(min_length=1, max_length=500)


class RewriteCandidate(_StrictModel):
    status: Literal["rewritten", "unchanged", "clarify"]
    query: str | None = Field(min_length=1, max_length=2000)
    hypothetical_document: str | None = Field(min_length=1, max_length=2000)
    clarification_question: str | None = Field(min_length=1, max_length=300)
    operations: list[RewriteOperation] = Field(max_length=4)
    context_sources: list[ContextSource] = Field(max_length=6)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def consistent_status(self) -> RewriteCandidate:
        if self.status == "clarify":
            if self.query is not None or self.hypothetical_document is not None or not self.clarification_question:
                raise ValueError("澄清不能同时生成检索输入")
            if self.operations or self.context_sources:
                raise ValueError("澄清不能声明已完成改写")
        elif self.query is None or self.clarification_question is not None:
            raise ValueError("检索输入与澄清状态不一致")
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("重复改写操作")
        return self


class RewriteAudit(_StrictModel):
    equivalent: bool
    intent_preserved: bool
    constraints_preserved: bool
    no_unsupported_additions: bool
    context_resolved: bool
    reason: str = Field(min_length=1, max_length=300)


@dataclass(frozen=True, slots=True)
class RewriteSummary:
    method: RewriteMethod
    status: Literal["skipped", "rewritten", "unchanged", "clarify", "degraded"]
    retrieval_query: str
    operations: tuple[str, ...] = ()
    reanalyzed: bool = False
    route_changed: bool = False
    attempts: int = 0
    degradation_reason: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RewriteResult:
    summary: RewriteSummary
    # 内部传输字段；不放进 Chat 响应，也不放进 evidence/context。
    hypothetical_document: str | None = None
    clarification_question: str | None = None


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


_NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")
_IDENTIFIERS = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")
_QUANTITIES = re.compile(r"\d+(?:[.,]\d+)*\s*(?:万元|亿元|美元|人民币|元|年|月|日|天|小时|分钟|秒|%|％|公里|米|吨|kg|GB|MB|ms)")
_LIMITS = re.compile(
    r"不包括|不包含|不得|不超过|不少于|不低于|不高于|未验收|未完成|未生效|"
    r"排除|除外|仅限|仅|只|所有|全部|之前|之后|截至|至少|至多|大于|小于|"
    r"超过|以内|以上|以下|去年|今年|最近|最新|不|未|无"
)


def validate_candidate(
    candidate: RewriteCandidate, request: QueryAnalysisRequest, analysis: QueryAnalysisResult,
) -> None:
    """本地保留显式条件，再由独立 LLM 审核语义与上下文选择。"""
    method = analysis.retrieval_strategy.rewrite_method
    if candidate.status == "clarify":
        return
    assert candidate.query is not None
    sources: list[str] = []
    for source in candidate.context_sources:
        if not analysis.query_type.has_context_reference or source.history_index >= len(request.history):
            raise ValueError("禁止无指代查询继承历史")
        message = request.history[source.history_index]
        if message.role != "user" or source.evidence_quote not in message.content:
            raise ValueError("上下文证据必须逐字来自用户历史")
        sources.append(source.evidence_quote)
    if method == "hyde":
        if candidate.query != request.query or candidate.operations or sources:
            raise ValueError("HyDE 不能改写问题或继承历史")
        if candidate.status == "rewritten" and not candidate.hypothetical_document:
            raise ValueError("HyDE 缺少假想文档")
        if candidate.status == "unchanged" and candidate.hypothetical_document is not None:
            raise ValueError("未改写不能携带假想文档")
        if candidate.hypothetical_document and _NUMBERS.findall(candidate.hypothetical_document):
            raise ValueError("HyDE 不能编造数字或编号")
        return
    if method != "query_rewrite" or candidate.hypothetical_document is not None:
        raise ValueError("改写方式不一致")
    if not set(candidate.operations).issubset(analysis.retrieval_strategy.rewrite_operations):
        raise ValueError("执行了未选择的改写操作")
    if candidate.status == "unchanged":
        if candidate.query != request.query or candidate.operations or sources:
            raise ValueError("未改写状态不一致")
        if analysis.query_type.has_context_reference:
            raise ValueError("指代问题未完成补全")
        return
    if not candidate.operations or candidate.query == request.query:
        raise ValueError("改写必须产生变化并声明操作")
    if analysis.query_type.has_context_reference and (
        not sources or "context_completion" not in candidate.operations
    ):
        raise ValueError("上下文补全必须引用用户历史依据")
    validate_query_constraints(candidate.query, request, analysis, sources=sources)


def protected_query_phrases(request: QueryAnalysisRequest, analysis: QueryAnalysisResult) -> list[str]:
    """检索表达需要原样保留的主体和显式条件，兼用于生成提示与校验。"""
    return list(dict.fromkeys([
        *(e.name for e in analysis.entities if e.source == "query"),
        *(m.value for m in analysis.metadata if m.source == "query"),
        *(k.text for k in analysis.keywords if k.source == "query" and k.kind in {"exact", "qualifier"}),
        *_IDENTIFIERS.findall(request.query), *_NUMBERS.findall(request.query),
        *_QUANTITIES.findall(request.query),
        *_LIMITS.findall(request.query),
    ]))


def validate_query_constraints(
    query: str, request: QueryAnalysisRequest, analysis: QueryAnalysisResult,
    *, sources: list[str] | None = None,
) -> None:
    """普通改写和等价查询扩展共同保留原文条件；语义仍需独立审核。"""
    protected = protected_query_phrases(request, analysis)
    normalized = _normalized(query)
    if any(_normalized(value) not in normalized for value in protected):
        raise ValueError("改写丢失主体或显式约束")
    allowed_numbers = set(_NUMBERS.findall("\n".join([request.query, *(sources or [])])))
    candidate_numbers = set(_NUMBERS.findall(query))
    if not set(_NUMBERS.findall(request.query)).issubset(candidate_numbers):
        raise ValueError("改写丢失原始数字")
    if not candidate_numbers.issubset(allowed_numbers):
        raise ValueError("改写增加没有依据的数字")
    if not set(_IDENTIFIERS.findall(query)).issubset(_IDENTIFIERS.findall("\n".join([request.query, *(sources or [])]))):
        raise ValueError("检索表达增加没有依据的标识符")


def missing_query_phrases(query: str, request: QueryAnalysisRequest, analysis: QueryAnalysisResult) -> list[str]:
    normalized = _normalized(query)
    return [phrase for phrase in protected_query_phrases(request, analysis) if _normalized(phrase) not in normalized]
