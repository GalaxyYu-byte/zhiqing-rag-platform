"""多查询扩展协议、触发边界及等价表达去重。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from .models import QueryAnalysisResult, _StrictModel


ExpansionTrigger = Literal["analyzer", "empty_retrieval"]
QueryText = Annotated[str, Field(min_length=1, max_length=2000)]


class MultiQueryCandidate(_StrictModel):
    queries: list[QueryText] = Field(max_length=2)
    reason: str = Field(min_length=1, max_length=300)


class QueryEquivalenceAudit(_StrictModel):
    query: QueryText
    equivalent: bool
    intent_preserved: bool
    constraints_preserved: bool
    no_unsupported_additions: bool
    route_preserved: bool
    reason: str = Field(min_length=1, max_length=300)


class MultiQueryAudit(_StrictModel):
    audits: list[QueryEquivalenceAudit] = Field(min_length=1, max_length=2)


@dataclass(frozen=True, slots=True)
class MultiQuerySummary:
    status: Literal["disabled", "skipped", "expanded", "unchanged", "degraded"]
    queries: tuple[str, ...]
    trigger: ExpansionTrigger | None = None
    applied_queries: tuple[str, ...] = ()
    failed_queries: tuple[str, ...] = ()
    attempts: int = 0
    reason: str = "no_trigger"
    degradation_reason: str | None = None
    warnings: tuple[str, ...] = ()


def query_key(query: str) -> str:
    return re.sub(r"[\s？?。.!！、，,]", "", unicodedata.normalize("NFKC", query)).casefold()


def expansion_block_reason(analysis: QueryAnalysisResult, *, hyde_selected: bool = False) -> str | None:
    if analysis.status != "ok":
        return "analysis_degraded"
    if hyde_selected or analysis.retrieval_strategy.use_hyde:
        return "hyde_not_combined"
    kind = analysis.query_type
    if kind.needs_clarification or kind.has_context_reference:
        return "unresolved_query"
    if (
        analysis.retrieval_strategy.path not in {"hybrid", "hybrid_graph"}
        or kind.requires_exhaustive or kind.primary in {"aggregate", "ambiguous"}
        or analysis.intent.primary in {"chitchat", "statistics", "other"}
        or "statistics" in analysis.intent.secondary
    ):
        return "unsupported_query"
    if (
        kind.primary == "exact" or any(k.kind == "exact" for k in analysis.keywords)
        or any(e.entity_type == "IDENTIFIER" for e in analysis.entities)
        or any(m.field == "document_code" for m in analysis.metadata)
        or re.search(r"\b[A-Za-z]+[-_][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*\b", analysis.original_query)
    ):
        return "exact_query"
    return None
