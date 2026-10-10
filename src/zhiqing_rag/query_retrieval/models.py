"""检索内部候选及 HTTP 证据契约；运行记录不修改 QueryPlan。"""

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field

from zhiqing_rag.query_analysis.models import QueryAnalysisResult, StrictModel
from zhiqing_rag.query_routing.models import PublicId, QueryPlan, ScopeSummary


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    tenant_id: int
    knowledge_base_id: int
    document_id: int
    revision_id: int
    generation_id: int
    chunk_id: int
    chunk_index: int
    document_title: str
    document_name: str
    content: str
    content_sha256: str
    token_count: int
    page_number: int | None
    section_title: str | None


@dataclass(frozen=True, slots=True)
class RecallHit:
    chunk: ChunkRecord
    score: float


class EvidenceScores(StrictModel):
    # 这些分数仅解释排序，不代表回答置信度或概率。
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = Field(default=None, allow_inf_nan=False)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = Field(default=None, allow_inf_nan=False)
    fusion_score: float = Field(ge=0, allow_inf_nan=False)
    rerank_score: float | None = Field(default=None, allow_inf_nan=False)


class Evidence(StrictModel):
    # 原文和文件名不能被严格输入模型的全局 strip 规则改写。
    model_config = ConfigDict(str_strip_whitespace=False)

    citation_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    rank: int = Field(ge=1)
    chunk_id: PublicId
    knowledge_base_id: PublicId
    document_id: PublicId
    revision_id: PublicId
    generation_id: PublicId
    document_title: str
    document_name: str
    chunk_index: int = Field(ge=0)
    content: str
    token_count: int = Field(ge=0)
    page_number: int | None
    section_title: str | None
    scores: EvidenceScores


class RetrievalTrace(StrictModel):
    rerank_query_count: int = Field(default=0, ge=0, le=4)
    dense_status: Literal["succeeded", "failed", "skipped"] = "skipped"
    bm25_status: Literal["succeeded", "failed", "skipped"] = "skipped"
    rerank_status: Literal["succeeded", "failed", "disabled", "skipped"] = "skipped"
    dense_count: int = Field(default=0, ge=0)
    bm25_count: int = Field(default=0, ge=0)
    fused_count: int = Field(default=0, ge=0)
    authorized_count: int = Field(default=0, ge=0)
    returned_count: int = Field(default=0, ge=0)
    embedding_ms: int = Field(default=0, ge=0)
    dense_ms: int = Field(default=0, ge=0)
    bm25_ms: int = Field(default=0, ge=0)
    fusion_ms: int = Field(default=0, ge=0)
    authorization_ms: int = Field(default=0, ge=0)
    rerank_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(default=0, ge=0)


class RetrievalResult(StrictModel):
    status: Literal["ok", "degraded", "empty", "skipped", "unsupported"]
    evidence: tuple[Evidence, ...] = ()
    degradation_codes: tuple[str, ...] = ()
    message: str | None = None
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)


class QuerySearchResponse(StrictModel):
    analysis: QueryAnalysisResult
    plan: QueryPlan
    scope: ScopeSummary
    retrieval: RetrievalResult
