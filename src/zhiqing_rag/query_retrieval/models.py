"""检索内部候选及 HTTP 证据契约；运行记录不修改 QueryPlan。"""

from dataclasses import dataclass
from typing import Literal

from pydantic import ConfigDict, Field

from zhiqing_rag.query_analysis.models import QueryAnalysisResult, StrictModel
from zhiqing_rag.query_routing.models import PlanQueryBody, PublicId, QueryPlan, ScopeSummary
from zhiqing_rag.query_structured.models import StructuredResult


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


class GraphRelationEvidence(StrictModel):
    """一跳关系的可审计出处；答案仍须引用当前授权的原文。"""

    model_config = ConfigDict(str_strip_whitespace=False)

    assertion_id: str = Field(min_length=1, max_length=150)
    seed_entity_uid: str = Field(min_length=1, max_length=150)
    subject_entity_uid: str = Field(min_length=1, max_length=150)
    object_entity_uid: str = Field(min_length=1, max_length=150)
    predicate: str = Field(min_length=1, max_length=100)
    condition_text: str = Field(max_length=2000)
    source_quote: str = Field(min_length=1)
    quote_start: int = Field(ge=0)
    quote_end: int = Field(ge=1)


class GraphSourceEvidence(StrictModel):
    """关系或实体映射来源的完整投影和原文作用域。"""

    model_config = ConfigDict(str_strip_whitespace=False)
    tenant_id: PublicId
    knowledge_base_id: PublicId
    document_id: PublicId
    revision_id: PublicId
    generation_id: PublicId
    chunk_id: PublicId
    projection_id: PublicId
    namespace: str = Field(min_length=1, max_length=100)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_quote: str = Field(min_length=1)
    quote_start: int = Field(ge=0)
    quote_end: int = Field(ge=1)


class GraphHopEvidence(StrictModel):
    relation: GraphRelationEvidence
    source: GraphSourceEvidence
    traversal: Literal["outgoing", "incoming"]
    subject_name: str = ""
    object_name: str = ""


class GraphBridgeEvidence(StrictModel):
    identity_id: str = ""
    identity_version: int = Field(default=1, ge=1)
    source_entity_uid: str
    target_entity_uid: str
    entity_type: Literal["Material", "Process", "Actor"]
    canonical_name: str
    basis: Literal["explicit_document", "shared_context", "explicit_alias"]
    sources: tuple[GraphSourceEvidence, ...] = Field(min_length=2, max_length=4)


class GraphPathEvidence(StrictModel):
    path_id: str
    knowledge_base_id: PublicId
    hops: tuple[GraphHopEvidence, ...] = Field(min_length=2, max_length=2)
    bridges: tuple[GraphBridgeEvidence, ...] = Field(default=(), max_length=1)
    required_chunk_ids: tuple[PublicId, ...] = Field(min_length=1, max_length=6)


@dataclass(frozen=True, slots=True)
class RecallHit:
    chunk: ChunkRecord
    score: float
    graph_relations: tuple[GraphRelationEvidence, ...] = ()
    graph_paths: tuple[GraphPathEvidence, ...] = ()


class EvidenceScores(StrictModel):
    # 这些分数仅解释排序，不代表回答置信度或概率。
    dense_rank: int | None = Field(default=None, ge=1)
    dense_score: float | None = Field(default=None, allow_inf_nan=False)
    bm25_rank: int | None = Field(default=None, ge=1)
    bm25_score: float | None = Field(default=None, allow_inf_nan=False)
    graph_rank: int | None = Field(default=None, ge=1)
    graph_score: float | None = Field(default=None, allow_inf_nan=False)
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
    graph_relations: tuple[GraphRelationEvidence, ...] = ()


class RetrievalTrace(StrictModel):
    rerank_query_count: int = Field(default=0, ge=0, le=4)
    dense_status: Literal["succeeded", "failed", "skipped"] = "skipped"
    bm25_status: Literal["succeeded", "failed", "skipped"] = "skipped"
    graph_status: Literal["succeeded", "failed", "skipped", "empty"] = "skipped"
    rerank_status: Literal["succeeded", "failed", "disabled", "skipped"] = "skipped"
    dense_count: int = Field(default=0, ge=0)
    bm25_count: int = Field(default=0, ge=0)
    graph_count: int = Field(default=0, ge=0)
    graph_seed_count: int = Field(default=0, ge=0)
    graph_assertion_count: int = Field(default=0, ge=0)
    graph_path_candidate_count: int = Field(default=0, ge=0)
    graph_path_count: int = Field(default=0, ge=0)
    graph_hops: int = Field(default=0, ge=0, le=2)
    fused_count: int = Field(default=0, ge=0)
    authorized_count: int = Field(default=0, ge=0)
    returned_count: int = Field(default=0, ge=0)
    embedding_ms: int = Field(default=0, ge=0)
    dense_ms: int = Field(default=0, ge=0)
    bm25_ms: int = Field(default=0, ge=0)
    graph_ms: int = Field(default=0, ge=0)
    fusion_ms: int = Field(default=0, ge=0)
    authorization_ms: int = Field(default=0, ge=0)
    rerank_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(default=0, ge=0)


class RetrievalResult(StrictModel):
    status: Literal["ok", "degraded", "empty", "skipped", "unsupported"]
    evidence: tuple[Evidence, ...] = ()
    graph_paths: tuple["GraphPathEvidence", ...] = ()
    degradation_codes: tuple[str, ...] = ()
    message: str | None = None
    trace: RetrievalTrace = Field(default_factory=RetrievalTrace)


class QuerySearchResponse(StrictModel):
    analysis: QueryAnalysisResult
    plan: QueryPlan
    scope: ScopeSummary
    retrieval: RetrievalResult
    structured: StructuredResult | None = None
    request: PlanQueryBody | None = None
