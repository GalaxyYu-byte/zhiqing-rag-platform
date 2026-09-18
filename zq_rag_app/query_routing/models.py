"""路由结果仅表达决策；执行器保留已校验的授权范围。"""

from dataclasses import dataclass
from typing import Literal

from ..query_analysis.models import RetrievalPath
from ..query_analysis.policy import NO_RETRIEVAL_WEIGHTS, SEMANTIC_HYBRID_WEIGHTS


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    requested_path: RetrievalPath
    path: RetrievalPath
    action: Literal["retrieve", "clarify", "unsupported", "chat"]
    reason: str
    analyzer_status: Literal["ok", "degraded"]
    graph_degraded: bool = False
    warnings: tuple[str, ...] = ()
    deferred_features: tuple[str, ...] = ()
    router_version: str = "query-router-v1"
    model_suggestion: dict[str, object] | None = None
    policy_path: RetrievalPath | None = None
    rule_path: RetrievalPath | None = None
    rule_reason: str | None = None
    rule_weights: tuple[float, float, float] = NO_RETRIEVAL_WEIGHTS


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    decision: RoutingDecision
    query: str
    message: str | None = None
    dense_weight: float = SEMANTIC_HYBRID_WEIGHTS[0]
    bm25_weight: float = SEMANTIC_HYBRID_WEIGHTS[1]
    graph_weight: float = SEMANTIC_HYBRID_WEIGHTS[2]
    max_hops: int = 1
    seed_entity_uids: tuple[str, ...] = ()
    # Router 按分析特征确定的两路权重，供 Graph 执行阶段降级使用。
    hybrid_fallback_weights: tuple[float, float] = SEMANTIC_HYBRID_WEIGHTS[:2]
    kb_ids: tuple[int, ...] = ()
    doc_ids: tuple[int, ...] | None = None
    candidate_k: int = 30
    top_k: int = 5
    min_dense_score: float = 0.3
    rerank: bool = True
    use_multi_query: bool = False
    use_hyde: bool = False
    graph_timeout_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    path: RetrievalPath
    query: str
    kb_ids: tuple[int, ...]
    doc_ids: tuple[int, ...] | None
    candidate_k: int
    top_k: int
    weights: tuple[float, float, float] = NO_RETRIEVAL_WEIGHTS
    min_dense_score: float | None = None
    graph_attempted: bool = False
    graph_max_hops: int | None = None
    graph_timeout_seconds: float | None = None
    graph_seed_entity_uids: tuple[str, ...] = ()
    rerank_requested: bool = False
    reranked: bool = False
    dense_input: Literal["query", "hypothetical_document", "skipped"] = "skipped"
    applied_queries: tuple[str, ...] = ()
    degradation_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphCoverage:
    status: Literal["covered", "partial", "unmatched", "ambiguous"]
    seed_entity_uids: tuple[str, ...] = ()
    unresolved_auxiliary: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphEntityCandidate:
    name: str
    entity_type: str | None = None
    required: bool = True
    qualifiers: tuple[str, ...] = ()
