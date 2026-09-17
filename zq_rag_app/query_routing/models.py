"""路由结果仅表达决策；执行器保留已校验的授权范围。"""

from dataclasses import dataclass
from typing import Literal

from ..query_analysis.models import RetrievalPath


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


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    decision: RoutingDecision
    query: str
    message: str | None = None
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    graph_weight: float = 0.0
    max_hops: int = 1
    seed_entity_uids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphCoverage:
    status: Literal["covered", "unmatched", "ambiguous"]
    seed_entity_uids: tuple[str, ...] = ()
