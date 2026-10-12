"""按排名做 RRF，避免混用向量相似度与 BM25 的不同分数量纲。"""

import math
from dataclasses import dataclass

from .models import ChunkRecord, EvidenceScores, GraphPathEvidence, GraphRelationEvidence, RecallHit
from .paths import select_path_bundles

RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk: ChunkRecord
    scores: EvidenceScores
    graph_relations: tuple[GraphRelationEvidence, ...] = ()
    graph_paths: tuple[GraphPathEvidence, ...] = ()


def fuse_candidates(
    dense: tuple[RecallHit, ...],
    bm25: tuple[RecallHit, ...],
    *,
    limit: int,
    graph: tuple[RecallHit, ...] = (),
) -> tuple[FusedCandidate, ...]:
    """相同 chunk 只计每路一次；同文档同原文保留一块，避免重复占预算。"""
    if not 1 <= limit <= 100:
        raise ValueError("候选预算应在 1 到 100 之间")
    chunks: dict[int, ChunkRecord] = {}
    ranks: dict[int, dict] = {}
    relations: dict[int, tuple[GraphRelationEvidence, ...]] = {}
    paths: dict[int, tuple[GraphPathEvidence, ...]] = {}
    for channel, hits in (("dense", dense), ("bm25", bm25), ("graph", graph)):
        seen = set()
        rank = 0
        for hit in hits:
            chunk_id = hit.chunk.chunk_id
            if chunk_id in seen:
                continue
            if not math.isfinite(hit.score):
                raise ValueError("召回分数必须为有限数值")
            seen.add(chunk_id)
            rank += 1
            if chunk_id in chunks and chunks[chunk_id] != hit.chunk:
                raise ValueError("同一分块在多路召回中的原文或来源不一致")
            chunks[chunk_id] = hit.chunk
            if channel == "graph":
                relations[chunk_id] = hit.graph_relations
                paths[chunk_id] = hit.graph_paths
            scores = ranks.setdefault(chunk_id, {"fusion_score": 0.0})
            scores[f"{channel}_rank"] = rank
            scores[f"{channel}_score"] = float(hit.score)
            scores["fusion_score"] += 1.0 / (RRF_K + rank)
    candidates = sorted(
        (
            FusedCandidate(
                chunks[chunk_id],
                EvidenceScores(**scores),
                relations.get(chunk_id, ()),
                paths.get(chunk_id, ()),
            )
            for chunk_id, scores in ranks.items()
        ),
        key=lambda item: (-item.scores.fusion_score, item.chunk.chunk_id),
    )
    unique = []
    seen_content = set()
    required_ids = {
        chunk_id
        for items in paths.values()
        for path in items
        for chunk_id in path.required_chunk_ids
    }
    for candidate in candidates:
        chunk = candidate.chunk
        # 不跨文档折叠，避免丢失两个文件分别提供的引用出处。
        key = (chunk.tenant_id, chunk.document_id, chunk.revision_id, chunk.content)
        if key in seen_content and str(chunk.chunk_id) not in required_ids:
            continue
        seen_content.add(key)
        unique.append(candidate)
    return select_path_bundles(tuple(unique), limit=limit)
