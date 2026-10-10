"""显式并列问题的保守拆分与候选覆盖；不推断隐含对象或修改权限范围。"""

import re
from itertools import zip_longest

from .fusion import RRF_K, FusedCandidate

_SEPARATE = re.compile(r"([^，。？?！!；;\n]+?)(?:分别|各自)(.+)")
_CONNECTOR = re.compile(r"以及|和|与|及|、")
_CLAUSE = re.compile(r"在|对于|如果|当|需要|应该|如何|为什么|哪些|什么|怎样|是否")


def rerank_queries(query: str) -> tuple[str, ...]:
    """只拆分“A 和 B 分别 …”形式，保留原句并限制到三个明确对象。"""
    matched = _SEPARATE.fullmatch(query)
    if matched is None:
        return (query,)
    subjects = tuple(subject.strip() for subject in _CONNECTOR.split(matched[1]))
    # 带有共同条件的长句不做机械拆分，以免丢失条件或改变问题含义。
    if not 2 <= len(subjects) <= 3 or any(
        not subject or len(subject) > 40 or _CLAUSE.search(subject) for subject in subjects
    ):
        return (query,)
    facets = tuple(subject + matched[2] for subject in subjects)
    return tuple(dict.fromkeys((query, *facets)))


def fuse_facet_rankings(
    fused: tuple[FusedCandidate, ...],
    facet_rankings: tuple[tuple[FusedCandidate, ...], ...],
) -> tuple[tuple[FusedCandidate, ...], ...]:
    """等权融合原召回秩与每个分面重排秩，仅决定证据选择顺序。"""
    fusion_ranks = {item.chunk.chunk_id: rank for rank, item in enumerate(fused, start=1)}
    combined = []
    for items in facet_rankings:
        facet_ranks = {item.chunk.chunk_id: rank for rank, item in enumerate(items, start=1)}

        def selection_key(item):
            chunk_id = item.chunk.chunk_id
            score = 1 / (RRF_K + fusion_ranks[chunk_id]) + 1 / (RRF_K + facet_ranks[chunk_id])
            return (-score, fusion_ranks[chunk_id], chunk_id)

        combined.append(tuple(sorted(items, key=selection_key)))
    return tuple(combined)


def cover_facets(ranked, facet_rankings, *, limit: int, anchors=()):
    """先保留原句与各分面语义第一名，剩余预算由分面的秩融合序交替填充。"""
    if not facet_rankings:
        return ranked[:limit]
    interleaved = (
        item for row in zip_longest(*facet_rankings) for item in row if item is not None
    )
    candidates = (*ranked[:1], *anchors, *interleaved, *ranked)
    original = {candidate.chunk.chunk_id: candidate for candidate in ranked}
    chosen, seen = [], set()
    for candidate in candidates:
        if candidate.chunk.chunk_id in seen:
            continue
        seen.add(candidate.chunk.chunk_id)
        # 对外分数仍来自原句重排；分面排序只用于证据覆盖，不混用不同查询的分数。
        chosen.append(original[candidate.chunk.chunk_id])
        if len(chosen) == limit:
            break
    return tuple(chosen)
