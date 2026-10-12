"""完整业务路径按来源集合占预算，任何必需出处缺失时移除整条路径。"""

import hashlib
from dataclasses import replace


def complete_paths(candidates):
    """重新验证路径与当前候选原文，不信任只由 ID 声称的完整链。"""
    records = {str(item.chunk.chunk_id): item.chunk for item in candidates}
    paths, conflicts = {}, set()
    for item in candidates:
        for path in item.graph_paths:
            previous = paths.setdefault(path.path_id, path)
            if previous != path:
                conflicts.add(path.path_id)
    valid = []
    for path_id, path in paths.items():
        if path_id in conflicts or not _connected(path):
            continue
        sources = [hop.source for hop in path.hops]
        sources.extend(source for bridge in path.bridges for source in bridge.sources)
        required = {source.chunk_id for source in sources}
        if any(source.knowledge_base_id != path.knowledge_base_id for source in sources):
            continue
        if (len(path.required_chunk_ids) != len(required)
                or required != set(path.required_chunk_ids) or not required <= records.keys()):
            continue
        if any(not _source_matches(source, records[source.chunk_id]) for source in sources):
            continue
        if any(
            hop.relation
            not in next(
                item.graph_relations
                for item in candidates
                if str(item.chunk.chunk_id) == hop.source.chunk_id
            )
            or hop.relation.source_quote != hop.source.source_quote
            or (hop.relation.quote_start, hop.relation.quote_end)
            != (hop.source.quote_start, hop.source.quote_end)
            or (
                hop.relation.condition_text
                and hop.relation.condition_text not in hop.source.source_quote
            )
            for hop in path.hops
        ):
            continue
        valid.append(path)
    return tuple(sorted(valid, key=lambda path: path.path_id))


def _connected(path):
    endpoints = []
    for hop in path.hops:
        relation = hop.relation
        nodes = (relation.subject_entity_uid, relation.object_entity_uid)
        if hop.traversal == "incoming":
            nodes = nodes[::-1]
        if relation.seed_entity_uid != nodes[0] or nodes[0] == nodes[1]:
            return False
        endpoints.append(nodes)
    first, second = endpoints
    if first[0] in second or second[1] == first[1]:
        return False
    if path.hops[0].relation.assertion_id == path.hops[1].relation.assertion_id:
        return False
    if path.bridges:
        bridge = path.bridges[0]
        return (bridge.source_entity_uid, bridge.target_entity_uid) == (first[1], second[0])
    return (
        first[1] == second[0]
        and all(getattr(path.hops[0].source, key) == getattr(path.hops[1].source, key)
                for key in ("document_id", "namespace", "generation_id"))
    )


def _source_matches(source, chunk):
    return (
        tuple(
            int(getattr(source, key))
            for key in (
                "tenant_id",
                "knowledge_base_id",
                "document_id",
                "revision_id",
                "generation_id",
            )
        )
        == (
            chunk.tenant_id,
            chunk.knowledge_base_id,
            chunk.document_id,
            chunk.revision_id,
            chunk.generation_id,
        )
        and source.content_sha256 == chunk.content_sha256
        and hashlib.sha256(chunk.content.encode()).hexdigest() == chunk.content_sha256
        and chunk.content[source.quote_start : source.quote_end] == source.source_quote
        and source.quote_end == source.quote_start + len(source.source_quote)
    )


def retain_complete_paths(candidates, *, allowed_chunk_ids=None):
    paths = complete_paths(candidates)
    if allowed_chunk_ids is not None:
        allowed = {str(value) for value in allowed_chunk_ids}
        paths = tuple(path for path in paths if set(path.required_chunk_ids) <= allowed)
    valid = {path.path_id for path in paths}
    return tuple(
        replace(item, graph_paths=tuple(path for path in item.graph_paths if path.path_id in valid))
        for item in candidates
    )


def select_path_bundles(candidates, *, limit):
    """按最高排名出处选择整个来源组；组超预算时仍可保留独立原文。"""
    if not 1 <= limit <= 100:
        raise ValueError("证据预算应在 1 到 100 之间")
    candidates = retain_complete_paths(tuple(candidates))
    paths = complete_paths(candidates)
    by_id = {str(item.chunk.chunk_id): item for item in candidates}
    selected, seen = [], set()
    for candidate in candidates:
        chunk_id = str(candidate.chunk.chunk_id)
        if chunk_id in seen:
            continue
        for path in paths:
            if chunk_id not in path.required_chunk_ids:
                continue
            missing = set(path.required_chunk_ids) - seen
            if len(selected) + len(missing) > limit:
                continue
            for item in candidates:
                key = str(item.chunk.chunk_id)
                if key in missing:
                    selected.append(by_id[key])
                    seen.add(key)
        if chunk_id not in seen and len(selected) < limit:
            selected.append(candidate)
            seen.add(chunk_id)
        if len(selected) == limit:
            break
    return retain_complete_paths(tuple(selected))
