"""重排与权限变化期间保留整条路径，预算不足时不发布半条链。"""

from dataclasses import replace
from types import SimpleNamespace

from test_query_graph_retrieval import Graph, graph_plan
from test_query_retrieval_service import (
    FakeRepository,
    FakeReranker,
    chunk,
    execute,
    hit,
    scope,
    service,
)

from zhiqing_rag.query_retrieval.fusion import fuse_candidates
from zhiqing_rag.query_retrieval.models import (
    GraphBridgeEvidence,
    GraphHopEvidence,
    GraphPathEvidence,
    GraphRelationEvidence,
    GraphSourceEvidence,
    RecallHit,
)
from zhiqing_rag.query_retrieval.paths import complete_paths, select_path_bundles
from zhiqing_rag.query_routing.models import GraphProjectionRef


def path_hits():
    first = chunk(10, content="步骤需要表单。")
    second = chunk(12, content="表单包含字段。", document_id=6, generation_id=9)
    sources, hops = [], []
    for record, subject, object_, predicate, projection in (
        (first, "step", "material-a", "REQUIRES_MATERIAL", 6),
        (second, "material-b", "field", "HAS_FIELD", 7),
    ):
        source = GraphSourceEvidence(
            tenant_id=str(record.tenant_id),
            knowledge_base_id=str(record.knowledge_base_id),
            document_id=str(record.document_id),
            revision_id=str(record.revision_id),
            generation_id=str(record.generation_id),
            chunk_id=str(record.chunk_id),
            projection_id=str(projection),
            namespace="test",
            content_sha256=record.content_sha256,
            source_quote=record.content,
            quote_start=0,
            quote_end=len(record.content),
        )
        relation = GraphRelationEvidence(
            assertion_id=f"assertion-{record.chunk_id}",
            seed_entity_uid=subject,
            subject_entity_uid=subject,
            object_entity_uid=object_,
            predicate=predicate,
            condition_text="",
            source_quote=record.content,
            quote_start=0,
            quote_end=len(record.content),
        )
        sources.append(source)
        hops.append(GraphHopEvidence(relation=relation, source=source, traversal="outgoing"))
    path = GraphPathEvidence(
        path_id="path",
        knowledge_base_id="3",
        hops=tuple(hops),
        bridges=(
            GraphBridgeEvidence(
                source_entity_uid="material-a",
                target_entity_uid="material-b",
                entity_type="Material",
                canonical_name="表单",
                basis="shared_context",
                sources=tuple(sources),
            ),
        ),
        required_chunk_ids=("10", "12"),
    )
    return tuple(
        RecallHit(record, 1.0, (hop.relation,), (path,))
        for record, hop in zip((first, second), hops, strict=True)
    )


def two_hop_plan(**changes):
    plan = graph_plan(max_hops=2, **changes)
    return plan.model_copy(
        update={
            "params": plan.params.model_copy(
                update={
                    "projections": (
                        *plan.params.projections,
                        GraphProjectionRef(
                            tenant_id="1",
                            document_id="6",
                            generation_id="9",
                            projection_id="7",
                            namespace="test",
                        ),
                    ),
                }
            )
        }
    )


class PathGraph(Graph):
    async def recall(self, user, scope, params, limit):
        value = await super().recall(user, scope, params, limit)
        return SimpleNamespace(**vars(value), path_count=1)

    async def revalidate_relations(self, user, scope, params, chunks):
        return frozenset(item.chunk_id for item in chunks)


def test_candidate_budget_keeps_partner_even_below_an_unrelated_candidate():
    graph = path_hits()
    fused = fuse_candidates((graph[0], hit(20)), (), graph=graph, limit=2)
    assert {item.chunk.chunk_id for item in fused} == {10, 12}
    assert len(complete_paths(fused)) == 1


def test_budget_one_returns_independent_raw_evidence_without_path():
    graph = path_hits()
    fused = fuse_candidates((graph[0],), (), graph=graph, limit=1)
    assert len(fused) == 1
    assert not complete_paths(fused)
    assert not fused[0].graph_paths
    assert fused[0].graph_relations


def test_missing_bridge_source_invalidates_whole_path():
    graph = path_hits()
    path = graph[0].graph_paths[0]
    missing = path.model_copy(update={"required_chunk_ids": ("10", "12", "99")})
    graph = tuple(replace(item, graph_paths=(missing,)) for item in graph)
    fused = fuse_candidates((), (), graph=graph, limit=5)
    assert all(not item.graph_paths for item in fused)


def test_shared_uid_without_bridge_requires_the_same_document_projection():
    first, second = path_hits()
    path = first.graph_paths[0]
    record = replace(second.chunk, document_id=first.chunk.document_id,
                     generation_id=first.chunk.generation_id)
    relation = path.hops[1].relation.model_copy(update={
        "seed_entity_uid": "material-a", "subject_entity_uid": "material-a"
    })
    source = path.hops[1].source.model_copy(update={
        "document_id": str(record.document_id), "generation_id": str(record.generation_id)
    })
    path = path.model_copy(update={"bridges": (), "hops": (
        path.hops[0], path.hops[1].model_copy(update={"source": source, "relation": relation})
    )})
    first = replace(first, graph_paths=(path,))
    second = replace(second, chunk=record, graph_relations=(relation,), graph_paths=(path,))
    assert len(complete_paths((first, second))) == 1
    for key, value in (("namespace", "other_projection"), ("generation_id", "99")):
        changed = path.model_copy(update={"hops": (
            path.hops[0], path.hops[1].model_copy(update={
                "source": source.model_copy(update={key: value})
            })
        )})
        changed_record = replace(record, generation_id=99) if key == "generation_id" else record
        assert not complete_paths((replace(first, graph_paths=(changed,)),
                                   replace(second, chunk=changed_record, graph_paths=(changed,))))


def test_duplicated_required_source_ids_do_not_count_as_a_complete_path():
    graph = path_hits()
    bad = graph[0].graph_paths[0].model_copy(update={"required_chunk_ids": ("10", "12", "12")})
    assert not complete_paths(tuple(replace(hit, graph_paths=(bad,)) for hit in graph))


def test_path_quote_must_match_the_current_raw_source():
    graph = path_hits()
    path = graph[0].graph_paths[0]
    wrong = path.model_copy(
        update={
            "hops": (
                path.hops[0].model_copy(
                    update={
                        "source": path.hops[0].source.model_copy(
                            update={"source_quote": "被伪造的原文"}
                        )
                    }
                ),
                path.hops[1],
            )
        }
    )
    graph = tuple(replace(item, graph_paths=(wrong,)) for item in graph)
    assert not complete_paths(fuse_candidates((), (), graph=graph, limit=5))


async def test_reranking_preserves_complete_chain_in_final_top_k():
    graph = path_hits()
    reranker = FakeReranker(scores=((0, 0.9), (1, 0.01), (2, 0.8)))
    result = await execute(
        service(
            FakeRepository(dense=(graph[0], hit(20))),
            reranker=reranker,
            graph_service=PathGraph(graph),
        ),
        query_plan=two_hop_plan(top_k=2),
        access=scope(second_document=True),
    )
    assert {item.chunk_id for item in result.evidence} == {"10", "12"}
    assert len(result.graph_paths) == 1
    assert result.trace.graph_path_count == 1
    assert result.trace.graph_path_candidate_count == 1
    assert result.trace.graph_hops == 2


async def test_partial_revocation_strips_path_from_surviving_raw_evidence():
    graph = path_hits()
    repo = FakeRepository(
        validations=(None, lambda rows: tuple(row for row in rows if row.chunk_id != 12))
    )
    result = await execute(
        service(repo, graph_service=PathGraph(graph)),
        query_plan=two_hop_plan(rerank=False),
        access=scope(second_document=True),
    )
    assert not result.graph_paths
    assert result.trace.graph_path_count == 0
    assert "GRAPH_PATH_INCOMPLETE" in result.degradation_codes


def test_top_k_selection_does_not_expand_the_requested_limit():
    fused = fuse_candidates((), (), graph=path_hits(), limit=5)
    assert len(select_path_bundles(fused, limit=1)) == 1
