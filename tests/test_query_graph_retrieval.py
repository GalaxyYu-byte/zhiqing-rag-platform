"""图增强分支与基础召回融合，覆盖故障、权限变化和关系出处。"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_query_retrieval_service import (
    QUERY,
    FakeEmbedder,
    FakeRepository,
    FakeReranker,
    chunk,
    execute,
    hit,
    service,
)

from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_retrieval.fusion import RRF_K, fuse_candidates
from zhiqing_rag.query_retrieval.models import GraphRelationEvidence, RecallHit
from zhiqing_rag.query_routing.models import GraphParams, GraphPlan, GraphProjectionRef


def graph_plan(**changes):
    return GraphPlan(
        original_query=QUERY,
        query=QUERY,
        requested_route="hybrid_graph",
        reason_code="GRAPH_APPROVED",
        params=GraphParams(
            seed_entity_uids=("seed",),
            projections=(
                GraphProjectionRef(
                    tenant_id="1",
                    document_id="4",
                    generation_id="5",
                    projection_id="6",
                    namespace="test",
                ),
            ),
            **changes,
        ),
    )


def graph_hit():
    quote = "工单关闭前须由申请人确认。"
    relation = GraphRelationEvidence(
        assertion_id="assertion",
        seed_entity_uid="seed",
        subject_entity_uid="seed",
        object_entity_uid="actor",
        predicate="RESPONSIBLE_FOR",
        condition_text="关闭前",
        source_quote=quote,
        quote_start=0,
        quote_end=len(quote),
    )
    return RecallHit(chunk(12, content=quote), 1.0, (relation,))


class Graph:
    def __init__(self, hits=(), *, failure=None, delay=0):
        self.hits = hits
        self.failure = failure
        self.delay = delay
        self.calls = []

    async def recall(self, user, scope, params, limit):
        self.calls.append((user, scope, params, limit))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure:
            raise self.failure
        return SimpleNamespace(hits=self.hits, seed_count=1, assertion_count=len(self.hits))


def test_three_branch_rrf_preserves_condition_and_deduplicates_chunk():
    graph = graph_hit()
    fused = fuse_candidates(
        (hit(10), RecallHit(graph.chunk, 0.8)),
        (RecallHit(graph.chunk, 7.0),),
        graph=(graph,),
        limit=5,
    )
    first = fused[0]
    assert first.chunk.chunk_id == 12
    assert first.scores.fusion_score == pytest.approx(2 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert first.scores.graph_rank == 1
    assert first.graph_relations[0].condition_text == "关闭前"
    assert len(fused) == 2


async def test_graph_only_original_evidence_enters_existing_rerank_and_output():
    graph = Graph((graph_hit(),))
    reranker = FakeReranker()
    executor = service(FakeRepository(dense=(hit(10),)), reranker=reranker, graph_service=graph)
    result = await execute(executor, query_plan=graph_plan())
    assert result.status == "ok"
    assert result.trace.graph_status == "succeeded"
    assert (result.trace.graph_count, result.trace.graph_seed_count) == (1, 1)
    assert (result.trace.graph_hops, result.trace.graph_assertion_count) == (1, 1)
    assert any(graph_hit().chunk.content in document for document in reranker.calls[0][1])
    evidence = next(item for item in result.evidence if item.chunk_id == "12")
    assert evidence.scores.graph_rank == 1
    assert evidence.graph_relations == graph_hit().graph_relations
    assert evidence.content == graph_hit().chunk.content


@pytest.mark.parametrize("graph", [None, Graph(failure=RuntimeError("driver"))])
async def test_graph_failure_falls_back_to_usable_text_candidates(graph):
    result = await execute(
        service(FakeRepository(dense=(hit(10),)), graph_service=graph),
        query_plan=graph_plan(),
    )
    assert result.status == "degraded"
    assert result.degradation_codes == ("GRAPH_UNAVAILABLE",)
    assert result.trace.graph_status == "failed"
    assert result.evidence[0].chunk_id == "10"


async def test_graph_timeout_does_not_cancel_text_recall():
    result = await execute(
        service(
            FakeRepository(dense=(hit(10),)),
            graph_service=Graph(delay=1),
            graph_timeout_seconds=0.01,
        ),
        query_plan=graph_plan(),
    )
    assert result.status == "degraded"
    assert "GRAPH_UNAVAILABLE" in result.degradation_codes
    assert result.trace.dense_status == "succeeded"


async def test_revoked_or_empty_graph_is_explicit_text_fallback():
    result = await execute(
        service(FakeRepository(dense=(hit(10),)), graph_service=Graph()),
        query_plan=graph_plan(),
    )
    assert result.trace.graph_status == "empty"
    assert result.trace.graph_hops == 0
    assert result.degradation_codes == ("GRAPH_NO_EVIDENCE",)
    assert not result.evidence[0].graph_relations


async def test_graph_can_supply_original_evidence_when_both_text_channels_fail():
    result = await execute(
        service(
            FakeRepository(failures={"bm25": RuntimeError()}),
            embedder=FakeEmbedder(failure=RuntimeError()),
            graph_service=Graph((graph_hit(),)),
        ),
        query_plan=graph_plan(),
    )
    assert result.status == "degraded"
    assert result.evidence[0].chunk_id == "12"


async def test_all_channels_unavailable_fail_closed():
    with pytest.raises(AuthError) as error:
        await execute(
            service(
                FakeRepository(failures={"bm25": RuntimeError()}),
                embedder=FakeEmbedder(failure=RuntimeError()),
                graph_service=Graph(failure=RuntimeError()),
            ),
            query_plan=graph_plan(),
        )
    assert error.value.code == "RETRIEVAL_UNAVAILABLE"


async def test_graph_cannot_swallow_session_expiry_as_a_degradation():
    graph = Graph(failure=AuthError(401, "SESSION_EXPIRED", "已失效"))
    with pytest.raises(AuthError) as error:
        await execute(service(graph_service=graph), query_plan=graph_plan())
    assert error.value.code == "SESSION_EXPIRED"


async def test_out_of_scope_graph_hit_is_rejected_before_rerank():
    graph = Graph((replace(graph_hit(), chunk=chunk(12, document_id=99)),))
    reranker = FakeReranker()
    with pytest.raises(AuthError) as error:
        await execute(service(graph_service=graph, reranker=reranker), query_plan=graph_plan())
    assert error.value.status_code == 403
    assert not reranker.calls


async def test_foreign_graph_projection_is_rejected_before_any_recall():
    query_plan = graph_plan()
    foreign = query_plan.params.projections[0].model_copy(update={"tenant_id": "99"})
    query_plan = query_plan.model_copy(
        update={"params": query_plan.params.model_copy(update={"projections": (foreign,)})}
    )
    executor = service(graph_service=Graph((graph_hit(),)))
    with pytest.raises(AuthError):
        await execute(executor, query_plan=query_plan)
    assert not executor.repository.calls
    assert not executor.graph_service.calls


async def test_graph_candidate_revoked_during_rerank_drops_its_relation_metadata():
    repository = FakeRepository(dense=(hit(10),), validations=(None, lambda items: items[:1]))
    result = await execute(
        service(repository, graph_service=Graph((graph_hit(),))), query_plan=graph_plan()
    )
    assert result.degradation_codes == ("SCOPE_CHANGED",)
    assert [item.chunk_id for item in result.evidence] == ["10"]
    assert not result.evidence[0].graph_relations


async def test_basic_hybrid_does_not_query_graph_service():
    graph = Graph((graph_hit(),))
    result = await execute(service(FakeRepository(dense=(hit(10),)), graph_service=graph))
    assert result.trace.graph_status == "skipped"
    assert not graph.calls


@pytest.mark.parametrize("failure", [None, RuntimeError("projection lookup")])
async def test_retired_or_unverifiable_graph_provenance_is_stripped_but_current_text_remains(
    failure,
):
    class ChangedGraph(Graph):
        async def revalidate_relations(self, user, scope, params, chunks):
            if failure:
                raise failure
            return frozenset()

    result = await execute(
        service(graph_service=ChangedGraph((graph_hit(),))), query_plan=graph_plan()
    )
    assert result.status == "degraded"
    expected = "GRAPH_PROVENANCE_UNAVAILABLE" if failure else "GRAPH_PROVENANCE_CHANGED"
    assert result.degradation_codes == (expected,)
    assert result.trace.graph_count == result.trace.graph_hops == 0
    evidence = result.evidence[0]
    assert evidence.content == graph_hit().chunk.content
    assert not evidence.graph_relations
    assert evidence.scores.graph_rank is None


async def test_current_graph_provenance_survives_final_ready_recheck():
    class CurrentGraph(Graph):
        async def revalidate_relations(self, user, scope, params, chunks):
            return frozenset(chunk.chunk_id for chunk in chunks)

    result = await execute(
        service(graph_service=CurrentGraph((graph_hit(),))), query_plan=graph_plan()
    )
    assert result.status == "ok"
    assert result.evidence[0].graph_relations == graph_hit().graph_relations


async def test_session_expired_during_final_graph_check_is_not_hidden():
    class ExpiredGraph(Graph):
        async def revalidate_relations(self, user, scope, params, chunks):
            raise AuthError(401, "SESSION_EXPIRED", "已失效")

    with pytest.raises(AuthError) as error:
        await execute(service(graph_service=ExpiredGraph((graph_hit(),))), query_plan=graph_plan())
    assert error.value.code == "SESSION_EXPIRED"


@pytest.mark.parametrize("facets", [False, True])
async def test_acl_revoked_in_final_graph_check_removes_original_text_and_facet_anchors(facets):
    class RevokedRepository(FakeRepository):
        revoked = False

        async def revalidate(self, user, scope, chunks):
            return tuple(item for item in chunks if not self.revoked or item.chunk_id != 12)

    repository = RevokedRepository(dense=(hit(10), hit(11)))

    class RevokedGraph(Graph):
        async def revalidate_relations(self, user, scope, params, chunks):
            repository.revoked = True
            return frozenset()

    query_plan = graph_plan()
    if facets:
        query_plan = query_plan.model_copy(update={"query": "工单和服务请求分别如何关闭？"})
    result = await execute(
        service(repository, graph_service=RevokedGraph((graph_hit(),))), query_plan=query_plan
    )
    assert result.status == "degraded"
    assert "GRAPH_PROVENANCE_CHANGED" in result.degradation_codes
    assert "SCOPE_CHANGED" in result.degradation_codes
    assert {item.chunk_id for item in result.evidence} == {"10", "11"}
    assert all(item.content != graph_hit().chunk.content for item in result.evidence)
    assert result.trace.authorized_count == 2
