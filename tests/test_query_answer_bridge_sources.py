"""独立桥接出处在撤权或投影退休后，不能藏在完整路径中继续发布。"""

from dataclasses import replace

import pytest
from test_query_answer_paths import Graph, paths_retrieval
from test_query_answering import Provider, draft
from test_query_graph_paths import two_hop_plan
from test_query_retrieval_service import FakeRepository, chunk, identity, scope

from zhiqing_rag.core.config import Settings
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_retrieval.fusion import FusedCandidate
from zhiqing_rag.query_retrieval.models import EvidenceScores
from zhiqing_rag.query_retrieval.service import HybridRetrievalService
from zhiqing_rag.query_routing.models import GraphProjectionRef
from zhiqing_rag.query_routing.scope import PublishedDocument

BRIDGE_TEXT = "独立桥接出处：本流程表单又称申请单，映射标识 BRIDGE_SOURCE_ONLY。"
ALL_SOURCES = frozenset((10, 12, 14))
HOP_SOURCES = frozenset((10, 12))


def bridge_fixture():
    retrieval = paths_retrieval()
    proof = chunk(
        14, content=BRIDGE_TEXT, document_id=8, revision_id=14, generation_id=15,
        document_title="实体映射说明", document_name="实体映射说明.md",
    )
    evidence = HybridRetrievalService._evidence(
        FusedCandidate(proof, EvidenceScores(fusion_score=0.1)), 3
    )
    path = retrieval.graph_paths[0]
    source = path.hops[0].source.model_copy(update={
        "document_id": "8", "revision_id": "14", "generation_id": "15",
        "chunk_id": "14", "projection_id": "17", "namespace": "bridge-only",
        "content_sha256": proof.content_sha256, "source_quote": BRIDGE_TEXT,
        "quote_start": 0, "quote_end": len(BRIDGE_TEXT),
    })
    bridge = path.bridges[0].model_copy(update={
        "sources": (path.bridges[0].sources[0], source), "basis": "explicit_alias",
    })
    path = path.model_copy(update={
        "bridges": (bridge,), "required_chunk_ids": ("10", "12", "14"),
    })
    retrieval = retrieval.model_copy(update={
        "evidence": (*retrieval.evidence, evidence), "graph_paths": (path,),
        "trace": retrieval.trace.model_copy(update={"returned_count": 3}),
    })
    plan = two_hop_plan()
    plan = plan.model_copy(update={"params": plan.params.model_copy(update={
        "seed_entity_uids": ("step",),
        "projections": (*plan.params.projections, GraphProjectionRef(
            tenant_id="1", document_id="8", generation_id="15",
            projection_id="17", namespace="bridge-only",
        )),
    })})
    access = scope(second_document=True)
    access = replace(
        access, document_ids=(*access.document_ids, 8),
        published_documents=(*access.published_documents, PublishedDocument(8, 3, 15)),
    )
    return retrieval, plan, access


class BridgeRepository(FakeRepository):
    revoked = False

    async def revalidate(self, user, access, records):
        current = await super().revalidate(user, access, records)
        return tuple(row for row in current if row.document_id != 8) if self.revoked else current


def supported_draft():
    return draft(statements=[{
        "text": "步骤需要表单，表单包含字段。", "citations": [
            {"citation_id": "E1", "quote": "步骤需要表单。"},
            {"citation_id": "E2", "quote": "表单包含字段。"},
            {"citation_id": "E3", "quote": BRIDGE_TEXT},
        ],
    }])


class BridgeProvider(Provider):
    async def generate(self, query, evidence):
        available = {item.citation_id for item in evidence}
        self.value = self.value.model_copy(update={"statements": [
            statement.model_copy(update={"citations": [
                citation for citation in statement.citations if citation.citation_id in available
            ]})
            for statement in self.value.statements
        ]})
        return await super().generate(query, evidence)


async def execute_bridge(*, graph, repository=None, provider=None):
    retrieval, plan, access = bridge_fixture()
    provider = provider or BridgeProvider(supported_draft())
    service = QueryAnswerService(
        repository or BridgeRepository(), provider, Settings(_env_file=None),
        graph_service=graph,
    )
    retrieval, answer = await service.execute(
        identity=identity(), scope=access, plan=plan, retrieval=retrieval,
    )
    return retrieval, answer, provider


async def test_bridge_only_source_is_checked_even_without_its_own_relation():
    graph = Graph([ALL_SOURCES] * 3)
    retrieval, answer, provider = await execute_bridge(graph=graph)
    assert answer.status == "answered" and len(retrieval.graph_paths) == 1
    assert all({row.chunk_id for row in records} == ALL_SOURCES for records in graph.calls)
    assert len(graph.calls) == 3
    assert retrieval.evidence[2].content == BRIDGE_TEXT
    assert not retrieval.evidence[2].graph_relations
    assert [name for name, _ in provider.calls] == ["generate", "review"]


@pytest.mark.parametrize("when", ["before_answer", "during_ready_check"])
async def test_revoked_bridge_only_source_never_reaches_provider_or_public_response(when):
    repo = BridgeRepository()
    repo.revoked = when == "before_answer"
    on_check = (lambda: setattr(repo, "revoked", True)) if when == "during_ready_check" else None
    graph = Graph([ALL_SOURCES] * 3, on_check=on_check)
    retrieval, answer, provider = await execute_bridge(graph=graph, repository=repo)
    assert answer.status == "answered"
    assert not retrieval.graph_paths and retrieval.trace.graph_path_count == 0
    assert {item.chunk_id for item in retrieval.evidence} == {"10", "12"}
    assert "BRIDGE_SOURCE_ONLY" not in retrieval.model_dump_json()
    assert "BRIDGE_SOURCE_ONLY" not in answer.model_dump_json()
    assert [name for name, _ in provider.calls] == ["generate", "review"]
    assert all("14" not in {item.chunk_id for item in records} for _, records in provider.calls)
    assert "GRAPH_PROVENANCE_CHANGED" in retrieval.degradation_codes


@pytest.mark.parametrize("stage", ["generate", "review"])
async def test_bridge_revocation_after_model_call_prevents_later_calls_and_answer_release(stage):
    repo = BridgeRepository()

    class RevokingProvider(BridgeProvider):
        async def generate(self, query, evidence):
            result = await super().generate(query, evidence)
            if stage == "generate":
                repo.revoked = True
            return result

        async def review(self, query, evidence, value):
            result = await super().review(query, evidence, value)
            repo.revoked = True
            return result

    provider = RevokingProvider(supported_draft())
    retrieval, answer, provider = await execute_bridge(
        graph=Graph([ALL_SOURCES] * 3), repository=repo, provider=provider,
    )
    assert answer.status == "rejected" and not answer.statements
    assert "ANSWER_SCOPE_CHANGED" in answer.codes
    assert not retrieval.graph_paths and retrieval.trace.graph_path_count == 0
    assert {item.chunk_id for item in retrieval.evidence} == {"10", "12"}
    assert "BRIDGE_SOURCE_ONLY" not in retrieval.model_dump_json()
    assert "BRIDGE_SOURCE_ONLY" not in answer.model_dump_json()
    # 外发时该来源仍有权限；撤权后的复查须停止下一次外发和最终草稿发布。
    assert [name for name, _ in provider.calls] == (
        ["generate"] if stage == "generate" else ["generate", "review"]
    )


@pytest.mark.parametrize("at", [0, 1, 2])
async def test_stale_bridge_only_projection_drops_whole_path_but_keeps_authorized_raw_fact(at):
    graph = Graph([ALL_SOURCES] * at + [HOP_SOURCES])
    retrieval, answer, provider = await execute_bridge(graph=graph)
    assert answer.status == "answered"
    assert retrieval.status == "degraded" and not retrieval.graph_paths
    assert retrieval.trace.graph_path_count == 0 and retrieval.trace.graph_hops == 1
    assert "GRAPH_PROVENANCE_CHANGED" in retrieval.degradation_codes
    assert {item.chunk_id for item in retrieval.evidence} == {"10", "12", "14"}
    assert retrieval.evidence[2].content == BRIDGE_TEXT
    assert not retrieval.evidence[2].graph_relations
    assert all({item.chunk_id for item in records} == {"10", "12", "14"}
               for _, records in provider.calls)
