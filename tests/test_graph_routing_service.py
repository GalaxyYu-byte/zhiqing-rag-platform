import asyncio
from types import SimpleNamespace

import pytest

from zq_rag_app.services.graph_routing_service import check_graph_coverage
from zq_rag_app.query_routing.models import GraphEntityCandidate


class Postgres:
    def __init__(self, rows=()):
        self.rows = rows
        self.statement = None
        self.rolled_back = False

    async def execute(self, statement):
        self.statement = statement
        return SimpleNamespace(all=lambda: self.rows)

    async def rollback(self):
        self.rolled_back = True


class Graph:
    def __init__(self, records):
        self.records = records
        self.parameters = None
        self.query = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, query, **parameters):
        self.query = query
        self.parameters = parameters
        async def data():
            return self.records
        return SimpleNamespace(data=data)


@pytest.mark.asyncio
async def test_resolver_requires_authorized_current_active_version():
    pg = Postgres([SimpleNamespace(id=7, version=2)])
    graph = Graph([{"name": "aurora-kb", "uid": "entity-1"}])
    result = await check_graph_coverage(pg, entity_names=["Aurora-KB"], kb_ids=[4], doc_ids=[7],
                                      graph_session_factory=lambda: graph)
    assert result.status == "covered" and result.seed_entity_uids == ("entity-1",)
    assert graph.parameters == {"entity_names": ["aurora-kb"], "kb_ids": [4],
                                "entity_candidates": [{"index": 0, "name": "aurora-kb", "entity_type": None, "qualifiers": []}],
                                "document_versions": [{"doc_id": 7, "version": 2}]}
    assert "chunk.graph_status = 'ACTIVE'" in graph.query
    assert "doc.active_graph_version = chunk.doc_version" in graph.query
    assert "version.doc_id = chunk.doc_id AND version.version = chunk.doc_version" in graph.query
    assert "claim.kb_id = seed.kb_id" in graph.query
    assert "toLower(trim(alias)) = candidate.name" in graph.query
    assert "LIMIT 2" in graph.query
    sql = str(pg.statement)
    assert "kb_document.kb_id" in sql and "kb_document.is_deleted" in sql and "kb_document.status" in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("records,status", [
    ([], "unmatched"),
    ([{"name": "aurora-kb", "uid": "one"}, {"name": "aurora-kb", "uid": "two"}], "ambiguous"),
    ([{"name": "aurora-kb", "uid": "one"}], "unmatched"),
    ([{"name": "aurora-kb", "uid": "one"}, {"name": "team", "uid": "two"}], "covered"),
])
async def test_all_requested_names_must_resolve_uniquely(records, status):
    result = await check_graph_coverage(Postgres([SimpleNamespace(id=7, version=2)]),
        entity_names=["Aurora-KB", "Team"], kb_ids=[4], doc_ids=[7], graph_session_factory=lambda: Graph(records))
    assert result.status == status


@pytest.mark.asyncio
async def test_no_current_versions_does_not_open_neo4j():
    def forbidden():
        raise AssertionError("不应连接 Neo4j")
    result = await check_graph_coverage(Postgres(), entity_names=["Aurora-KB"], kb_ids=[4], doc_ids=[7],
                                      graph_session_factory=forbidden)
    assert result.status == "unmatched"


@pytest.mark.asyncio
async def test_cancelled_postgres_read_resets_transaction():
    class CancelledPostgres(Postgres):
        async def execute(self, statement):
            raise asyncio.CancelledError()
    pg = CancelledPostgres()
    with pytest.raises(asyncio.CancelledError):
        await check_graph_coverage(pg, entity_names=["Aurora-KB"], kb_ids=[4], doc_ids=[7])
    assert pg.rolled_back


@pytest.mark.asyncio
@pytest.mark.parametrize("auxiliary_uids", [[], ["tech-a", "tech-b"]])
async def test_missing_or_ambiguous_auxiliary_keeps_confirmed_subject(auxiliary_uids):
    records = [{"candidate_index": 0, "name": "project", "uid": "project-one"}]
    records += [{"candidate_index": 1, "name": "neo4j", "uid": uid} for uid in auxiliary_uids]
    candidates = [GraphEntityCandidate("Project"), GraphEntityCandidate("Neo4j", "TECHNOLOGY", False)]
    result = await check_graph_coverage(Postgres([SimpleNamespace(id=7, version=2)]),
        entity_names=[c.name for c in candidates], entity_candidates=candidates, kb_ids=[4], doc_ids=[7],
        graph_session_factory=lambda: Graph(records))
    assert result.status == "partial"
    assert result.seed_entity_uids == ("project-one",)
    assert result.unresolved_auxiliary == ("Neo4j",)


@pytest.mark.asyncio
@pytest.mark.parametrize("subject_uids,status", [([], "unmatched"), (["one", "two"], "ambiguous")])
async def test_required_subject_still_blocks_even_if_auxiliary_resolves(subject_uids, status):
    records = [{"candidate_index": 0, "uid": uid} for uid in subject_uids]
    records.append({"candidate_index": 1, "uid": "tech"})
    candidates = [GraphEntityCandidate("Project"), GraphEntityCandidate("Neo4j", "TECHNOLOGY", False)]
    result = await check_graph_coverage(Postgres([SimpleNamespace(id=7, version=2)]),
        entity_names=[c.name for c in candidates], entity_candidates=candidates, kb_ids=[4], doc_ids=[7],
        graph_session_factory=lambda: Graph(records))
    assert result.status == status and not result.seed_entity_uids


@pytest.mark.asyncio
async def test_same_name_different_types_and_grounded_qualifiers_are_separate_candidates():
    graph = Graph([{"candidate_index": 0, "name": "张三", "uid": "person"},
                   {"candidate_index": 1, "name": "张三", "uid": "system"}])
    candidates = [GraphEntityCandidate("张三", "PERSON", True, ("研发部",)),
                  GraphEntityCandidate("张三", "SYSTEM", False)]
    result = await check_graph_coverage(Postgres([SimpleNamespace(id=7, version=2)]),
        entity_names=["张三"], entity_candidates=candidates, kb_ids=[4], doc_ids=[7],
        graph_session_factory=lambda: graph)
    assert result.status == "covered" and result.seed_entity_uids == ("person", "system")
    assert graph.parameters["entity_candidates"] == [
        {"index": 0, "name": "张三", "entity_type": "PERSON", "qualifiers": ["研发部"]},
        {"index": 1, "name": "张三", "entity_type": "SYSTEM", "qualifiers": []},
    ]
    # 身份限定的证据同样需要当前授权版本；负向关系和一般关联不能证明身份。
    assert "identity_claim.polarity = 'POSITIVE'" in graph.query
    assert "identity_claim.predicate IN ['PART_OF', 'OWNED_BY', 'LOCATED_IN']" in graph.query
    assert "identity_doc.active_graph_version = identity_chunk.doc_version" in graph.query
    assert "version.doc_id = identity_chunk.doc_id AND version.version = identity_chunk.doc_version" in graph.query
