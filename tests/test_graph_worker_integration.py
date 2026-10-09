"""可选 Neo4j 实测：隔离命名空间写图、重放与空结果替换。"""

import os
from uuid import uuid4

import pytest

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.workers.graph_extraction import Extraction
from zhiqing_rag.workers.graph_storage import GraphStore

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_GRAPH_INTEGRATION") != "1", reason="需要显式启用 Neo4j 实测"
)


@pytest.mark.asyncio
async def test_neo4j_idempotency_and_provenance():
    namespace = f"graph-test-{uuid4().hex}"
    settings = get_settings().model_copy(update={"graph_namespace": namespace})
    store = GraphStore(settings)
    scope = {"tenant_id": 1, "kb_id": 2, "document_id": 3, "revision_id": 4, "generation_id": 5}
    chunk = {
        "id": 6,
        "chunk_index": 0,
        "content": "财务审核报销申请",
        "content_sha256": "a" * 64,
        "page_number": 2,
        "section_title": "报销",
        "start_offset": 0,
        "end_offset": 8,
        "metadata_": {"source_refs": [{"page": 2}]},
    }
    extraction = Extraction.model_validate(
        {
            "entities": [
                {
                    "id": "a",
                    "type": "Actor",
                    "name": "财务",
                    "canonical_name": "财务",
                    "evidence_quote": "财务审核报销申请",
                },
                {
                    "id": "s",
                    "type": "Step",
                    "name": "审核报销申请",
                    "canonical_name": "审核报销申请",
                    "evidence_quote": "财务审核报销申请",
                },
            ],
            "relations": [
                {
                    "type": "REVIEWS",
                    "source_id": "a",
                    "target_id": "s",
                    "evidence_quote": "财务审核报销申请",
                    "condition_text": None,
                }
            ],
        }
    )
    try:
        await store.prepare()
        await store.write_chunk(scope, chunk, extraction)
        await store.write_chunk(scope, chunk, extraction)
        async with store.driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                "MATCH (a:Assertion {namespace:$namespace})-[:SUPPORTED_BY]->(e:Evidence) "
                "RETURN count(a) AS assertions, collect(e.quote) AS quotes",
                namespace=namespace,
            )
            row = await result.single()
            assert row["assertions"] == 1
            assert row["quotes"] == ["财务审核报销申请"]
        await store.write_chunk(scope, chunk, Extraction(entities=[], relations=[]))
        async with store.driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                "MATCH (e:Evidence {namespace:$namespace}) RETURN count(e) AS evidence",
                namespace=namespace,
            )
            assert (await result.single())["evidence"] == 0
        await store.write_chunk(scope, chunk, extraction)
        await store.delete_generation(1, 3, 5, namespace)
        with pytest.raises(ValueError, match="GRAPH_GENERATION_RETIRED"):
            await store.write_chunk(scope, chunk, extraction)
        async with store.driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                "MATCH (e:Evidence {namespace:$namespace}) RETURN count(e) AS evidence",
                namespace=namespace,
            )
            assert (await result.single())["evidence"] == 0
    finally:
        async with store.driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                "MATCH (n) WHERE n.namespace=$namespace DETACH DELETE n", namespace=namespace
            )
            await result.consume()
        await store.close()


@pytest.mark.asyncio
async def test_candidate_namespace_preserves_old_graph_and_shared_entity():
    """在真实 Neo4j 的专用测试空间验证共享实体和版本隔离，不调用外部模型。"""
    base_namespace = f"graph-test-base-{uuid4().hex}"
    candidate_namespace = f"graph-test-candidate-{uuid4().hex}"
    settings = get_settings()
    base = GraphStore(settings.model_copy(update={"graph_namespace": base_namespace}))
    candidate = GraphStore(settings.model_copy(update={"graph_namespace": candidate_namespace}))
    scope = {"tenant_id": 1, "kb_id": 2, "document_id": 3, "revision_id": 4, "generation_id": 5}
    other_scope = {**scope, "document_id": 13, "revision_id": 14, "generation_id": 15}
    chunk = {
        "id": 6,
        "chunk_index": 0,
        "content": "财务审核报销申请",
        "content_sha256": "a" * 64,
        "page_number": None,
        "section_title": None,
        "start_offset": None,
        "end_offset": None,
        "metadata_": {},
    }
    other_chunk = {**chunk, "id": 16}
    extraction = Extraction.model_validate(
        {
            "entities": [
                {
                    "id": "a",
                    "type": "Actor",
                    "name": "财务",
                    "canonical_name": "财务",
                    "evidence_quote": chunk["content"],
                },
                {
                    "id": "s",
                    "type": "Step",
                    "name": "审核报销申请",
                    "canonical_name": "审核报销申请",
                    "evidence_quote": chunk["content"],
                },
            ],
            "relations": [
                {
                    "type": "REVIEWS",
                    "source_id": "a",
                    "target_id": "s",
                    "evidence_quote": chunk["content"],
                    "condition_text": None,
                }
            ],
        }
    )
    try:
        await base.prepare()
        await candidate.prepare()
        await base.write_chunk(scope, chunk, extraction)
        await base.write_chunk(other_scope, other_chunk, extraction)
        old = await base.snapshot(scope)
        other = await base.snapshot(other_scope)
        shared_actor = next(row["key"] for row in old["entities"] if row["type"] == "Actor")
        assert shared_actor == next(
            row["key"] for row in other["entities"] if row["type"] == "Actor"
        )
        await candidate.write_chunk(scope, chunk, extraction)
        await candidate.write_chunk(scope, chunk, extraction)
        assert len((await candidate.snapshot(scope))["assertions"]) == 1
        assert await base.snapshot(scope) == old
        assert await base.snapshot(other_scope) == other
        await candidate.delete_generation(1, 3, 5, candidate_namespace)
        assert await base.snapshot(scope) == old
        assert await base.snapshot(other_scope) == other
        await base.delete_generation(1, 3, 5, base_namespace)
        assert await base.snapshot(other_scope) == other
    finally:
        for store, namespace in ((base, base_namespace), (candidate, candidate_namespace)):
            async with store.driver.session(database=settings.neo4j_database) as session:
                result = await session.run(
                    "MATCH (n) WHERE n.namespace=$namespace DETACH DELETE n",
                    namespace=namespace,
                )
                await result.consume()
            await store.close()
