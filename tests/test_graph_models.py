from pathlib import Path

import pytest
from pydantic import ValidationError

from zq_rag_app.graph.identity import (
    build_chunk_uid,
    build_entity_uid,
)
from zq_rag_app.graph.models import (
    ExtractedEntity,
    GraphChunkContext,
    GraphExtraction,
    GraphImportSample,
)
from zq_rag_app.graph.repository import GraphRepository
from zq_rag_app.graph.schema import load_schema_statements


def _context(*, chunk_index: int = 0) -> GraphChunkContext:
    return GraphChunkContext(
        kb_id=4,
        doc_id=26,
        doc_version=2,
        chunk_index=chunk_index,
        file_name="故障复盘.md",
        section_title="根因",
        content="订单服务依赖 Redis。Redis 故障导致订单服务超时。",
    )


def _extraction() -> GraphExtraction:
    return GraphExtraction.model_validate(
        {
            "schema_version": "1.0",
            "entities": [
                {
                    "local_id": "e1",
                    "entity_type": "SERVICE",
                    "name": "订单服务",
                    "aliases": ["订单服务", "Order Service", "order service"],
                    "description": None,
                    "external_id_source": None,
                    "external_id": None,
                    "evidence_quote": "订单服务",
                },
                {
                    "local_id": "e2",
                    "entity_type": "TECHNOLOGY",
                    "name": "Redis",
                    "aliases": [],
                    "description": None,
                    "external_id_source": None,
                    "external_id": None,
                    "evidence_quote": "Redis",
                },
            ],
            "relations": [
                {
                    "source_local_id": "e1",
                    "target_local_id": "e2",
                    "predicate": "DEPENDS_ON",
                    "polarity": "POSITIVE",
                    "confidence": 0.98,
                    "evidence_quote": "订单服务依赖 Redis。",
                    "valid_from": None,
                    "valid_to": None,
                }
            ],
        }
    )


def test_extraction_deduplicates_aliases_case_insensitively() -> None:
    extraction = _extraction()

    assert extraction.entities[0].aliases == ["Order Service"]


def test_extraction_rejects_unknown_fields() -> None:
    payload = _extraction().model_dump()
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        GraphExtraction.model_validate(payload)


def test_extraction_rejects_unknown_relation_reference() -> None:
    payload = _extraction().model_dump()
    payload["relations"][0]["target_local_id"] = "e99"

    with pytest.raises(ValidationError, match="target_local_id=e99"):
        GraphExtraction.model_validate(payload)


def test_evidence_validation_allows_only_whitespace_differences() -> None:
    extraction = _extraction()
    extraction.relations[0].evidence_quote = "订单服务依赖   Redis。"

    extraction.validate_evidence(_context().content)

    extraction.relations[0].evidence_quote = "数据库故障导致超时"
    with pytest.raises(ValueError, match="证据不在当前 Chunk"):
        extraction.validate_evidence(_context().content)


def test_evidence_validation_restores_exact_source_quotes() -> None:
    extraction = GraphExtraction.model_validate(
        {
            "schema_version": "1.0",
            "entities": [
                {
                    "local_id": "e1",
                    "entity_type": "SERVICE",
                    "name": "订单服务",
                    "evidence_quote": '订单服务 依赖 "Redis"',
                }
            ],
            "relations": [],
        }
    )

    extraction.validate_evidence('记录：订单服务  依赖 “Redis”')

    assert extraction.entities[0].evidence_quote == "订单服务  依赖 “Redis”"


def test_discard_unsupported_evidence_removes_entity_and_its_relations() -> None:
    extraction = _extraction()
    extraction.entities[1].evidence_quote = "只在文件名里的 Redis.pdf"

    removed = extraction.discard_unsupported_evidence(_context().content)

    assert removed == (1, 1)
    assert [entity.local_id for entity in extraction.entities] == ["e1"]
    assert extraction.relations == []
    extraction.validate_evidence(_context().content)


def test_external_identity_requires_source_and_id_together() -> None:
    with pytest.raises(ValidationError, match="必须同时提供"):
        ExtractedEntity(
            local_id="e1",
            entity_type="SYSTEM",
            name="CRM",
            external_id="crm-1",
            evidence_quote="CRM",
        )


def test_uids_are_stable_and_scoped() -> None:
    entity = _extraction().entities[0]

    assert build_entity_uid(4, entity) == build_entity_uid(4, entity)
    assert build_entity_uid(4, entity) != build_entity_uid(5, entity)
    assert build_chunk_uid(_context(chunk_index=0)) != build_chunk_uid(
        _context(chunk_index=1)
    )


def test_repository_prepares_traceable_idempotent_payload() -> None:
    context = _context()
    extraction = _extraction()

    first = GraphRepository.prepare_payload(
        context=context,
        extraction=extraction,
        extractor_version="extractor-test",
    )
    second = GraphRepository.prepare_payload(
        context=context,
        extraction=extraction,
        extractor_version="extractor-test",
    )

    assert first == second
    assert first.context["chunk_uid"].startswith("chunk:v1:")
    assert first.entities[0]["identity_status"] == "PROVISIONAL"
    assert first.claims[0]["subject_uid"] == first.entities[0]["uid"]
    assert first.claims[0]["object_uid"] == first.entities[1]["uid"]
    assert first.claims[0]["uid"].startswith("claim:v1:")


def test_claim_uid_does_not_depend_on_chunk() -> None:
    extraction = _extraction()
    first = GraphRepository.prepare_payload(
        context=_context(chunk_index=0),
        extraction=extraction,
        extractor_version="extractor-test",
    )
    second = GraphRepository.prepare_payload(
        context=_context(chunk_index=1),
        extraction=extraction,
        extractor_version="extractor-test",
    )

    assert first.context["chunk_uid"] != second.context["chunk_uid"]
    assert first.claims[0]["uid"] == second.claims[0]["uid"]


def test_neo4j_schema_has_named_idempotent_unique_constraints() -> None:
    statements = load_schema_statements()
    constraints = [item for item in statements if item.startswith("CREATE CONSTRAINT")]

    assert len(constraints) == 8
    assert all("IF NOT EXISTS" in item for item in constraints)
    assert any("graph_entity_uid_unique" in item for item in constraints)
    assert any("graph_claim_uid_unique" in item for item in constraints)
    assert any("graph_evidence_uid_unique" in item for item in constraints)


def test_fixed_json_sample_is_valid_and_has_stable_expected_counts() -> None:
    sample_path = (
        Path(__file__).parents[1]
        / "zq_rag_app"
        / "schemas"
        / "graph_extraction_sample.json"
    )
    sample = GraphImportSample.model_validate_json(
        sample_path.read_text(encoding="utf-8")
    )
    sample.extraction.validate_evidence(sample.context.content)

    payload = GraphRepository.prepare_payload(
        context=sample.context,
        extraction=sample.extraction,
        extractor_version=sample.extractor_version,
    )

    assert len({row["uid"] for row in payload.entities}) == 3
    assert len({row["uid"] for row in payload.claims}) == 2
    assert len({row["mention_uid"] for row in payload.entities}) == 3
    assert len({row["evidence_uid"] for row in payload.claims}) == 2


class _FakeResult:
    def __init__(self) -> None:
        self.consumed = False

    async def consume(self) -> None:
        self.consumed = True


class _FakeTransaction:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def run(self, query: str, **parameters):
        self.calls.append((query, parameters))
        return _FakeResult()


class _FakeSession:
    def __init__(self) -> None:
        self.transaction = _FakeTransaction()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def execute_write(self, callback, *args):
        return await callback(self.transaction, *args)


@pytest.mark.asyncio
async def test_repository_writes_context_entities_and_claims_in_one_transaction() -> None:
    session = _FakeSession()
    repository = GraphRepository(session_factory=lambda: session)

    result = await repository.upsert_extraction(
        context=_context(),
        extraction=_extraction(),
        extractor_version="extractor-test",
    )

    assert result.entity_count == 2
    assert result.claim_count == 1
    assert len(session.transaction.calls) == 3
    mention_parameters = session.transaction.calls[1][1]
    claim_parameters = session.transaction.calls[2][1]
    assert mention_parameters["entities"][0]["mention_uid"].startswith(
        "mention:v1:"
    )
    assert claim_parameters["claims"][0]["evidence_uid"].startswith(
        "evidence:v1:"
    )
