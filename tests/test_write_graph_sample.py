from pathlib import Path

import pytest

from scripts.write_graph_sample import write_and_verify
from zq_rag_app.graph.repository import GraphChunkSummary, GraphWriteResult


class _IdempotentRepository:
    def __init__(self) -> None:
        self.schema_initialized = False
        self.write_count = 0
        self.chunk_uid = "chunk:v1:fixture"

    async def ensure_schema(self) -> None:
        self.schema_initialized = True

    async def upsert_extraction(self, **_) -> GraphWriteResult:
        self.write_count += 1
        return GraphWriteResult(
            chunk_uid=self.chunk_uid,
            entity_count=3,
            claim_count=2,
        )

    async def get_chunk_summary(self, chunk_uid: str) -> GraphChunkSummary:
        assert chunk_uid == self.chunk_uid
        return GraphChunkSummary(
            chunk_uid=chunk_uid,
            entity_count=3,
            mention_count=3,
            claim_count=2,
            evidence_count=2,
        )


@pytest.mark.asyncio
async def test_fixed_sample_is_written_twice_and_verified(capsys) -> None:
    repository = _IdempotentRepository()
    input_path = (
        Path(__file__).parents[1]
        / "zq_rag_app"
        / "schemas"
        / "graph_extraction_sample.json"
    )

    result = await write_and_verify(
        input_path,
        2,
        repository=repository,
    )

    assert repository.schema_initialized is True
    assert repository.write_count == 2
    assert result == {
        "status": "idempotent",
        "repeat": 2,
        "chunk_uid": "chunk:v1:fixture",
        "entity_count": 3,
        "mention_count": 3,
        "claim_count": 2,
        "evidence_count": 2,
    }
    assert len(capsys.readouterr().out.splitlines()) == 2
