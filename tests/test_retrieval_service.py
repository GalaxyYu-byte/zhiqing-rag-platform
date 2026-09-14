from types import SimpleNamespace

import pytest

from zq_rag_app.services.retrieval_service import (
    search_by_cosine,
    search_by_cosine_vector,
)


class _FakeEmbeddingService:
    model = "test-embedding"
    dimensions = 3

    async def embed_many(self, texts):
        assert texts == ["年假有几天？"]
        return SimpleNamespace(vectors=[[0.1, 0.2, 0.3]])


class _FakeResult:
    def all(self):
        return [
            SimpleNamespace(
                chunk_id=11,
                doc_id=7,
                document="员工手册.pdf",
                chunk_index=2,
                section="休假管理",
                page=18,
                content="每年享受十天带薪年假。",
                token_count=12,
                score=0.9234,
            ),
            SimpleNamespace(
                chunk_id=12,
                doc_id=7,
                document="员工手册.pdf",
                chunk_index=3,
                section="请假流程",
                page=19,
                content="年假需要提前申请。",
                token_count=9,
                score=0.8123,
            ),
        ]


class _FakeSession:
    def __init__(self):
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _FakeResult()


@pytest.mark.asyncio
async def test_cosine_retrieval_returns_ranked_chunks():
    session = _FakeSession()

    result = await search_by_cosine(
        session,
        query="  年假有几天？  ",
        kb_ids=[2, 1, 2],
        top_k=5,
        min_score=0.5,
        embedding_service=_FakeEmbeddingService(),
    )

    assert result.query == "年假有几天？"
    assert result.embedding_model == "test-embedding"
    assert result.dimensions == 3
    assert [item.rank for item in result.results] == [1, 2]
    assert result.results[0].score == pytest.approx(0.9234)
    assert "<=>" in str(session.statement)
    assert "kb_document.status" in str(session.statement)


@pytest.mark.asyncio
async def test_cosine_retrieval_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="query"):
        await search_by_cosine(
            _FakeSession(),
            query=" ",
            kb_ids=[1],
            top_k=5,
            min_score=0.5,
            embedding_service=_FakeEmbeddingService(),
        )

    with pytest.raises(ValueError, match="top_k"):
        await search_by_cosine(
            _FakeSession(),
            query="测试",
            kb_ids=[1],
            top_k=0,
            min_score=0.5,
            embedding_service=_FakeEmbeddingService(),
        )


@pytest.mark.asyncio
async def test_cosine_vector_retrieval_can_limit_documents():
    session = _FakeSession()

    await search_by_cosine_vector(
        session,
        query="年假有几天？",
        query_vector=[0.1, 0.2, 0.3],
        kb_ids=[1],
        doc_ids=[9, 7, 9],
        top_k=5,
        min_score=0.0,
        embedding_model="test-embedding",
        dimensions=3,
    )

    assert "kb_doc_chunk.doc_id" in str(session.statement)


@pytest.mark.asyncio
async def test_cosine_vector_retrieval_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="维度"):
        await search_by_cosine_vector(
            _FakeSession(),
            query="测试",
            query_vector=[0.1, 0.2],
            kb_ids=[1],
            top_k=5,
            min_score=0.0,
            embedding_model="test-embedding",
            dimensions=3,
        )
