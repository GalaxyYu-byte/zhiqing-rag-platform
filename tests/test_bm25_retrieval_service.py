from types import SimpleNamespace

import pytest

from zq_rag_app.services.bm25_retrieval_service import search_by_bm25


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
                score=4.25,
            )
        ]


class _FakeSession:
    def __init__(self):
        self.statement = None
        self.parameters = None

    async def execute(self, statement, parameters):
        self.statement = statement
        self.parameters = parameters
        return _FakeResult()


@pytest.mark.asyncio
async def test_bm25_retrieval_uses_pg_search_and_document_scope():
    session = _FakeSession()

    result = await search_by_bm25(
        session,
        query="  年假规定  ",
        kb_ids=[4, 4],
        doc_ids=[9, 7, 9],
        top_k=20,
    )

    sql = str(session.statement)
    assert "c.content |||" in sql
    assert "pdb.score(c.id)" in sql
    assert "c.doc_id = ANY" in sql
    assert session.parameters == {
        "query": "年假规定",
        "kb_ids": [4],
        "doc_ids": [7, 9],
        "top_k": 20,
    }
    assert result.engine == "pg_search"
    assert result.results[0].score == pytest.approx(4.25)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "kb_ids", "top_k", "doc_ids", "message"),
    [
        (" ", [4], 20, None, "query"),
        ("测试", [], 20, None, "kb_ids"),
        ("测试", [4], 0, None, "top_k"),
        ("测试", [4], 20, [], "doc_ids"),
    ],
)
async def test_bm25_retrieval_rejects_invalid_parameters(
    query,
    kb_ids,
    top_k,
    doc_ids,
    message,
):
    with pytest.raises(ValueError, match=message):
        await search_by_bm25(
            _FakeSession(),
            query=query,
            kb_ids=kb_ids,
            doc_ids=doc_ids,
            top_k=top_k,
        )
