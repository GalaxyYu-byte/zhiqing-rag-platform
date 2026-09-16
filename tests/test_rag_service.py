from types import SimpleNamespace

import pytest

from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult
from zq_rag_app.services.graph_retrieval_service import GraphRetrievalResult
from zq_rag_app.services.rag_service import RagAnswerService
from zq_rag_app.services.retrieval_service import (
    CosineRetrievalResult,
    RetrievedChunk,
)


def _chunk(rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=2,
        doc_id=1,
        document="2026年9月AI平台周会纪要.txt",
        chunk_index=1,
        section="最新决定",
        page=None,
        content=(
            "Aurora-KB 正式上线计划调整到 2026-09-28。"
            "原因：预留权限回归、故障演练和业务验收时间。"
        ),
        token_count=35,
        score=0.9,
    )


class _Completions:
    def __init__(self):
        self.calls = 0
        self.kwargs = None

    async def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="延期是为了完成权限回归、故障演练和业务验收。[S1]"
                    )
                )
            ],
            usage=SimpleNamespace(total_tokens=88),
        )


class _Client:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_Completions())


def _searches(*, graph_fails: bool = False, empty: bool = False):
    chunks = [] if empty else [_chunk()]

    async def dense(*_args, **kwargs):
        return CosineRetrievalResult(
            query=kwargs["query"],
            embedding_model="test",
            dimensions=3,
            latency_ms=10,
            results=chunks,
        )

    async def bm25(*_args, **kwargs):
        return BM25RetrievalResult(
            query=kwargs["query"],
            engine="pg_search",
            latency_ms=5,
            results=chunks,
        )

    async def graph(*_args, **kwargs):
        if graph_fails:
            raise ConnectionError("neo4j unavailable")
        return GraphRetrievalResult(
            query=kwargs["query"],
            engine="neo4j_active_claims",
            latency_ms=8,
            results=chunks,
            matches=(),
        )

    return dense, bm25, graph


@pytest.mark.asyncio
async def test_rag_answer_uses_hybrid_context_and_returns_sources():
    client = _Client()
    dense, bm25, graph = _searches()
    service = RagAnswerService(
        client=client,
        dense_search=dense,
        bm25_search=bm25,
        graph_search=graph,
    )

    result = await service.answer(
        object(),
        query="Aurora-KB 为什么延期？",
        kb_ids=[4],
        top_k=1,
        candidate_k=3,
        rerank=False,
    )

    assert result.answer.endswith("[S1]")
    assert result.token_count == 88
    assert result.sources[0].citation_id == "S1"
    assert result.sources[0].graph_rank == 1
    assert result.graph_degraded is False
    assert "[S1] 文档：2026年9月AI平台周会纪要.txt" in (
        client.chat.completions.kwargs["messages"][1]["content"]
    )


@pytest.mark.asyncio
async def test_rag_answer_degrades_when_graph_is_unavailable():
    client = _Client()
    dense, bm25, graph = _searches(graph_fails=True)
    service = RagAnswerService(
        client=client,
        dense_search=dense,
        bm25_search=bm25,
        graph_search=graph,
    )

    result = await service.answer(
        object(),
        query="Aurora-KB 为什么延期？",
        kb_ids=[4],
        top_k=1,
        candidate_k=3,
        rerank=False,
    )

    assert result.graph_degraded is True
    assert result.engine.endswith("+graph_fallback")
    assert result.sources[0].graph_rank is None


@pytest.mark.asyncio
async def test_rag_answer_skips_llm_when_no_evidence_is_found():
    client = _Client()
    dense, bm25, graph = _searches(empty=True)
    service = RagAnswerService(
        client=client,
        dense_search=dense,
        bm25_search=bm25,
        graph_search=graph,
    )

    result = await service.answer(
        object(),
        query="不存在的信息",
        kb_ids=[4],
        top_k=1,
        candidate_k=3,
        rerank=False,
    )

    assert result.sources == ()
    assert result.token_count == 0
    assert "无法回答" in result.answer
    assert client.chat.completions.calls == 0


def test_rag_answer_deduplicates_identical_chunks_from_duplicate_documents():
    first = _chunk()
    duplicate = RetrievedChunk(
        rank=2,
        chunk_id=16,
        doc_id=8,
        document=first.document,
        chunk_index=first.chunk_index,
        section=first.section,
        page=first.page,
        content=first.content,
        token_count=first.token_count,
        score=0.8,
    )

    unique = RagAnswerService._deduplicate_chunks([first, duplicate])

    assert [chunk.chunk_id for chunk in unique] == [2]
