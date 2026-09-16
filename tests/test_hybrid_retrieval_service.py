import pytest

from zq_rag_app.api import retrieval as retrieval_api
from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult
from zq_rag_app.services.hybrid_retrieval_service import (
    fuse_dense_bm25_rrf,
    fuse_dense_bm25_graph_rrf,
)
from zq_rag_app.services.graph_retrieval_service import GraphRetrievalResult
from zq_rag_app.services.retrieval_service import (
    CosineRetrievalResult,
    RetrievedChunk,
)


def _chunk(rank: int, chunk_id: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        doc_id=chunk_id,
        document=f"{chunk_id}.pdf",
        chunk_index=0,
        section=None,
        page=None,
        content=f"chunk {chunk_id}",
        token_count=2,
        score=score,
    )


def _dense(*chunks: RetrievedChunk) -> CosineRetrievalResult:
    return CosineRetrievalResult(
        query="问题",
        embedding_model="test",
        dimensions=3,
        latency_ms=10,
        results=list(chunks),
    )


def _bm25(*chunks: RetrievedChunk) -> BM25RetrievalResult:
    return BM25RetrievalResult(
        query="问题",
        engine="pg_search",
        latency_ms=20,
        results=list(chunks),
    )


def _graph(*chunks: RetrievedChunk) -> GraphRetrievalResult:
    return GraphRetrievalResult(
        query="问题",
        engine="neo4j",
        latency_ms=15,
        results=list(chunks),
        matches=(),
    )


def test_fusion_rewards_candidates_found_by_both_retrievers():
    result = fuse_dense_bm25_rrf(
        _dense(_chunk(1, 1, 0.9), _chunk(2, 2, 0.8)),
        _bm25(_chunk(1, 3, 8.0), _chunk(2, 1, 7.0)),
        top_k=3,
        dense_weight=0.5,
        bm25_weight=0.5,
        rrf_weight=0.5,
        rrf_k=60,
    )

    assert [chunk.chunk_id for chunk in result.results] == [1, 3, 2]
    assert result.latency_ms >= 30
    assert result.candidate_scores[0].dense_rank == 1
    assert result.candidate_scores[0].bm25_rank == 2
    assert result.candidate_scores[0].final_score == pytest.approx(
        result.results[0].score
    )


def test_min_max_normalization_makes_score_scales_comparable():
    result = fuse_dense_bm25_rrf(
        _dense(_chunk(1, 1, 0.9), _chunk(2, 2, 0.1)),
        _bm25(_chunk(1, 2, 1000.0), _chunk(2, 1, 10.0)),
        top_k=2,
        rrf_weight=0.0,
    )

    assert result.candidate_scores[0].normalized_weighted_score == pytest.approx(
        0.5
    )
    assert result.candidate_scores[1].normalized_weighted_score == pytest.approx(
        0.5
    )


def test_three_way_fusion_rewards_dense_and_graph_overlap():
    result = fuse_dense_bm25_graph_rrf(
        _dense(_chunk(1, 1, 0.9), _chunk(2, 2, 0.7)),
        _bm25(_chunk(1, 3, 9.0), _chunk(2, 2, 8.0)),
        _graph(_chunk(1, 1, 0.95), _chunk(2, 4, 0.8)),
        top_k=4,
        dense_weight=0.4,
        bm25_weight=0.3,
        graph_weight=0.3,
    )

    assert result.results[0].chunk_id == 1
    assert result.engine == "dense_bm25_graph_normalized_weighted_rrf"
    assert result.latency_ms >= 45
    assert result.candidate_scores[0].dense_rank == 1
    assert result.candidate_scores[0].graph_rank == 1
    assert result.candidate_scores[0].graph_score == pytest.approx(0.95)


def test_three_way_fusion_falls_back_when_graph_has_no_match():
    dense = _dense(_chunk(1, 1, 0.9), _chunk(2, 2, 0.7))
    bm25 = _bm25(_chunk(1, 2, 9.0), _chunk(2, 3, 8.0))

    result = fuse_dense_bm25_graph_rrf(
        dense,
        bm25,
        _graph(),
        top_k=3,
    )

    expected = fuse_dense_bm25_rrf(
        dense,
        bm25,
        top_k=3,
        dense_weight=0.4,
        bm25_weight=0.3,
    )
    assert result.engine.endswith("+graph_fallback")
    assert [chunk.chunk_id for chunk in result.results] == [
        chunk.chunk_id for chunk in expected.results
    ]
    assert all(score.graph_rank is None for score in result.candidate_scores)


@pytest.mark.asyncio
async def test_hybrid_api_falls_back_when_neo4j_is_unavailable(monkeypatch):
    received_doc_ids: list[list[int] | None] = []

    async def fake_access(*_args, **_kwargs):
        return [1]

    async def fake_dense(*_args, **kwargs):
        received_doc_ids.append(kwargs.get("doc_ids"))
        return _dense(_chunk(1, 1, 0.9))

    async def fake_bm25(*_args, **kwargs):
        received_doc_ids.append(kwargs.get("doc_ids"))
        return _bm25(_chunk(1, 1, 9.0))

    async def failing_graph(*_args, **kwargs):
        received_doc_ids.append(kwargs.get("doc_ids"))
        raise ConnectionError("neo4j unavailable")

    monkeypatch.setattr(retrieval_api, "search_by_cosine", fake_dense)
    monkeypatch.setattr(retrieval_api, "search_by_bm25", fake_bm25)
    monkeypatch.setattr(retrieval_api, "search_by_graph", failing_graph)
    monkeypatch.setattr(
        retrieval_api,
        "_resolve_accessible_doc_ids",
        fake_access,
    )

    response = await retrieval_api.hybrid_graph_search(
        retrieval_api.HybridGraphSearchRequest(
            query="问题",
            kb_ids=[4],
            rerank=False,
        ),
        current_user=retrieval_api.UserContext(
            user_id=1,
            username="admin",
            department_id="ADMIN",
            role="ADMIN",
            clearance="机密",
        ),
        session=object(),
    )

    assert response.engine.endswith("+graph_fallback")
    assert [chunk.chunk_id for chunk in response.results] == [1]
    assert response.graph_claim_count == 0
    assert received_doc_ids == [[1], [1], [1]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"top_k": 2, "dense_weight": -1}, "权重"),
        ({"top_k": 2, "dense_weight": 0, "bm25_weight": 0}, "权重之和"),
        ({"top_k": 2, "rrf_weight": 1.1}, "rrf_weight"),
        ({"top_k": 2, "rrf_k": 0}, "rrf_k"),
    ],
)
def test_fusion_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fuse_dense_bm25_rrf(_dense(), _bm25(), **kwargs)
