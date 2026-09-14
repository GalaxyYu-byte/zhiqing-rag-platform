import pytest

from zq_rag_app.services.bm25_retrieval_service import BM25RetrievalResult
from zq_rag_app.services.hybrid_retrieval_service import (
    fuse_dense_bm25_rrf,
)
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
