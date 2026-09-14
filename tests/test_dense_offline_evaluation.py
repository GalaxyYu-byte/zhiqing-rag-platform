from pathlib import Path

import pytest

from zq_rag_app.evaluation.dense_offline import (
    CorpusChunk,
    DocumentSnapshot,
    GroundTruthItem,
    aggregate_metrics,
    build_blind_queries,
    embed_evaluation_questions,
    evaluate_retrieval,
    load_ground_truth,
    resolve_gold_chunk_groups,
    select_corpus,
)
from zq_rag_app.services.embedding_service import EmbeddingBatchResult
from zq_rag_app.services.retrieval_service import (
    CosineRetrievalResult,
    RetrievedChunk,
)


def _snapshot(doc_id: int, name: str, chunks: int = 2) -> DocumentSnapshot:
    return DocumentSnapshot(
        id=doc_id,
        kb_id=4,
        file_name=name,
        status="DONE",
        declared_chunk_count=chunks,
        current_chunk_count=chunks,
        token_count=100,
        version=1,
    )


def _chunk(rank: int, name: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=rank,
        doc_id=rank,
        document=name,
        chunk_index=0,
        section=None,
        page=None,
        content="内容",
        token_count=10,
        score=1.0 - rank / 100,
    )


class _FakeEmbeddingService:
    batch_size = 2

    def __init__(self):
        self.calls = []

    async def embed_many(self, texts):
        self.calls.append(list(texts))
        return EmbeddingBatchResult(
            vectors=[[float(len(text))] for text in texts],
            local_cache_hits=0,
            redis_cache_hits=0,
            generated=len(texts),
        )


def test_select_corpus_uses_latest_done_document_and_excludes_readme():
    snapshots = [
        _snapshot(1, "制度.pdf"),
        _snapshot(3, "制度.pdf"),
        _snapshot(2, "手册.docx"),
        _snapshot(4, "README.md"),
    ]

    selection = select_corpus(snapshots, ["制度.pdf", "手册.docx"])

    assert [item.id for item in selection.selected] == [3, 2]
    assert selection.missing_files == ()
    assert [item.id for item in selection.duplicate_files["制度.pdf"]] == [1, 3]
    assert {item.id for item in selection.excluded} == {1, 4}


def test_evaluate_and_aggregate_document_metrics():
    item = GroundTruthItem(
        question_id="Q1",
        question="问题",
        expected_answer="答案",
        question_type="multi_hop",
        source_files=("A.pdf", "B.pdf"),
        source_location="A 第一章 + B 第二章",
        should_answer=True,
        keywords=(),
    )
    result = CosineRetrievalResult(
        query="问题",
        embedding_model="test",
        dimensions=3,
        latency_ms=12,
        results=[
            _chunk(1, "X.pdf"),
            _chunk(2, "A.pdf"),
            _chunk(3, "A.pdf"),
            _chunk(4, "B.pdf"),
        ],
    )

    record = evaluate_retrieval(
        item,
        result,
        gold_chunk_groups={"A.pdf": (2,), "B.pdf": (4,)},
        cutoffs=(1, 3, 5),
    )
    metrics = aggregate_metrics([record], cutoffs=(1, 3, 5))

    assert record["first_relevant_rank"] == 2
    assert record["recall_at_3"] == pytest.approx(0.5)
    assert record["recall_at_5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["hit_at_1"] == 0.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["chunk_mrr"] == pytest.approx(0.5)
    assert metrics["chunk_recall_at_5"] == pytest.approx(1.0)


def test_load_ground_truth_rejects_answerable_item_without_source(tmp_path: Path):
    path = tmp_path / "ground-truth.jsonl"
    path.write_text(
        '{"question_id":"Q1","question":"问题","should_answer":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="没有 source_files"):
        load_ground_truth(path)


@pytest.mark.asyncio
async def test_embed_evaluation_questions_uses_sequential_provider_batches():
    service = _FakeEmbeddingService()

    result = await embed_evaluation_questions(service, ["a", "bb", "ccc"])

    assert service.calls == [["a", "bb"], ["ccc"]]
    assert result.vectors == [[1.0], [2.0], [3.0]]
    assert result.generated == 3


def test_blind_queries_drop_all_gold_fields():
    item = GroundTruthItem(
        question_id="Q1",
        question="问题",
        expected_answer="秘密答案",
        question_type="single_hop",
        source_files=("制度.pdf",),
        source_location="第一章",
        should_answer=True,
        keywords=("关键词",),
    )

    query = build_blind_queries([item])[0]

    assert query.question_id == "Q1"
    assert query.question == "问题"
    assert not hasattr(query, "expected_answer")
    assert not hasattr(query, "keywords")
    assert not hasattr(query, "source_files")


def test_resolve_gold_chunk_groups_uses_keywords_inside_expected_source():
    item = GroundTruthItem(
        question_id="Q1",
        question="临时权限多久？",
        expected_answer="4小时",
        question_type="single_hop",
        source_files=("制度.pdf",),
        source_location="权限章节",
        should_answer=True,
        keywords=("4小时",),
    )
    chunks = [
        CorpusChunk(1, 1, "制度.pdf", 0, "简介", None, "普通账号规则"),
        CorpusChunk(2, 1, "制度.pdf", 1, "权限", None, "默认有效期为4小时"),
        CorpusChunk(3, 2, "其他.pdf", 0, "权限", None, "默认有效期为4小时"),
    ]

    mappings, issues = resolve_gold_chunk_groups([item], chunks)

    assert mappings == {"Q1": {"制度.pdf": (2,)}}
    assert issues == []
