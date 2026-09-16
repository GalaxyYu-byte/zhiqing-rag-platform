from pathlib import Path

import pytest

from zq_rag_app.evaluation.graph_rag import (
    GraphRagEvalReport,
    GraphRagQualityThresholds,
    evaluate_graph_rag,
    evaluate_quality_gate,
    load_graph_rag_cases,
)
from zq_rag_app.services.rag_service import (
    RagAnswerResult,
    RagSource,
    RagTiming,
)


class _Service:
    async def answer(self, *_args, **kwargs):
        question = kwargs["query"]
        is_first = "延期" in question
        return RagAnswerResult(
            query=question,
            answer=(
                "调整到 2026-09-28，原因是权限回归。[S1]"
                if is_first
                else "当前 MRR 是 0.79。[S1]"
            ),
            model="test",
            engine="hybrid",
            token_count=10,
            graph_degraded=False,
            reranked=True,
            sources=(
                RagSource(
                    citation_id="S1",
                    rank=1,
                    chunk_id=1,
                    doc_id=1,
                    document="周会.txt",
                    chunk_index=0,
                    section=None,
                    page=None,
                    excerpt="证据",
                    score=0.9,
                    dense_rank=1,
                    bm25_rank=1,
                    graph_rank=1,
                ),
            ),
            timing=RagTiming(
                retrieval_ms=10,
                generation_ms=10,
                total_ms=100 if is_first else 200,
            ),
        )


def test_load_graph_rag_cases_rejects_incomplete_rows(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"question_id":"Q1","question":"问题","expected_keywords":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="缺少期望关键词"):
        load_graph_rag_cases(path)


@pytest.mark.asyncio
async def test_graph_rag_evaluation_reports_quality_and_p95(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"question_id":"Q1","question":"为什么延期？",'
                '"expected_keywords":["2026-09-28","权限回归"],'
                '"expected_sources":["周会.txt"]}',
                '{"question_id":"Q2","question":"MRR是多少？",'
                '"expected_keywords":["0.79","Recall"],'
                '"expected_sources":["周会.txt"]}',
            )
        ),
        encoding="utf-8",
    )

    report = await evaluate_graph_rag(
        object(),
        service=_Service(),
        cases=load_graph_rag_cases(path),
        kb_id=4,
        rerank=False,
    )

    assert report.case_count == 2
    assert report.evaluated_count == 2
    assert report.source_hit_rate == 1.0
    assert report.mean_keyword_recall == 0.75
    assert report.valid_citation_rate == 1.0
    assert report.latency_p95_ms == 200


def test_full_ground_truth_loads_all_categories():
    cases = load_graph_rag_cases("qa-docs/qa_ground_truth.jsonl")

    assert len(cases) == 50
    assert sum(case.question_type == "permission" for case in cases) == 5
    assert sum(not case.should_answer for case in cases) == 10


def test_quality_gate_reports_actionable_failures():
    report = GraphRagEvalReport(
        case_count=50,
        evaluated_count=45,
        answerable_case_count=40,
        no_answer_case_count=5,
        permission_case_count=5,
        skipped_permission_count=5,
        source_hit_rate=0.95,
        mean_keyword_recall=0.90,
        valid_citation_rate=1.0,
        no_answer_accuracy=1.0,
        permission_accuracy=0.0,
        latency_p95_ms=10_000,
        records=(),
    )

    gate = evaluate_quality_gate(report)
    development_gate = evaluate_quality_gate(
        report,
        GraphRagQualityThresholds(require_permission_coverage=False),
    )

    assert gate.passed is False
    assert gate.failures == (
        "permission_accuracy 0.000 < 1.000",
        "skipped_permission_count 5 > 0",
    )
    assert development_gate.passed is True
