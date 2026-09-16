"""Graph RAG 端到端答案、引用和延迟评测。"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from ..services.rag_service import RagAnswerService


_CITATION_PATTERN = re.compile(r"\[S(\d+)]")


@dataclass(slots=True, frozen=True)
class GraphRagEvalCase:
    question_id: str
    question: str
    expected_keywords: tuple[str, ...]
    expected_sources: tuple[str, ...]
    question_type: str = "single_hop"
    should_answer: bool = True


@dataclass(slots=True, frozen=True)
class GraphRagEvalRecord:
    question_id: str
    question: str
    answer: str
    source_hit: bool | None
    keyword_recall: float | None
    citations_valid: bool
    refusal_correct: bool | None
    latency_ms: int
    source_documents: tuple[str, ...]
    skipped_reason: str | None = None
    question_type: str = "single_hop"
    permission_isolated: bool | None = None


@dataclass(slots=True, frozen=True)
class GraphRagEvalReport:
    case_count: int
    evaluated_count: int
    answerable_case_count: int
    no_answer_case_count: int
    permission_case_count: int
    skipped_permission_count: int
    source_hit_rate: float
    mean_keyword_recall: float
    valid_citation_rate: float
    no_answer_accuracy: float
    permission_accuracy: float
    latency_p95_ms: int
    records: tuple[GraphRagEvalRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class GraphRagQualityThresholds:
    min_case_count: int = 30
    min_source_hit_rate: float = 0.90
    min_keyword_recall: float = 0.80
    min_valid_citation_rate: float = 1.0
    min_no_answer_accuracy: float = 0.80
    min_permission_accuracy: float = 1.0
    max_latency_p95_ms: int = 15_000
    require_permission_coverage: bool = True


@dataclass(slots=True, frozen=True)
class GraphRagQualityGate:
    passed: bool
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_quality_gate(
    report: GraphRagEvalReport,
    thresholds: GraphRagQualityThresholds = GraphRagQualityThresholds(),
) -> GraphRagQualityGate:
    failures: list[str] = []
    checks = (
        (
            report.case_count >= thresholds.min_case_count,
            f"case_count {report.case_count} < {thresholds.min_case_count}",
        ),
        (
            report.source_hit_rate >= thresholds.min_source_hit_rate,
            "source_hit_rate "
            f"{report.source_hit_rate:.3f} < {thresholds.min_source_hit_rate:.3f}",
        ),
        (
            report.mean_keyword_recall >= thresholds.min_keyword_recall,
            "mean_keyword_recall "
            f"{report.mean_keyword_recall:.3f} < {thresholds.min_keyword_recall:.3f}",
        ),
        (
            report.valid_citation_rate >= thresholds.min_valid_citation_rate,
            "valid_citation_rate "
            f"{report.valid_citation_rate:.3f} < "
            f"{thresholds.min_valid_citation_rate:.3f}",
        ),
        (
            report.no_answer_case_count == 0
            or report.no_answer_accuracy >= thresholds.min_no_answer_accuracy,
            "no_answer_accuracy "
            f"{report.no_answer_accuracy:.3f} < "
            f"{thresholds.min_no_answer_accuracy:.3f}",
        ),
        (
            not thresholds.require_permission_coverage
            or report.permission_case_count == 0
            or report.permission_accuracy >= thresholds.min_permission_accuracy,
            "permission_accuracy "
            f"{report.permission_accuracy:.3f} < "
            f"{thresholds.min_permission_accuracy:.3f}",
        ),
        (
            report.latency_p95_ms <= thresholds.max_latency_p95_ms,
            f"latency_p95_ms {report.latency_p95_ms} > "
            f"{thresholds.max_latency_p95_ms}",
        ),
        (
            not thresholds.require_permission_coverage
            or report.skipped_permission_count == 0,
            f"skipped_permission_count {report.skipped_permission_count} > 0",
        ),
    )
    failures.extend(message for passed, message in checks if not passed)
    return GraphRagQualityGate(passed=not failures, failures=tuple(failures))


def load_graph_rag_cases(path: str | Path) -> tuple[GraphRagEvalCase, ...]:
    cases: list[GraphRagEvalCase] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        case = GraphRagEvalCase(
            question_id=str(payload["question_id"]).strip(),
            question=str(payload["question"]).strip(),
            expected_keywords=tuple(
                str(value).strip()
                for value in payload.get(
                    "expected_keywords", payload.get("keywords", [])
                )
                if str(value).strip()
            ),
            expected_sources=tuple(
                str(value).strip()
                for value in payload.get(
                    "expected_sources", payload.get("source_files", [])
                )
                if str(value).strip()
            ),
            question_type=str(payload.get("question_type", "single_hop")).strip(),
            should_answer=bool(payload.get("should_answer", True)),
        )
        if not case.question_id or not case.question:
            raise ValueError(f"第 {line_number} 行缺少 question_id 或 question")
        if not case.expected_keywords:
            raise ValueError(f"第 {line_number} 行缺少期望关键词")
        if case.should_answer and not case.expected_sources:
            raise ValueError(f"第 {line_number} 行可回答题缺少期望来源")
        cases.append(case)
    if not cases:
        raise ValueError("评测集不能为空")
    return tuple(cases)


async def evaluate_graph_rag(
    session: AsyncSession,
    *,
    service: RagAnswerService,
    cases: tuple[GraphRagEvalCase, ...],
    kb_id: int,
    candidate_k: int = 30,
    top_k: int = 5,
    rerank: bool = True,
    permission_checker: Callable[
        [GraphRagEvalCase], Awaitable[bool]
    ] | None = None,
) -> GraphRagEvalReport:
    records: list[GraphRagEvalRecord] = []
    for case in cases:
        if case.question_type == "permission":
            if permission_checker is not None:
                started_at = time.perf_counter()
                isolated = await permission_checker(case)
                records.append(
                    GraphRagEvalRecord(
                        question_id=case.question_id,
                        question=case.question,
                        answer="访问已被检索前 ACL 阻止" if isolated else "权限隔离失败",
                        source_hit=None,
                        keyword_recall=None,
                        citations_valid=True,
                        refusal_correct=None,
                        latency_ms=max(
                            0, round((time.perf_counter() - started_at) * 1000)
                        ),
                        source_documents=(),
                        question_type=case.question_type,
                        permission_isolated=isolated,
                    )
                )
                continue
            records.append(
                GraphRagEvalRecord(
                    question_id=case.question_id,
                    question=case.question,
                    answer="",
                    source_hit=None,
                    keyword_recall=None,
                    citations_valid=False,
                    refusal_correct=None,
                    latency_ms=0,
                    source_documents=(),
                    skipped_reason="需要使用受限用户上下文执行文档级 ACL 验证",
                    question_type=case.question_type,
                )
            )
            continue
        result = await service.answer(
            session,
            query=case.question,
            kb_ids=[kb_id],
            candidate_k=candidate_k,
            top_k=top_k,
            rerank=rerank,
        )
        source_documents = tuple(source.document for source in result.sources)
        source_hit: bool | None = None
        keyword_recall: float | None = None
        refusal_correct: bool | None = None
        if case.should_answer:
            expected_sources = {name.casefold() for name in case.expected_sources}
            source_hit = any(
                document.casefold() in expected_sources
                for document in source_documents
            )
            keyword_recall = sum(
                keyword.casefold() in result.answer.casefold()
                for keyword in case.expected_keywords
            ) / len(case.expected_keywords)
        else:
            refusal_correct = any(
                marker in result.answer
                for marker in ("无法", "未提供", "没有提供", "不能确定", "不应提供")
            )
        citations = {
            int(value) for value in _CITATION_PATTERN.findall(result.answer)
        }
        citations_valid = bool(citations) and citations.issubset(
            set(range(1, len(result.sources) + 1))
        )
        records.append(
            GraphRagEvalRecord(
                question_id=case.question_id,
                question=case.question,
                answer=result.answer,
                source_hit=source_hit,
                keyword_recall=keyword_recall,
                citations_valid=citations_valid,
                refusal_correct=refusal_correct,
                latency_ms=result.timing.total_ms,
                source_documents=source_documents,
                question_type=case.question_type,
            )
        )

    evaluated = [record for record in records if record.skipped_reason is None]
    answerable = [record for record in evaluated if record.source_hit is not None]
    no_answer = [
        record for record in evaluated if record.question_type == "no_answer"
    ]
    permission = [
        record for record in evaluated if record.question_type == "permission"
    ]
    latencies = sorted(record.latency_ms for record in evaluated)
    p95_index = max(0, math.ceil(len(latencies) * 0.95) - 1)
    return GraphRagEvalReport(
        case_count=len(records),
        evaluated_count=len(evaluated),
        answerable_case_count=len(answerable),
        no_answer_case_count=len(no_answer),
        permission_case_count=sum(
            case.question_type == "permission" for case in cases
        ),
        skipped_permission_count=sum(
            record.skipped_reason is not None for record in records
        ),
        source_hit_rate=(
            sum(bool(record.source_hit) for record in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        mean_keyword_recall=(
            sum(float(record.keyword_recall) for record in answerable)
            / len(answerable)
            if answerable
            else 0.0
        ),
        valid_citation_rate=(
            sum(record.citations_valid for record in answerable) / len(answerable)
            if answerable
            else 0.0
        ),
        no_answer_accuracy=(
            sum(bool(record.refusal_correct) for record in no_answer)
            / len(no_answer)
            if no_answer
            else 0.0
        ),
        permission_accuracy=(
            sum(bool(record.permission_isolated) for record in permission)
            / len(permission)
            if permission
            else 0.0
        ),
        latency_p95_ms=latencies[p95_index] if latencies else 0,
        records=tuple(records),
    )
