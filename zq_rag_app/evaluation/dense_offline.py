"""Dense、BM25 与混合检索的离线评估命令实现。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import SessionLocal, close_database
from ..core.redis import close_redis
from ..models.document import DocChunk, Document
from ..services.bm25_retrieval_service import (
    BM25RetrievalResult,
    search_by_bm25,
)
from ..services.embedding_service import (
    EmbeddingBatchResult,
    EmbeddingService,
    get_embedding_service,
)
from ..services.hybrid_retrieval_service import (
    HybridRetrievalResult,
    fuse_dense_bm25_rrf,
)
from ..services.reranker_service import RerankerResult, RerankerService
from ..services.retrieval_service import (
    CosineRetrievalResult,
    search_by_cosine_vector,
)


DEFAULT_CUTOFFS = (1, 3, 5, 10, 20)
RetrievalResult = (
    CosineRetrievalResult
    | BM25RetrievalResult
    | HybridRetrievalResult
    | RerankerResult
)


@dataclass(slots=True, frozen=True)
class GroundTruthItem:
    question_id: str
    question: str
    expected_answer: str
    question_type: str
    source_files: tuple[str, ...]
    source_location: str
    should_answer: bool
    keywords: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class BlindQuery:
    """检索阶段唯一允许接触的评估输入。"""

    question_id: str
    question: str


@dataclass(slots=True, frozen=True)
class DocumentSnapshot:
    id: int
    kb_id: int
    file_name: str
    status: str
    declared_chunk_count: int
    current_chunk_count: int
    token_count: int
    version: int


@dataclass(slots=True, frozen=True)
class CorpusChunk:
    id: int
    doc_id: int
    document: str
    chunk_index: int
    section: str | None
    page: int | None
    content: str


@dataclass(slots=True, frozen=True)
class CorpusSelection:
    selected: tuple[DocumentSnapshot, ...]
    missing_files: tuple[str, ...]
    duplicate_files: dict[str, tuple[DocumentSnapshot, ...]]
    excluded: tuple[DocumentSnapshot, ...]


def _normalized_file_name(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_file_names(path: Path) -> list[str]:
    """读取语料清单中的业务文档文件名并保持清单顺序。"""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "file_name" not in rows[0]:
        raise ValueError(f"manifest 缺少 file_name 列或没有数据: {path}")
    names = [str(row["file_name"]).strip() for row in rows]
    if any(not name for name in names):
        raise ValueError(f"manifest 存在空 file_name: {path}")
    if len({_normalized_file_name(name) for name in names}) != len(names):
        raise ValueError(f"manifest 存在重复 file_name: {path}")
    return names


def load_ground_truth(path: Path) -> list[GroundTruthItem]:
    """读取并校验 JSONL Golden Dataset。"""

    items: list[GroundTruthItem] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Golden Dataset 第 {line_number} 行不是合法 JSON"
                ) from exc
            question_id = str(payload.get("question_id", "")).strip()
            question = str(payload.get("question", "")).strip()
            if not question_id or not question:
                raise ValueError(
                    f"Golden Dataset 第 {line_number} 行缺少 question_id/question"
                )
            if question_id in seen_ids:
                raise ValueError(f"Golden Dataset question_id 重复: {question_id}")
            seen_ids.add(question_id)
            source_files = tuple(
                str(value).strip()
                for value in payload.get("source_files", [])
                if str(value).strip()
            )
            should_answer = bool(payload.get("should_answer", True))
            if should_answer and not source_files:
                raise ValueError(f"{question_id} 应回答但没有 source_files")
            items.append(
                GroundTruthItem(
                    question_id=question_id,
                    question=question,
                    expected_answer=str(payload.get("expected_answer", "")),
                    question_type=str(
                        payload.get("question_type", "unknown")
                    ).strip()
                    or "unknown",
                    source_files=source_files,
                    source_location=str(payload.get("source_location", "")).strip(),
                    should_answer=should_answer,
                    keywords=tuple(
                        str(value).strip()
                        for value in payload.get("keywords", [])
                        if str(value).strip()
                    ),
                )
            )
    if not items:
        raise ValueError(f"Golden Dataset 没有可评估问题: {path}")
    return items


def build_blind_queries(items: Sequence[GroundTruthItem]) -> list[BlindQuery]:
    """丢弃答案和标签字段，只保留实际允许进入检索链路的内容。"""

    return [
        BlindQuery(question_id=item.question_id, question=item.question)
        for item in items
    ]


async def embed_evaluation_questions(
    service: EmbeddingService,
    questions: Sequence[str],
) -> EmbeddingBatchResult:
    """按供应商单批限制顺序向量化，避免缓存锁释放挤满 Redis 连接池。"""

    vectors: list[list[float]] = []
    local_cache_hits = 0
    redis_cache_hits = 0
    generated = 0
    for start in range(0, len(questions), service.batch_size):
        batch = await service.embed_many(
            questions[start : start + service.batch_size]
        )
        vectors.extend(batch.vectors)
        local_cache_hits += batch.local_cache_hits
        redis_cache_hits += batch.redis_cache_hits
        generated += batch.generated
    return EmbeddingBatchResult(
        vectors=vectors,
        local_cache_hits=local_cache_hits,
        redis_cache_hits=redis_cache_hits,
        generated=generated,
    )


async def load_document_snapshots(
    session: AsyncSession,
    *,
    kb_ids: Sequence[int] | None = None,
) -> list[DocumentSnapshot]:
    """读取未软删除文档及其当前版本的实际 Chunk 数。"""

    statement = (
        select(
            Document.id,
            Document.kb_id,
            Document.file_name,
            Document.status,
            Document.chunk_count,
            func.count(DocChunk.id).label("current_chunk_count"),
            Document.token_count,
            Document.version,
        )
        .outerjoin(
            DocChunk,
            and_(
                DocChunk.doc_id == Document.id,
                DocChunk.doc_version == Document.version,
            ),
        )
        .where(Document.is_deleted.is_(False))
        .group_by(
            Document.id,
            Document.kb_id,
            Document.file_name,
            Document.status,
            Document.chunk_count,
            Document.token_count,
            Document.version,
        )
        .order_by(Document.kb_id.asc(), Document.id.asc())
    )
    if kb_ids:
        statement = statement.where(Document.kb_id.in_(sorted(set(kb_ids))))
    rows = (await session.execute(statement)).all()
    return [
        DocumentSnapshot(
            id=int(row.id),
            kb_id=int(row.kb_id),
            file_name=str(row.file_name),
            status=str(row.status),
            declared_chunk_count=int(row.chunk_count),
            current_chunk_count=int(row.current_chunk_count),
            token_count=int(row.token_count),
            version=int(row.version),
        )
        for row in rows
    ]


async def load_corpus_chunks(
    session: AsyncSession,
    *,
    doc_ids: Sequence[int],
) -> list[CorpusChunk]:
    """读取已选文档的当前版本 Chunk，用于建立独立 Gold 标签。"""

    if not doc_ids:
        return []
    statement = (
        select(
            DocChunk.id,
            DocChunk.doc_id,
            Document.file_name.label("document"),
            DocChunk.chunk_index,
            DocChunk.section_title.label("section"),
            DocChunk.page_num.label("page"),
            DocChunk.content,
        )
        .join(Document, Document.id == DocChunk.doc_id)
        .where(
            DocChunk.doc_id.in_(sorted(set(doc_ids))),
            DocChunk.doc_version == Document.version,
            Document.status == "DONE",
            Document.is_deleted.is_(False),
        )
        .order_by(DocChunk.doc_id.asc(), DocChunk.chunk_index.asc())
    )
    rows = (await session.execute(statement)).all()
    return [
        CorpusChunk(
            id=int(row.id),
            doc_id=int(row.doc_id),
            document=str(row.document),
            chunk_index=int(row.chunk_index),
            section=row.section,
            page=row.page,
            content=str(row.content),
        )
        for row in rows
    ]


def _normalized_evidence_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", "", normalized)


def _location_terms(item: GroundTruthItem, source_file: str) -> list[str]:
    location = item.source_location.replace(source_file, "")
    parts = re.split(r"[+，,；;：:/（）()【】\[\]=“”\"']+", location)
    terms: list[str] = []
    for part in parts:
        normalized = _normalized_evidence_text(part.strip("-—. "))
        if len(normalized) >= 3 and normalized not in {"sheet", "全库无来源"}:
            terms.append(normalized)
    return terms


def resolve_gold_chunk_groups(
    items: Sequence[GroundTruthItem],
    chunks: Sequence[CorpusChunk],
) -> tuple[dict[str, dict[str, tuple[int, ...]]], list[dict[str, str]]]:
    """将 Gold 来源映射为证据 Chunk 组。

    每个来源文件是一个证据组；组内允许多个包含同一证据的重叠 Chunk，召回
    任意一个即视为该来源证据命中。关键词只参与离线标签解析，不进入检索。
    """

    chunks_by_file: dict[str, list[CorpusChunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_file[_normalized_file_name(chunk.document)].append(chunk)

    mappings: dict[str, dict[str, tuple[int, ...]]] = {}
    issues: list[dict[str, str]] = []
    for item in items:
        if not item.should_answer:
            mappings[item.question_id] = {}
            continue
        source_groups: dict[str, tuple[int, ...]] = {}
        normalized_keywords = [
            _normalized_evidence_text(keyword)
            for keyword in item.keywords
            if _normalized_evidence_text(keyword)
        ]
        for source_file in item.source_files:
            candidates = chunks_by_file.get(_normalized_file_name(source_file), [])
            location_terms = _location_terms(item, source_file)
            scored: list[tuple[tuple[int, int], CorpusChunk]] = []
            for chunk in candidates:
                haystack = _normalized_evidence_text(
                    f"{chunk.section or ''}\n{chunk.content}"
                )
                keyword_score = sum(
                    keyword in haystack for keyword in normalized_keywords
                )
                location_score = sum(term in haystack for term in location_terms)
                scored.append(((keyword_score, location_score), chunk))
            best_score = max((score for score, _ in scored), default=(0, 0))
            matched = [
                chunk
                for score, chunk in scored
                if score == best_score and score != (0, 0)
            ]
            if not matched:
                issues.append(
                    {
                        "question_id": item.question_id,
                        "source_file": source_file,
                        "reason": "关键词和 source_location 均无法映射到当前 Chunk",
                    }
                )
                continue
            source_groups[source_file] = tuple(sorted(chunk.id for chunk in matched))
        mappings[item.question_id] = source_groups
    return mappings, issues


def select_corpus(
    snapshots: Sequence[DocumentSnapshot],
    manifest_file_names: Sequence[str],
) -> CorpusSelection:
    """按 manifest 选择每个文件最新的成功入库记录。"""

    by_name: dict[str, list[DocumentSnapshot]] = defaultdict(list)
    for snapshot in snapshots:
        by_name[_normalized_file_name(snapshot.file_name)].append(snapshot)

    selected: list[DocumentSnapshot] = []
    missing: list[str] = []
    duplicates: dict[str, tuple[DocumentSnapshot, ...]] = {}
    selected_ids: set[int] = set()
    for file_name in manifest_file_names:
        matches = by_name.get(_normalized_file_name(file_name), [])
        done_matches = [
            item
            for item in matches
            if item.status == "DONE" and item.current_chunk_count > 0
        ]
        if not done_matches:
            missing.append(file_name)
            continue
        chosen = max(done_matches, key=lambda item: item.id)
        selected.append(chosen)
        selected_ids.add(chosen.id)
        if len(matches) > 1:
            duplicates[file_name] = tuple(
                sorted(matches, key=lambda item: item.id)
            )

    return CorpusSelection(
        selected=tuple(selected),
        missing_files=tuple(missing),
        duplicate_files=duplicates,
        excluded=tuple(item for item in snapshots if item.id not in selected_ids),
    )


def evaluate_retrieval(
    item: GroundTruthItem,
    result: RetrievalResult,
    *,
    gold_chunk_groups: dict[str, tuple[int, ...]] | None = None,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """在检索完成后计算文档级和 Chunk 级指标。"""

    fusion_by_chunk = {
        candidate.chunk_id: asdict(candidate)
        for candidate in getattr(result, "candidate_scores", ())
    }
    retrieved = []
    for chunk in result.results:
        payload = {
            "rank": chunk.rank,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "document": chunk.document,
            "chunk_index": chunk.chunk_index,
            "section": chunk.section,
            "page": chunk.page,
            "score": chunk.score,
            "content": chunk.content,
        }
        if chunk.chunk_id in fusion_by_chunk:
            payload["fusion"] = fusion_by_chunk[chunk.chunk_id]
        retrieved.append(payload)
    record: dict[str, Any] = {
        "question_id": item.question_id,
        "question": item.question,
        "question_type": item.question_type,
        "should_answer": item.should_answer,
        "expected_answer": item.expected_answer,
        "expected_source_files": list(item.source_files),
        "source_location": item.source_location,
        "keywords": list(item.keywords),
        "latency_ms": result.latency_ms,
        "top_score": retrieved[0]["score"] if retrieved else None,
        "retrieved": retrieved,
        "document_evaluable": item.should_answer and bool(item.source_files),
    }
    record["evaluable"] = record["document_evaluable"]
    record["chunk_evaluable"] = bool(
        record["document_evaluable"]
        and gold_chunk_groups
        and len(gold_chunk_groups) == len(set(item.source_files))
    )
    record["gold_chunk_groups"] = {
        source_file: list(chunk_ids)
        for source_file, chunk_ids in (gold_chunk_groups or {}).items()
    }
    if not record["document_evaluable"]:
        record["document_first_relevant_rank"] = None
        record["document_miss"] = False
        record["chunk_first_relevant_rank"] = None
        record["chunk_miss"] = False
        # 兼容第一版逐题结果字段。
        record["first_relevant_rank"] = None
        record["miss"] = False
        return record

    expected = {_normalized_file_name(name) for name in item.source_files}
    document_first_ranks: dict[str, int] = {}
    for chunk in result.results:
        normalized = _normalized_file_name(chunk.document)
        if normalized in expected and normalized not in document_first_ranks:
            document_first_ranks[normalized] = chunk.rank
    _add_rank_metrics(
        record,
        prefix="document",
        first_ranks=document_first_ranks.values(),
        expected_group_count=len(expected),
        cutoffs=cutoffs,
    )

    if record["chunk_evaluable"]:
        chunk_first_ranks: dict[str, int] = {}
        for source_file, chunk_ids in (gold_chunk_groups or {}).items():
            expected_chunk_ids = set(chunk_ids)
            rank = next(
                (
                    chunk.rank
                    for chunk in result.results
                    if chunk.chunk_id in expected_chunk_ids
                ),
                None,
            )
            if rank is not None:
                chunk_first_ranks[source_file] = rank
        _add_rank_metrics(
            record,
            prefix="chunk",
            first_ranks=chunk_first_ranks.values(),
            expected_group_count=len(gold_chunk_groups or {}),
            cutoffs=cutoffs,
        )
    else:
        record["chunk_first_relevant_rank"] = None
        record["chunk_reciprocal_rank"] = 0.0
        record["chunk_miss"] = False

    # 兼容第一版字段；无前缀指标始终代表文档级。
    record["first_relevant_rank"] = record["document_first_relevant_rank"]
    record["reciprocal_rank"] = record["document_reciprocal_rank"]
    record["miss"] = record["document_miss"]
    for cutoff in cutoffs:
        for metric in ("hit", "recall", "ndcg"):
            record[f"{metric}_at_{cutoff}"] = record[
                f"document_{metric}_at_{cutoff}"
            ]
    return record


def _add_rank_metrics(
    record: dict[str, Any],
    *,
    prefix: str,
    first_ranks: Iterable[int],
    expected_group_count: int,
    cutoffs: Sequence[int],
) -> None:
    ranks = list(first_ranks)
    first_relevant_rank = min(ranks, default=None)
    record[f"{prefix}_first_relevant_rank"] = first_relevant_rank
    record[f"{prefix}_reciprocal_rank"] = (
        1.0 / first_relevant_rank if first_relevant_rank else 0.0
    )
    record[f"{prefix}_miss"] = first_relevant_rank is None
    for cutoff in cutoffs:
        found = sum(rank <= cutoff for rank in ranks)
        record[f"{prefix}_hit_at_{cutoff}"] = float(found > 0)
        record[f"{prefix}_recall_at_{cutoff}"] = found / expected_group_count
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in ranks
            if rank <= cutoff
        )
        ideal_count = min(expected_group_count, cutoff)
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        record[f"{prefix}_ndcg_at_{cutoff}"] = dcg / idcg if idcg else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def aggregate_metrics(
    records: Iterable[dict[str, Any]],
    *,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> dict[str, Any]:
    """分别聚合文档级和 Chunk 级宏平均检索指标。"""

    all_records = list(records)
    latencies = [float(record["latency_ms"]) for record in all_records]
    metrics: dict[str, Any] = {
        "question_count": len(all_records),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
    }
    for level in ("document", "chunk"):
        evaluable = [
            record for record in all_records if record[f"{level}_evaluable"]
        ]
        metrics[f"{level}_evaluable_question_count"] = len(evaluable)
        metrics[f"{level}_excluded_question_count"] = (
            len(all_records) - len(evaluable)
        )
        metrics[f"{level}_miss_count"] = sum(
            bool(record.get(f"{level}_miss")) for record in evaluable
        )
        metrics[f"{level}_mrr"] = statistics.fmean(
            float(record[f"{level}_reciprocal_rank"])
            for record in evaluable
        ) if evaluable else 0.0
        for cutoff in cutoffs:
            for metric in ("hit", "recall", "ndcg"):
                key = f"{level}_{metric}_at_{cutoff}"
                metrics[key] = statistics.fmean(
                    float(record[key]) for record in evaluable
                ) if evaluable else 0.0

    # 兼容第一版 summary 字段；无前缀指标始终代表文档级。
    metrics["evaluable_question_count"] = metrics[
        "document_evaluable_question_count"
    ]
    metrics["excluded_question_count"] = metrics[
        "document_excluded_question_count"
    ]
    metrics["miss_count"] = metrics["document_miss_count"]
    metrics["mrr"] = metrics["document_mrr"]
    for cutoff in cutoffs:
        for metric in ("hit", "recall", "ndcg"):
            metrics[f"{metric}_at_{cutoff}"] = metrics[
                f"document_{metric}_at_{cutoff}"
            ]
    return metrics


def _group_metrics(
    records: Sequence[dict[str, Any]],
    *,
    cutoffs: Sequence[int],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["question_type"])].append(record)
    return {
        name: aggregate_metrics(group, cutoffs=cutoffs)
        for name, group in sorted(groups.items())
    }


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown_report(summary: dict[str, Any]) -> str:
    corpus = summary["corpus"]
    metrics = summary["metrics"]
    strategy_label = summary["strategy_label"]
    lines = [
        f"# {strategy_label} 检索离线评估",
        "",
        f"- 运行时间：{summary['run_at']}",
        f"- Top-K：{summary['parameters']['top_k']}",
        f"- Golden Dataset SHA256：{summary['dataset']['ground_truth_sha256']}",
        f"- 目标文档：{corpus['manifest_file_count']}，成功选中："
        f"{corpus['selected_document_count']}",
        f"- 选中 Chunk：{corpus['selected_chunk_count']}，Token："
        f"{corpus['selected_token_count']}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 文档级 | Chunk 级 |",
        "|---|---:|---:|",
        f"| 可评估题数 | {metrics['document_evaluable_question_count']} | "
        f"{metrics['chunk_evaluable_question_count']} |",
        f"| MRR | {metrics['document_mrr']:.4f} | "
        f"{metrics['chunk_mrr']:.4f} |",
    ]
    if summary["embedding"] is not None:
        detail_lines = [
            f"- Embedding：{summary['embedding']['model']} / "
            f"{summary['embedding']['dimensions']} 维",
            f"- Min score：{summary['parameters']['min_score']}",
        ]
        if summary["strategy"] in {
            "dense_bm25_normalized_weighted_rrf",
            "dense_bm25_normalized_weighted_rrf_reranker",
        }:
            detail_lines.extend(
                [
                "- 融合：Dense/BM25 候选分数分别做 Min-Max 归一化，"
                "加权后与加权 RRF 分数组合",
                f"- 候选池：每路 Top-{summary['parameters']['candidate_k']}",
                f"- Dense/BM25 权重：{summary['parameters']['dense_weight']} / "
                f"{summary['parameters']['bm25_weight']}",
                f"- RRF 权重 / k：{summary['parameters']['rrf_weight']} / "
                f"{summary['parameters']['rrf_k']}",
                ]
            )
        if summary["reranker"] is not None:
            detail_lines.extend(
                [
                    f"- Reranker：{summary['reranker']['model']}，候选 Top-"
                    f"{summary['parameters']['rerank_candidate_k']} → Top-"
                    f"{summary['parameters']['top_k']}",
                    f"- Reranker 延迟 P50/P95："
                    f"{summary['reranker']['latency_ms_p50']:.0f}/"
                    f"{summary['reranker']['latency_ms_p95']:.0f} ms",
                    f"- Reranker 总 Token："
                    f"{summary['reranker']['total_tokens']}",
                ]
            )
        lines[3:3] = detail_lines
    else:
        lines[3:3] = ["- 检索引擎：ParadeDB pg_search BM25"]
    for cutoff in summary["parameters"]["cutoffs"]:
        lines.append(
            f"| Hit@{cutoff} | "
            f"{_format_percent(metrics[f'document_hit_at_{cutoff}'])} | "
            f"{_format_percent(metrics[f'chunk_hit_at_{cutoff}'])} |"
        )
        lines.append(
            f"| Recall@{cutoff} | "
            f"{_format_percent(metrics[f'document_recall_at_{cutoff}'])} | "
            f"{_format_percent(metrics[f'chunk_recall_at_{cutoff}'])} |"
        )
        lines.append(
            f"| nDCG@{cutoff} | "
            f"{metrics[f'document_ndcg_at_{cutoff}']:.4f} | "
            f"{metrics[f'chunk_ndcg_at_{cutoff}']:.4f} |"
        )
    lines.extend(
        [
            f"| 检索延迟 P50 | {metrics['latency_ms_p50']:.0f} ms |",
            f"| 检索延迟 P95 | {metrics['latency_ms_p95']:.0f} ms |",
            "",
            "## 按题型",
            "",
            "| 题型 | 题数 | 文档 Hit@5 | Chunk Hit@5 | 文档 R@20 | "
            "Chunk R@20 | 文档 MRR | Chunk MRR |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, group in summary["metrics_by_question_type"].items():
        lines.append(
            f"| {name} | {group['document_evaluable_question_count']} | "
            f"{_format_percent(group.get('document_hit_at_5', 0.0))} | "
            f"{_format_percent(group.get('chunk_hit_at_5', 0.0))} | "
            f"{_format_percent(group.get('document_recall_at_20', 0.0))} | "
            f"{_format_percent(group.get('chunk_recall_at_20', 0.0))} | "
            f"{group['document_mrr']:.4f} | {group['chunk_mrr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 语料选择",
            "",
            "| ID | KB | 文件 | Chunk | Token |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for document in corpus["selected_documents"]:
        lines.append(
            f"| {document['id']} | {document['kb_id']} | "
            f"{document['file_name']} | {document['current_chunk_count']} | "
            f"{document['token_count']} |"
        )
    if corpus["duplicate_files"]:
        lines.extend(["", "### 重复上传（评估仅选最新成功记录）", ""])
        for file_name, ids in corpus["duplicate_files"].items():
            lines.append(f"- {file_name}: IDs {', '.join(map(str, ids))}")
    if corpus["excluded_documents"]:
        lines.extend(["", "### 同知识库中已排除的文档", ""])
        for document in corpus["excluded_documents"]:
            lines.append(
                f"- ID {document['id']}：{document['file_name']} "
                f"({document['current_chunk_count']} chunks)"
            )
    document_misses = summary["document_misses"]
    chunk_misses = summary["chunk_misses"]
    lines.extend(
        ["", f"## Top-{summary['parameters']['top_k']} 文档级未命中", ""]
    )
    if document_misses:
        for miss in document_misses:
            lines.append(f"- {miss['question_id']} {miss['question']}")
    else:
        lines.append("无。")
    lines.extend(
        ["", f"## Top-{summary['parameters']['top_k']} Chunk 级未命中", ""]
    )
    if chunk_misses:
        for miss in chunk_misses:
            lines.append(f"- {miss['question_id']} {miss['question']}")
    else:
        lines.append("无。")
    lines.extend(
        [
            "",
            "> 检索阶段只使用 question_id 和 question。文档级指标按 source_files "
            "评分；Chunk 级指标按来源文件内的 Gold 关键词/位置证据 Chunk 组评分。"
            "标准答案、关键词、来源和 Gold Chunk 均不进入 Embedding 或检索。",
            "",
            "> no_answer 与 permission 题不纳入本次召回指标，需要在回答层和 "
            "ACL 层单独评估。",
            "",
        ]
    )
    return "\n".join(lines)


def _corpus_payload(
    selection: CorpusSelection,
    *,
    manifest_file_count: int,
) -> dict[str, Any]:
    selected_documents = [asdict(item) for item in selection.selected]
    return {
        "manifest_file_count": manifest_file_count,
        "selected_document_count": len(selection.selected),
        "selected_chunk_count": sum(
            item.current_chunk_count for item in selection.selected
        ),
        "selected_token_count": sum(item.token_count for item in selection.selected),
        "selected_documents": selected_documents,
        "missing_files": list(selection.missing_files),
        "duplicate_files": {
            name: [item.id for item in snapshots]
            for name, snapshots in selection.duplicate_files.items()
        },
        "excluded_documents": [asdict(item) for item in selection.excluded],
    }


async def run_dense_evaluation(args: argparse.Namespace) -> Path:
    manifest_files = load_manifest_file_names(args.manifest)
    ground_truth = load_ground_truth(args.ground_truth)
    cutoffs = tuple(
        sorted(set(value for value in args.cutoffs if value <= args.top_k))
    )
    if not cutoffs:
        raise ValueError("cutoffs 中至少要有一个值不大于 top_k")

    async with SessionLocal() as session:
        snapshots = await load_document_snapshots(session, kb_ids=args.kb_ids)
        selection = select_corpus(snapshots, manifest_files)
        corpus = _corpus_payload(
            selection,
            manifest_file_count=len(manifest_files),
        )
        print(
            f"语料统计: manifest={len(manifest_files)}, "
            f"selected={len(selection.selected)}, "
            f"chunks={corpus['selected_chunk_count']}, "
            f"tokens={corpus['selected_token_count']}"
        )
        if selection.missing_files:
            missing = "、".join(selection.missing_files)
            raise RuntimeError(f"以下语料没有成功入库或没有 Chunk: {missing}")
        selected_names = {
            _normalized_file_name(item.file_name) for item in selection.selected
        }
        unknown_sources = sorted(
            {
                source_file
                for item in ground_truth
                if item.should_answer
                for source_file in item.source_files
                if _normalized_file_name(source_file) not in selected_names
            }
        )
        if unknown_sources:
            sources = "、".join(unknown_sources)
            raise RuntimeError(f"可回答题引用了未入选的来源文件: {sources}")
        selected_doc_ids = [item.id for item in selection.selected]
        selected_kb_ids = sorted({item.kb_id for item in selection.selected})
        corpus_chunks = await load_corpus_chunks(
            session,
            doc_ids=selected_doc_ids,
        )
        gold_chunk_groups, gold_mapping_issues = resolve_gold_chunk_groups(
            ground_truth,
            corpus_chunks,
        )
        corpus["loaded_chunk_count"] = len(corpus_chunks)
        corpus["gold_chunk_group_count"] = sum(
            len(groups) for groups in gold_chunk_groups.values()
        )
        corpus["gold_chunk_mapping_issues"] = gold_mapping_issues
        if args.stats_only:
            output_dir = args.output_dir or (
                args.ground_truth.parent
                / "eval-results"
                / f"{args.strategy}-stats"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "corpus-stats.json"
            output_path.write_text(
                json.dumps(corpus, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return output_path
        if gold_mapping_issues:
            details = "；".join(
                f"{issue['question_id']}:{issue['source_file']}"
                for issue in gold_mapping_issues
            )
            raise RuntimeError(f"Gold Chunk 标签映射不完整: {details}")

    # 批量调用外部 Embedding 时不占用数据库连接，避免长事务和空闲超时。
    blind_queries = build_blind_queries(ground_truth)
    embedding_summary: dict[str, Any] | None = None
    vectors: list[list[float] | None]
    service: EmbeddingService | None = None
    hybrid_strategies = {"hybrid_rrf", "hybrid_reranker"}
    if args.strategy in {"dense", *hybrid_strategies}:
        service = get_embedding_service()
        embedding_started_at = time.perf_counter()
        embedding = await embed_evaluation_questions(
            service,
            [query.question for query in blind_queries],
        )
        embedding_latency_ms = round(
            (time.perf_counter() - embedding_started_at) * 1000
        )
        vectors = list(embedding.vectors)
        embedding_summary = {
            "model": service.model,
            "dimensions": service.dimensions,
            "batch_latency_ms": embedding_latency_ms,
            "local_cache_hits": embedding.local_cache_hits,
            "redis_cache_hits": embedding.redis_cache_hits,
            "generated": embedding.generated,
        }
    else:
        vectors = [None] * len(blind_queries)

    reranker_service: RerankerService | None = None
    if args.strategy == "hybrid_reranker":
        reranker_kwargs: dict[str, Any] = {}
        if args.reranker_model:
            reranker_kwargs["model"] = args.reranker_model
        if args.reranker_timeout_ms:
            reranker_kwargs["timeout_ms"] = args.reranker_timeout_ms
        reranker_service = RerankerService(**reranker_kwargs)

    retrieval_outputs: list[tuple[str, RetrievalResult]] = []
    try:
        async with SessionLocal() as session:
            for index, (query, vector) in enumerate(
                zip(blind_queries, vectors, strict=True),
                start=1,
            ):
                if args.strategy in {"dense", *hybrid_strategies}:
                    assert vector is not None and service is not None
                    dense_result = await search_by_cosine_vector(
                        session,
                        query=query.question,
                        query_vector=vector,
                        kb_ids=selected_kb_ids,
                        doc_ids=selected_doc_ids,
                        top_k=(
                            args.candidate_k
                            if args.strategy in hybrid_strategies
                            else args.top_k
                        ),
                        min_score=args.min_score,
                        embedding_model=service.model,
                        dimensions=service.dimensions,
                    )
                    if args.strategy == "dense":
                        result = dense_result
                    else:
                        bm25_result = await search_by_bm25(
                            session,
                            query=query.question,
                            kb_ids=selected_kb_ids,
                            doc_ids=selected_doc_ids,
                            top_k=args.candidate_k,
                        )
                        hybrid_result = fuse_dense_bm25_rrf(
                            dense_result,
                            bm25_result,
                            top_k=(
                                args.rerank_candidate_k
                                if args.strategy == "hybrid_reranker"
                                else args.top_k
                            ),
                            dense_weight=args.dense_weight,
                            bm25_weight=args.bm25_weight,
                            rrf_weight=args.rrf_weight,
                            rrf_k=args.rrf_k,
                        )
                        if args.strategy == "hybrid_reranker":
                            assert reranker_service is not None
                            result = await reranker_service.rerank(
                                hybrid_result,
                                top_n=args.top_k,
                            )
                        else:
                            result = hybrid_result
                else:
                    result = await search_by_bm25(
                        session,
                        query=query.question,
                        kb_ids=selected_kb_ids,
                        doc_ids=selected_doc_ids,
                        top_k=args.top_k,
                    )
                retrieval_outputs.append((query.question_id, result))
                print(
                    f"[{index:02d}/{len(blind_queries)}] "
                    f"{query.question_id} RETRIEVED"
                )
    finally:
        if reranker_service is not None:
            await reranker_service.aclose()

    # 全部盲检结束后才加载对应 Gold 字段参与评分。
    ground_truth_by_id = {item.question_id: item for item in ground_truth}
    records: list[dict[str, Any]] = []
    for question_id, result in retrieval_outputs:
        item = ground_truth_by_id[question_id]
        records.append(
            evaluate_retrieval(
                item,
                result,
                gold_chunk_groups=gold_chunk_groups.get(question_id),
                cutoffs=cutoffs,
            )
        )

    run_at = datetime.now().astimezone().isoformat(timespec="seconds")
    strategy_names = {
        "dense": ("dense_cosine", "Dense 余弦"),
        "bm25": ("bm25", "BM25"),
        "hybrid_rrf": (
            "dense_bm25_normalized_weighted_rrf",
            "Dense + BM25 归一化加权 + RRF",
        ),
        "hybrid_reranker": (
            "dense_bm25_normalized_weighted_rrf_reranker",
            "Dense + BM25 归一化加权 + RRF + Reranker",
        ),
    }
    strategy_name, strategy_label = strategy_names[args.strategy]
    parameters = {
        "top_k": args.top_k,
        "min_score": args.min_score,
        "cutoffs": list(cutoffs),
        "kb_ids": selected_kb_ids,
        "doc_ids": selected_doc_ids,
    }
    if args.strategy in hybrid_strategies:
        parameters.update(
            {
                "candidate_k": args.candidate_k,
                "dense_weight": args.dense_weight,
                "bm25_weight": args.bm25_weight,
                "rrf_weight": args.rrf_weight,
                "rrf_k": args.rrf_k,
            }
        )
    reranker_summary: dict[str, Any] | None = None
    if reranker_service is not None:
        reranked = [
            result
            for _, result in retrieval_outputs
            if isinstance(result, RerankerResult)
        ]
        reranker_latencies = [
            float(result.reranker_latency_ms) for result in reranked
        ]
        known_tokens = [
            result.total_tokens
            for result in reranked
            if result.total_tokens is not None
        ]
        parameters["rerank_candidate_k"] = args.rerank_candidate_k
        parameters["reranker_timeout_ms"] = reranker_service.timeout_ms
        reranker_summary = {
            "model": reranker_service.model,
            "request_count": len(reranked),
            "latency_ms_mean": (
                statistics.fmean(reranker_latencies)
                if reranker_latencies
                else 0.0
            ),
            "latency_ms_p50": _percentile(reranker_latencies, 0.50),
            "latency_ms_p95": _percentile(reranker_latencies, 0.95),
            "total_tokens": sum(known_tokens) if known_tokens else None,
        }
    summary = {
        "run_at": run_at,
        "strategy": strategy_name,
        "strategy_label": strategy_label,
        "dataset": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": _sha256_file(args.manifest),
            "ground_truth_path": str(args.ground_truth.resolve()),
            "ground_truth_sha256": _sha256_file(args.ground_truth),
            "question_count": len(ground_truth),
        },
        "methodology": {
            "retrieval_input_fields": ["question_id", "question"],
            "fields_never_sent_to_embedding_or_retrieval": [
                "expected_answer",
                "keywords",
                "source_files",
                "source_location",
                "gold_chunk_groups",
            ],
            "document_relevance": "source_files 中的来源文档",
            "chunk_relevance": (
                "每个来源文件内关键词优先、source_location 兜底映射的 Gold Chunk 组"
            ),
            "scoring_timing": "所有问题完成检索后统一评分",
            "fusion": (
                "每路候选分数独立 Min-Max 归一化；Dense/BM25 加权分"
                "与归一化加权 RRF 分按 rrf_weight 混合"
                if args.strategy in hybrid_strategies
                else None
            ),
            "reranking_input_fields": (
                ["question", "retrieved_chunk.content"]
                if args.strategy == "hybrid_reranker"
                else None
            ),
        },
        "parameters": parameters,
        "embedding": embedding_summary,
        "reranker": reranker_summary,
        "corpus": corpus,
        "metrics": aggregate_metrics(records, cutoffs=cutoffs),
        "metrics_by_question_type": _group_metrics(records, cutoffs=cutoffs),
        "document_misses": [
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "expected_source_files": record["expected_source_files"],
            }
            for record in records
            if record.get("document_miss")
        ],
        "chunk_misses": [
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "expected_source_files": record["expected_source_files"],
                "gold_chunk_groups": record["gold_chunk_groups"],
            }
            for record in records
            if record.get("chunk_miss")
        ],
    }
    summary["misses"] = summary["document_misses"]
    run_name = datetime.now().strftime(f"{args.strategy}-%Y%m%d-%H%M%S")
    output_dir = args.output_dir or (
        args.ground_truth.parent / "eval-results" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "per-question.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "blind-retrieval-input.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for query in blind_queries:
            handle.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")
    chunk_by_id = {chunk.id: chunk for chunk in corpus_chunks}
    with (output_dir / "gold-chunk-map.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for item in ground_truth:
            groups = gold_chunk_groups.get(item.question_id, {})
            if not groups:
                continue
            payload = {
                "question_id": item.question_id,
                "source_location": item.source_location,
                "keywords": list(item.keywords),
                "groups": [
                    {
                        "source_file": source_file,
                        "chunks": [
                            asdict(chunk_by_id[chunk_id])
                            for chunk_id in chunk_ids
                        ],
                    }
                    for source_file, chunk_ids in groups.items()
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    report_path = output_dir / "report.md"
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    return report_path


def build_argument_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行检索离线盲评")
    parser.add_argument(
        "--strategy",
        choices=("dense", "bm25", "hybrid_rrf", "hybrid_reranker"),
        default="dense",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "qa-docs" / "manifest.csv",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=project_root / "qa-docs" / "qa_ground_truth.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="结果目录；默认写入 qa-docs/eval-results/<strategy>-时间戳",
    )
    parser.add_argument(
        "--kb-id",
        dest="kb_ids",
        action="append",
        type=int,
        help="仅检查指定知识库，可重复传入；默认自动从 manifest 匹配",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="混合检索时 Dense 与 BM25 各自召回的候选数",
    )
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument(
        "--rrf-weight",
        type=float,
        default=0.5,
        help="最终分数中归一化 RRF 的占比",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--rerank-candidate-k",
        type=int,
        default=30,
        help="进入 Reranker 的融合候选数",
    )
    parser.add_argument("--reranker-model")
    parser.add_argument(
        "--reranker-timeout-ms",
        type=int,
        help="评估时覆盖 Reranker 请求超时",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="只统计并校验语料，不调用 Embedding 服务",
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = build_argument_parser(project_root)
    args = parser.parse_args(argv)
    if not 1 <= args.top_k <= 100:
        parser.error("--top-k 必须在 1 到 100 之间")
    if not 1 <= args.candidate_k <= 100:
        parser.error("--candidate-k 必须在 1 到 100 之间")
    if args.strategy == "hybrid_rrf" and args.candidate_k < args.top_k:
        parser.error("混合检索的 --candidate-k 不能小于 --top-k")
    if not 1 <= args.rerank_candidate_k <= 100:
        parser.error("--rerank-candidate-k 必须在 1 到 100 之间")
    if args.strategy == "hybrid_reranker":
        if args.rerank_candidate_k < args.top_k:
            parser.error("--rerank-candidate-k 不能小于 --top-k")
        if args.rerank_candidate_k > args.candidate_k:
            parser.error("--rerank-candidate-k 不能大于 --candidate-k")
    if args.reranker_timeout_ms is not None and args.reranker_timeout_ms <= 0:
        parser.error("--reranker-timeout-ms 必须大于 0")
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score 必须在 0 到 1 之间")
    if any(value <= 0 for value in args.cutoffs):
        parser.error("--cutoffs 必须全部大于 0")
    if args.dense_weight < 0 or args.bm25_weight < 0:
        parser.error("Dense/BM25 权重不能小于 0")
    if args.dense_weight + args.bm25_weight <= 0:
        parser.error("Dense/BM25 权重之和必须大于 0")
    if not 0.0 <= args.rrf_weight <= 1.0:
        parser.error("--rrf-weight 必须在 0 到 1 之间")
    if args.rrf_k <= 0:
        parser.error("--rrf-k 必须大于 0")
    if args.kb_ids and any(value <= 0 for value in args.kb_ids):
        parser.error("--kb-id 必须大于 0")
    try:
        report_path = await run_dense_evaluation(args)
        print(f"结果已写入: {report_path}")
        return 0
    finally:
        await close_redis()
        await close_database()
