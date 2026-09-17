"""基于 Neo4j 活动图版本的实体、事实和证据 Chunk 召回。"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.neo4j import get_neo4j_session
from ..graph.identity import normalize_identity_text
from ..models.document import DocChunk, Document
from .retrieval_service import RetrievedChunk


_GRAPH_CLAIM_SEARCH = """
MATCH (seed:Entity)
WHERE seed.kb_id IN $kb_ids
  AND (
    ($seed_entity_uids IS NOT NULL AND seed.uid IN $seed_entity_uids)
    OR ($seed_entity_uids IS NULL AND (
    $query_normalized = seed.canonical_name_normalized
    OR $query_normalized CONTAINS seed.canonical_name_normalized
    OR seed.canonical_name_normalized CONTAINS $query_normalized
    OR any(alias IN coalesce(seed.aliases, [])
           WHERE $query_normalized CONTAINS toLower(trim(alias)))
    ))
  )
WITH seed,
     CASE
       WHEN $query_normalized = seed.canonical_name_normalized THEN 1.0
       WHEN $query_normalized CONTAINS seed.canonical_name_normalized THEN 0.9
       ELSE 0.8
     END AS seed_score
ORDER BY seed_score DESC, size(seed.canonical_name_normalized) DESC, seed.uid
LIMIT $entity_limit
MATCH path=(seed)-[:SUBJECT_OF|OBJECT*1..4]-(neighbor:Entity)
WHERE length(path) IN [2, 4]
UNWIND [node IN nodes(path) WHERE node:Claim] AS claim
MATCH (subject:Entity)-[:SUBJECT_OF]->(claim)-[:OBJECT]->(object:Entity)
MATCH (claim)-[evidence:SUPPORTED_BY]->(chunk:Chunk)
MATCH (doc:Document {doc_id: chunk.doc_id})
WHERE claim.kb_id IN $kb_ids
  AND chunk.kb_id IN $kb_ids
  AND doc.kb_id = chunk.kb_id
  AND doc.active_graph_version = chunk.doc_version
  AND chunk.graph_status = 'ACTIVE'
  AND ($doc_ids IS NULL OR chunk.doc_id IN $doc_ids)
WITH claim, subject, object, chunk, evidence,
     min(CASE WHEN subject = seed OR object = seed THEN 1 ELSE 2 END) AS hop,
     max(seed_score) AS seed_score
RETURN claim.uid AS claim_uid,
       subject.uid AS subject_uid,
       subject.canonical_name AS subject,
       claim.predicate AS predicate,
       object.uid AS object_uid,
       object.canonical_name AS object,
       claim.polarity AS polarity,
       claim.valid_from AS valid_from,
       claim.valid_to AS valid_to,
       evidence.quote AS evidence_quote,
       coalesce(evidence.confidence, 0.5) AS confidence,
       chunk.uid AS chunk_uid,
       chunk.kb_id AS kb_id,
       chunk.doc_id AS doc_id,
       chunk.doc_version AS doc_version,
       chunk.chunk_index AS chunk_index,
       chunk.page_num AS page,
       chunk.section_title AS section,
       hop,
       0.7 * seed_score
         + 0.3 * coalesce(evidence.confidence, 0.5)
         - 0.1 * (hop - 1) AS score
ORDER BY score DESC, hop ASC, claim.uid, chunk.uid
LIMIT $claim_limit
"""


@dataclass(slots=True, frozen=True)
class GraphClaimMatch:
    claim_uid: str
    subject_uid: str
    subject: str
    predicate: str
    object_uid: str
    object: str
    polarity: str
    valid_from: str | None
    valid_to: str | None
    evidence_quote: str
    confidence: float
    chunk_uid: str
    kb_id: int
    doc_id: int
    doc_version: int
    chunk_index: int
    page: int | None
    section: str | None
    hop: int
    score: float


@dataclass(slots=True, frozen=True)
class GraphRetrievalResult:
    query: str
    engine: str
    latency_ms: int
    results: list[RetrievedChunk]
    matches: tuple[GraphClaimMatch, ...]


_DATE_PATTERN = re.compile(
    r"(?:(?P<year>20\d{2})\s*(?:[-/.]|年)\s*)?"
    r"(?P<month>1[0-2]|0?[1-9])\s*(?:[-/.]|月)\s*"
    r"(?P<day>3[01]|[12]\d|0?[1-9])\s*日?"
)
_LATIN_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9_.+-]{1,}")
_CHINESE_SPAN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
_QUERY_FILLERS = ("请问", "麻烦", "一下", "告诉我", "是什么", "怎么样")
_STALE_MARKERS = ("已归档", "已废止", "历史草案", "不应作为当前", "仅供历史")
_INTENT_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("为什么", "为何", "原因", "怎么会"),
        ("原因", "因为", "由于", "导致", "异常", "故障", "失败", "未达到", "否则", "阻塞"),
    ),
    (
        ("延期", "推迟", "上线", "开放", "日期", "时间", "哪一天", "什么时候"),
        ("延期", "推迟", "调整", "原定", "上线", "开放", "日期", "时间", "计划", "安排"),
    ),
    (
        ("谁负责", "负责人", "负责", "归属"),
        ("负责", "负责人", "归属", "owner", "owned_by", "responsible"),
    ),
)


def _normalize_relevance_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _extract_dates(value: str) -> set[str]:
    dates: set[str] = set()
    for match in _DATE_PATTERN.finditer(_normalize_relevance_text(value)):
        month_day = f"{int(match.group('month')):02d}-{int(match.group('day')):02d}"
        dates.add(month_day)
        if match.group("year"):
            dates.add(f"{match.group('year')}-{month_day}")
    return dates


def _query_terms(value: str) -> set[str]:
    normalized = _normalize_relevance_text(value)
    normalized = _DATE_PATTERN.sub(" ", normalized)
    for filler in _QUERY_FILLERS:
        normalized = normalized.replace(filler, " ")

    terms = set(_LATIN_TOKEN_PATTERN.findall(normalized))
    for span in _CHINESE_SPAN_PATTERN.findall(normalized):
        if len(span) <= 3:
            terms.add(span)
            continue
        terms.update(span[index : index + 2] for index in range(len(span) - 1))
        terms.update(span[index : index + 3] for index in range(len(span) - 2))
    return {term for term in terms if term not in {"什么", "怎么", "哪一", "一天"}}


def _text_relevance(query_without_entities: str, evidence_text: str) -> float:
    normalized_evidence = _normalize_relevance_text(evidence_text)
    terms = _query_terms(query_without_entities)
    lexical_score = (
        sum(1 for term in terms if term in normalized_evidence) / len(terms)
        if terms
        else 0.0
    )

    query_dates = _extract_dates(query_without_entities)
    evidence_dates = _extract_dates(evidence_text)
    date_score = 0.0
    if query_dates:
        date_score = max(
            (
                1.0
                if query_date in evidence_dates
                or any(date.endswith(query_date) for date in evidence_dates)
                else 0.0
            )
            for query_date in query_dates
        )

    active_intents = [
        evidence_terms
        for query_terms, evidence_terms in _INTENT_GROUPS
        if any(term in query_without_entities for term in query_terms)
    ]
    intent_score = (
        sum(
            1.0
            if any(term in normalized_evidence for term in evidence_terms)
            else 0.0
            for evidence_terms in active_intents
        )
        / len(active_intents)
        if active_intents
        else 0.0
    )

    if query_dates:
        score = 0.5 * date_score + 0.2 * lexical_score + 0.3 * intent_score
    else:
        score = 0.75 * lexical_score + 0.25 * intent_score
    asks_for_reason = any(
        term in query_without_entities for term in ("为什么", "为何", "原因", "怎么会")
    )
    if asks_for_reason:
        contains_reason = any(
            term in normalized_evidence
            for term in ("原因", "因为", "由于", "导致")
        )
        score = min(1.0, score + 0.15) if contains_reason else score * 0.8
    if any(marker in normalized_evidence for marker in _STALE_MARKERS):
        score *= 0.55
    return score


def _query_without_seed_entities(
    query: str, matches: tuple[GraphClaimMatch, ...]
) -> str:
    normalized_query = _normalize_relevance_text(query)
    entity_names = {
        _normalize_relevance_text(name).strip()
        for match in matches
        for name in (match.subject, match.object)
    }
    for entity_name in sorted(entity_names, key=len, reverse=True):
        if len(entity_name) >= 2 and entity_name in normalized_query:
            normalized_query = normalized_query.replace(entity_name, " ")
    return normalized_query


def _lexical_relevance(
    query_without_entities: str, match: GraphClaimMatch
) -> float:
    evidence_text = " ".join(
        filter(
            None,
            (
                match.subject,
                match.predicate,
                match.object,
                match.evidence_quote,
                match.section,
            ),
        )
    )
    return _text_relevance(query_without_entities, evidence_text)


def _rescore_matches(
    query: str, matches: tuple[GraphClaimMatch, ...]
) -> tuple[GraphClaimMatch, ...]:
    query_without_entities = _query_without_seed_entities(query, matches)
    rescored = (
        replace(
            match,
            score=min(
                1.0,
                0.25 * match.score
                + 0.75 * _lexical_relevance(query_without_entities, match),
            ),
        )
        for match in matches
    )
    return tuple(sorted(rescored, key=lambda match: (-match.score, match.hop, match.claim_uid)))


async def search_by_graph(
    session: AsyncSession,
    *,
    query: str,
    kb_ids: list[int],
    top_k: int,
    max_hops: int = 2,
    entity_limit: int = 20,
    doc_ids: list[int] | None = None,
    seed_entity_uids: list[str] | None = None,
    graph_session_factory: Callable[[], Any] = get_neo4j_session,
) -> GraphRetrievalResult:
    """按问题中的实体召回 1–2 跳事实，并映射回 PostgreSQL 正式 Chunk。"""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")
    normalized_kb_ids = sorted(set(kb_ids))
    if not normalized_kb_ids or any(kb_id <= 0 for kb_id in normalized_kb_ids):
        raise ValueError("kb_ids 必须包含至少一个有效知识库 ID")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k 必须在 1 到 100 之间")
    if max_hops not in {1, 2}:
        raise ValueError("max_hops 只能是 1 或 2")
    if not 1 <= entity_limit <= 50:
        raise ValueError("entity_limit 必须在 1 到 50 之间")
    if seed_entity_uids is not None and (
        not seed_entity_uids or len(seed_entity_uids) > 20
        or any(not isinstance(uid, str) or not uid.strip() for uid in seed_entity_uids)
    ):
        raise ValueError("seed_entity_uids 必须包含 1 到 20 个有效 UID")

    normalized_doc_ids: list[int] | None = None
    if doc_ids is not None:
        normalized_doc_ids = sorted(set(doc_ids))
        if not normalized_doc_ids or any(doc_id <= 0 for doc_id in normalized_doc_ids):
            raise ValueError("doc_ids 必须包含至少一个有效文档 ID")

    started_at = time.perf_counter()
    query_normalized = normalize_identity_text(normalized_query)
    # 图结构分数只能表达实体距离，无法在 Cypher 阶段判断“延期/日期”等问题意图。
    # 保留足够大的候选池给后续证据文本重排；最终响应仍只返回 top_k Chunk。
    claim_limit = 500
    async with graph_session_factory() as graph_session:
        result = await graph_session.run(
            _GRAPH_CLAIM_SEARCH,
            query_normalized=query_normalized,
            kb_ids=normalized_kb_ids,
            doc_ids=normalized_doc_ids,
            entity_limit=entity_limit,
            claim_limit=claim_limit,
            seed_entity_uids=seed_entity_uids,
        )
        records = await result.data()

    matches = tuple(
        GraphClaimMatch(
            claim_uid=str(record["claim_uid"]),
            subject_uid=str(record["subject_uid"]),
            subject=str(record["subject"]),
            predicate=str(record["predicate"]),
            object_uid=str(record["object_uid"]),
            object=str(record["object"]),
            polarity=str(record["polarity"]),
            valid_from=(
                str(record["valid_from"])
                if record.get("valid_from") is not None
                else None
            ),
            valid_to=(
                str(record["valid_to"])
                if record.get("valid_to") is not None
                else None
            ),
            evidence_quote=str(record["evidence_quote"]),
            confidence=float(record["confidence"]),
            chunk_uid=str(record["chunk_uid"]),
            kb_id=int(record["kb_id"]),
            doc_id=int(record["doc_id"]),
            doc_version=int(record["doc_version"]),
            chunk_index=int(record["chunk_index"]),
            page=record.get("page"),
            section=record.get("section"),
            hop=int(record["hop"]),
            score=float(record["score"]),
        )
        for record in records
        if int(record["hop"]) <= max_hops
    )
    matches = _rescore_matches(normalized_query, matches)
    if not matches:
        return GraphRetrievalResult(
            query=normalized_query,
            engine="neo4j_active_claims",
            latency_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
            results=[],
            matches=(),
        )

    matches_by_key: dict[tuple[int, int, int], list[GraphClaimMatch]] = defaultdict(list)
    for match in matches:
        matches_by_key[(match.doc_id, match.doc_version, match.chunk_index)].append(match)

    matched_document_versions = sorted(
        {(match.doc_id, match.doc_version) for match in matches}
    )
    statement = (
        select(
            DocChunk.id.label("chunk_id"),
            DocChunk.doc_id,
            Document.file_name.label("document"),
            DocChunk.doc_version,
            DocChunk.chunk_index,
            DocChunk.section_title.label("section"),
            DocChunk.page_num.label("page"),
            DocChunk.content,
            DocChunk.token_count,
        )
        .join(Document, Document.id == DocChunk.doc_id)
        .where(
            tuple_(DocChunk.doc_id, DocChunk.doc_version).in_(
                matched_document_versions
            ),
            DocChunk.kb_id.in_(normalized_kb_ids),
            Document.kb_id.in_(normalized_kb_ids),
            DocChunk.doc_version == Document.version,
            Document.status == "DONE",
            Document.is_deleted.is_(False),
        )
    )
    if normalized_doc_ids is not None:
        statement = statement.where(DocChunk.doc_id.in_(normalized_doc_ids))
    try:
        rows = (await session.execute(statement)).all()
    except asyncio.CancelledError:
        await session.rollback()
        raise
    scored_rows: list[tuple[Any, float]] = []
    query_without_entities = _query_without_seed_entities(normalized_query, matches)
    for row in rows:
        key = (int(row.doc_id), int(row.doc_version), int(row.chunk_index))
        chunk_matches = matches_by_key.get(key, [])
        direct_score = 0.0
        if chunk_matches:
            direct_score = max(match.score for match in chunk_matches)
            direct_score += min(0.1, 0.02 * (len(chunk_matches) - 1))
        content_text = " ".join(
            filter(None, (row.section, str(row.content)))
        )
        content_score = 0.05 + 0.85 * _text_relevance(
            query_without_entities, content_text
        )
        score = min(1.0, max(direct_score, content_score))
        scored_rows.append((row, score))
    scored_rows.sort(key=lambda item: (-item[1], int(item[0].chunk_id)))

    chunks = [
        RetrievedChunk(
            rank=rank,
            chunk_id=int(row.chunk_id),
            doc_id=int(row.doc_id),
            document=str(row.document),
            chunk_index=int(row.chunk_index),
            section=row.section,
            page=row.page,
            content=str(row.content),
            token_count=int(row.token_count),
            score=score,
        )
        for rank, (row, score) in enumerate(scored_rows[:top_k], start=1)
    ]
    selected_keys = {
        (int(row.doc_id), int(row.doc_version), int(row.chunk_index))
        for row, _ in scored_rows[:top_k]
    }
    selected_matches = tuple(
        match
        for match in matches
        if (match.doc_id, match.doc_version, match.chunk_index) in selected_keys
    )[: min(100, top_k * 10)]
    return GraphRetrievalResult(
        query=normalized_query,
        engine="neo4j_active_claims",
        latency_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
        results=chunks,
        matches=selected_matches,
    )
