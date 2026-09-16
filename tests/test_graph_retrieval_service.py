from types import SimpleNamespace

import pytest

from zq_rag_app.services.graph_retrieval_service import search_by_graph


class _GraphResult:
    def __init__(self, records=None):
        self.records = records

    async def data(self):
        return self.records or [
            {
                "claim_uid": "claim-1",
                "subject_uid": "entity-1",
                "subject": "Aurora-KB",
                "predicate": "OWNED_BY",
                "object_uid": "entity-2",
                "object": "AI平台部",
                "polarity": "POSITIVE",
                "valid_from": None,
                "valid_to": None,
                "evidence_quote": "Aurora-KB 由 AI平台部负责。",
                "confidence": 0.95,
                "chunk_uid": "chunk-1",
                "kb_id": 4,
                "doc_id": 7,
                "doc_version": 2,
                "chunk_index": 3,
                "page": 8,
                "section": "项目职责",
                "hop": 1,
                "score": 0.915,
            }
        ]


class _GraphSession:
    def __init__(self, records=None):
        self.query = None
        self.parameters = None
        self.records = records

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def run(self, query, **parameters):
        self.query = query
        self.parameters = parameters
        return _GraphResult(self.records)


class _PostgresResult:
    def __init__(self, rows=None):
        self.rows = rows

    def all(self):
        return self.rows or [
            SimpleNamespace(
                chunk_id=71,
                doc_id=7,
                document="Aurora周会.pdf",
                doc_version=2,
                chunk_index=3,
                section="项目职责",
                page=8,
                content="Aurora-KB 由 AI平台部负责。",
                token_count=12,
            )
        ]


class _PostgresSession:
    def __init__(self, rows=None):
        self.statement = None
        self.rows = rows

    async def execute(self, statement):
        self.statement = statement
        return _PostgresResult(self.rows)


@pytest.mark.asyncio
async def test_graph_retrieval_reads_only_active_evidence_and_hydrates_chunk():
    graph_session = _GraphSession()
    postgres_session = _PostgresSession()

    result = await search_by_graph(
        postgres_session,
        query="  Aurora-KB 谁负责？ ",
        kb_ids=[4, 4],
        doc_ids=[9, 7, 9],
        top_k=10,
        graph_session_factory=lambda: graph_session,
    )

    assert "doc.active_graph_version = chunk.doc_version" in graph_session.query
    assert "chunk.graph_status = 'ACTIVE'" in graph_session.query
    assert graph_session.parameters["kb_ids"] == [4]
    assert graph_session.parameters["doc_ids"] == [7, 9]
    assert graph_session.parameters["claim_limit"] == 500
    assert result.query == "Aurora-KB 谁负责？"
    assert result.results[0].chunk_id == 71
    assert result.results[0].score > 0.4
    assert result.matches[0].evidence_quote == "Aurora-KB 由 AI平台部负责。"
    sql = str(postgres_session.statement)
    assert "kb_doc_chunk.doc_version = kb_document.version" in sql
    assert "kb_document.is_deleted" in sql


@pytest.mark.asyncio
async def test_graph_retrieval_rejects_invalid_scope():
    with pytest.raises(ValueError, match="kb_ids"):
        await search_by_graph(
            _PostgresSession(),
            query="问题",
            kb_ids=[],
            top_k=10,
        )

    with pytest.raises(ValueError, match="max_hops"):
        await search_by_graph(
            _PostgresSession(),
            query="问题",
            kb_ids=[4],
            top_k=10,
            max_hops=3,
        )


@pytest.mark.asyncio
async def test_graph_retrieval_prioritizes_date_and_decision_evidence():
    common = {
        "subject_uid": "entity-1",
        "subject": "Aurora-KB",
        "object_uid": "entity-2",
        "polarity": "POSITIVE",
        "valid_from": None,
        "valid_to": None,
        "confidence": 0.95,
        "kb_id": 4,
        "doc_id": 7,
        "doc_version": 2,
        "page": 8,
        "hop": 1,
        "score": 0.915,
    }
    records = [
        {
            **common,
            "claim_uid": "claim-tech",
            "predicate": "USES",
            "object": "HNSW",
            "evidence_quote": "Aurora-KB 使用 HNSW 和 BM25。",
            "chunk_uid": "chunk-tech",
            "chunk_index": 3,
            "section": "技术方案",
        },
        {
            **common,
            "claim_uid": "claim-date",
            "predicate": "OPEN_DATE_ADJUSTED_TO",
            "object": "2026-09-28",
            "evidence_quote": "原定 2026-09-15 上线，调整为 2026-09-28 全公司开放。",
            "chunk_uid": "chunk-date",
            "chunk_index": 4,
            "section": "上线安排",
        },
    ]
    rows = [
        SimpleNamespace(
            chunk_id=71,
            doc_id=7,
            document="Aurora周会.pdf",
            doc_version=2,
            chunk_index=3,
            section="技术方案",
            page=8,
            content="Aurora-KB 使用 HNSW 和 BM25。",
            token_count=12,
        ),
        SimpleNamespace(
            chunk_id=72,
            doc_id=7,
            document="Aurora周会.pdf",
            doc_version=2,
            chunk_index=4,
            section="上线安排",
            page=9,
            content=(
                "原定 2026-09-15 上线，调整为 2026-09-28 全公司开放。"
                "原因：权限回归和故障演练需要更多时间。"
            ),
            token_count=28,
        ),
        SimpleNamespace(
            chunk_id=74,
            doc_id=7,
            document="Aurora周会.pdf",
            doc_version=2,
            chunk_index=6,
            section="项目台账",
            page=10,
            content="Aurora-KB 当前计划日期 2026-09-28，状态：延期中。",
            token_count=16,
        ),
        SimpleNamespace(
            chunk_id=73,
            doc_id=7,
            document="Aurora周会.pdf",
            doc_version=2,
            chunk_index=5,
            section="推荐问题",
            page=10,
            content="推荐问题：Aurora-KB 什么时间上线？答案：2026-09-28。",
            token_count=16,
        ),
    ]

    result = await search_by_graph(
        _PostgresSession(rows),
        query="Aurora-KB 为什么延期到 9 月 28 日？",
        kb_ids=[4],
        top_k=1,
        graph_session_factory=lambda: _GraphSession(records),
    )

    assert [chunk.chunk_id for chunk in result.results] == [72]
    assert [match.claim_uid for match in result.matches] == ["claim-date"]


@pytest.mark.asyncio
async def test_graph_retrieval_expands_to_relevant_sibling_chunk_in_active_document():
    records = [
        {
            "claim_uid": "claim-project",
            "subject_uid": "entity-1",
            "subject": "Aurora-KB",
            "predicate": "USES",
            "object_uid": "entity-2",
            "object": "HNSW",
            "polarity": "POSITIVE",
            "valid_from": None,
            "valid_to": None,
            "evidence_quote": "Aurora-KB 使用 HNSW。",
            "confidence": 0.95,
            "chunk_uid": "chunk-project",
            "kb_id": 4,
            "doc_id": 7,
            "doc_version": 2,
            "chunk_index": 3,
            "page": 8,
            "section": "技术方案",
            "hop": 1,
            "score": 0.915,
        }
    ]
    rows = [
        SimpleNamespace(
            chunk_id=71,
            doc_id=7,
            document="Aurora周会.pdf",
            doc_version=2,
            chunk_index=3,
            section="技术方案",
            page=8,
            content="Aurora-KB 使用 HNSW。",
            token_count=8,
        ),
        SimpleNamespace(
            chunk_id=72,
            doc_id=7,
            document="Aurora周会.pdf",
            doc_version=2,
            chunk_index=4,
            section="上线安排",
            page=9,
            content=(
                "原定 2026-09-15 上线，调整为 2026-09-28 全公司开放。"
                "原因：权限回归和故障演练需要更多时间。"
            ),
            token_count=28,
        ),
        SimpleNamespace(
            chunk_id=73,
            doc_id=7,
            document="Aurora周会.pdf",
            doc_version=2,
            chunk_index=5,
            section="项目台账",
            page=10,
            content="Aurora-KB 当前计划日期 2026-09-28，状态：延期中。",
            token_count=16,
        ),
    ]

    result = await search_by_graph(
        _PostgresSession(rows),
        query="Aurora-KB 为什么延期到 9 月 28 日？",
        kb_ids=[4],
        top_k=1,
        graph_session_factory=lambda: _GraphSession(records),
    )

    assert [chunk.chunk_id for chunk in result.results] == [72]
    assert result.matches == ()
