"""统计来源进入聊天快照后，提交、回放及历史范围都必须复核。"""

from contextlib import contextmanager
from copy import deepcopy

import pytest
from test_chat_repository import Cursor, FakeConnection, identity, response, run_row
from test_query_structured import NOW, document, scope

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.chat import repository as chat_module
from zhiqing_rag.modules.chat.repository import ChatRepository, _structured_sources
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_execution import structured_answer_text
from zhiqing_rag.query_structured.models import StructuredProvenance, StructuredResult
from zhiqing_rag.query_structured.repository import _document, _source


def statistics(*, rows=None, operation="count_documents"):
    rows = [] if rows is None else rows
    provenance = StructuredProvenance(
        tenant_id="1", user_id="2", knowledge_base_ids=("3",), operation=operation,
        selected_document_ids=tuple(str(row["document_id"]) for row in rows) or ("4",),
        executed_at=NOW, sources=tuple(map(_source, rows)),
    )
    result = StructuredResult(
        operation=operation, total=len(rows), executed_at=NOW, provenance=provenance,
        documents=tuple(map(_document, rows[:20])) if operation == "list_documents" else (),
    )
    data = response(with_evidence=False)
    data["plan"] = {
        "route": "structured", "original_query": "统计文档数量", "query": "统计文档数量",
        "requested_route": "structured", "reason_code": "CATALOG_SELECTED",
        "params": {"operation": operation},
    }
    data["structured"] = result.model_dump(mode="json")
    data["answer"] = {
        "status": "structured", "message": structured_answer_text(result),
        "review_status": "passed", "statements": [], "model": None,
        "trace": {"input_tokens": 0, "output_tokens": 0},
    }
    return data, provenance.model_dump(mode="json")


class StructuredConnection(FakeConnection):
    def __init__(self):
        super().__init__()
        self.anchors = [3]
        self.document_anchors = [4, 8]
        self.documents = []

    def execute(self, statement, parameters=()):
        rendered = statement if isinstance(statement, str) else statement.as_string(None)
        if "SELECT kb.id" in rendered:
            self.calls.append((rendered, parameters))
            return Cursor([{"id": value} for value in self.anchors])
        if "SELECT doc.id FROM" in rendered:
            self.calls.append((rendered, parameters))
            return Cursor([
                {"id": value} for value in self.document_anchors if value in parameters[-2]
            ])
        if "SELECT doc.kb_id AS knowledge_base_id" in rendered:
            self.calls.append((rendered, parameters))
            return Cursor(self.documents)
        return super().execute(statement, parameters)


@pytest.fixture
def database(monkeypatch):
    connection = StructuredConnection()

    @contextmanager
    def borrow(settings):
        original = deepcopy((connection.runs, connection.messages, connection.citations))
        try:
            yield connection
        except BaseException:
            connection.runs, connection.messages, connection.citations = original
            raise

    monkeypatch.setattr(chat_module, "database_connection", borrow)
    return connection


def repository():
    return ChatRepository(Settings(_env_file=None))


async def complete(data, provenance=None, history=None):
    return await repository().complete(
        identity(), 20, 1, lease_owner="worker", lease_epoch=1, response=data,
        standalone_question="统计文档数量", structured_provenance=provenance,
        history_run_ids=history,
    )


async def test_zero_count_commit_and_replay_require_knowledge_base_grant(database):
    data, provenance = statistics()
    database.runs[1] = run_row()
    await complete(data, provenance)
    assert database.runs[1]["execution_snapshot"]["structured_provenance"] == provenance
    assert "provenance" not in database.runs[1]["execution_snapshot"]["response"]["structured"]
    restored = await repository().get_run(identity(), 20, 1)
    assert restored["visibility"] == "visible" and restored["response"]["structured"]["total"] == 0
    database.anchors = []
    restored = await repository().get_run(identity(), 20, 1)
    assert restored["visibility"] == "unavailable" and restored["response"] is None


async def test_zero_count_revoked_before_commit_is_not_persisted(database):
    data, provenance = statistics()
    database.runs[1] = run_row()
    database.anchors = []
    with pytest.raises(AuthError) as error:
        await complete(data, provenance)
    assert error.value.code == "CHAT_SOURCE_UNAVAILABLE"
    assert database.runs[1]["status"] == "RUNNING" and not database.messages


async def test_zero_count_selected_document_revocation_hides_replay(database):
    data, provenance = statistics()
    database.runs[1] = run_row()
    await complete(data, provenance)
    database.document_anchors = []
    restored = await repository().get_run(identity(), 20, 1)
    assert restored["visibility"] == "unavailable" and restored["response"] is None


async def test_statistic_missing_private_provenance_fails_closed(database):
    data, _ = statistics()
    database.runs[1] = run_row()
    with pytest.raises(ValueError, match="授权来源"):
        await complete(data)
    database.runs[1] = run_row(status="SUCCEEDED", result=data)
    restored = await repository().get_run(identity(), 20, 1)
    assert restored["visibility"] == "unavailable"


async def test_metadata_change_hides_full_statistic_even_without_chunk_evidence(database):
    database.documents = [document()]
    data, provenance = statistics(rows=database.documents)
    database.runs[1] = run_row()
    await complete(data, provenance)
    database.documents[0]["document_code"] = "CHANGED"
    restored = await repository().get_run(identity(), 20, 1)
    assert restored["visibility"] == "unavailable" and restored["response"] is None


async def test_history_inherits_zero_result_anchor_and_checks_it_on_new_commit(database):
    data, provenance = statistics()
    prior = run_row(2, status="SUCCEEDED", result=data)
    prior["execution_snapshot"]["structured_provenance"] = provenance
    database.runs[2] = prior
    database.runs[1] = run_row()
    await complete(response(), history=[2])
    inherited = database.runs[1]["execution_snapshot"]["history_structured_provenance"]
    assert inherited == [provenance]
    database.anchors = []
    restored = await repository().get_run(identity(), 20, 1)
    assert restored["visibility"] == "unavailable"
    database.runs[1] = run_row()
    with pytest.raises(AuthError) as error:
        await complete(response(), history=[2])
    assert error.value.code == "CHAT_SOURCE_UNAVAILABLE"


async def test_statistic_history_requires_all_sources_in_new_selection(database):
    database.documents = [document(4), document(8)]
    data, provenance = statistics(rows=database.documents)
    prior = run_row(2, status="SUCCEEDED", result=data)
    prior["execution_snapshot"]["structured_provenance"] = provenance
    database.runs[2] = prior
    assert await repository().structured_history_in_scope(
        identity(), 20, 2, scope(identifiers=(4, 8)),
    )
    assert not await repository().structured_history_in_scope(identity(), 20, 2, scope())


async def test_zero_statistic_history_cannot_follow_a_different_document_selection(database):
    data, provenance = statistics()
    prior = run_row(2, status="SUCCEEDED", result=data)
    prior["execution_snapshot"]["structured_provenance"] = provenance
    database.runs[2] = prior
    assert await repository().structured_history_in_scope(identity(), 20, 2, scope())
    assert not await repository().structured_history_in_scope(
        identity(), 20, 2, scope(identifiers=(8,)),
    )


def test_inconsistent_statistics_provenance_is_rejected():
    data, provenance = statistics()
    provenance["operation"] = "list_documents"
    assert _structured_sources({
        "schema_version": 1, "response": data, "structured_provenance": provenance,
    }) is None
