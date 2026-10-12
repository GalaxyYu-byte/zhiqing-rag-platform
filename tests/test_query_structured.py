"""授权目录统计、分页快照和发布复核；假连接不访问外部数据库。"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime

import psycopg
import pytest
from test_chat_repository import identity

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_metadata.models import ResolvedFilter
from zhiqing_rag.query_retrieval.repository import QueryRetrievalRepository
from zhiqing_rag.query_routing.models import StructuredParams, StructuredPlan
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument
from zhiqing_rag.query_structured import repository as repository_module
from zhiqing_rag.query_structured.models import MAX_STRUCTURED_DOCUMENTS
from zhiqing_rag.query_structured.repository import StructuredQueryRepository
from zhiqing_rag.query_structured.service import StructuredQueryService

NOW = datetime(2026, 10, 11, tzinfo=UTC)


def document(identifier=4, **changes):
    return {
        "knowledge_base_id": 3, "document_id": identifier, "revision_id": 6,
        "generation_id": 5, "title": "工单管理", "document_name": " 工单.pdf ",
        "file_type": "application/pdf", "business_status": "EFFECTIVE",
        "department_id": 7, "event_date": date(2026, 10, 1), "uploaded_at": NOW,
        "updated_at": NOW, "document_code": "TEST-1", "confidentiality": 0,
        "metadata": {}, **changes,
    }


def scope(*, filters=(), identifiers=(4,)):
    return AccessScope(
        1, 2, (3,), identifiers,
        tuple(PublishedDocument(identifier, 3, 5) for identifier in identifiers),
        resolved_filters=filters,
    )


def plan(operation="count_documents", **pagination):
    return StructuredPlan(
        original_query="知识库有多少文档", query="知识库有多少文档",
        requested_route="structured", reason_code="CATALOG_SELECTED",
        params=StructuredParams(operation=operation, **pagination),
    )


class Cursor:
    def __init__(self, rows):
        self.rows = deepcopy(rows)

    def fetchall(self):
        return self.rows


class Connection:
    def __init__(self):
        self.rows = [document()]
        self.anchors = [3]
        self.document_anchors = [4, 8]
        self.calls = []
        self.error = None
        self.refreshes = []

    def execute(self, statement, parameters=()):
        text = statement.as_string(None) if not isinstance(statement, str) else statement
        self.calls.append((text, parameters))
        if self.error:
            raise self.error
        if "SELECT kb.id" in text:
            return Cursor([{"id": value} for value in self.anchors])
        if "SELECT doc.id FROM" in text:
            return Cursor([
                {"id": value} for value in self.document_anchors if value in parameters[-2]
            ])
        if "SELECT doc.kb_id AS knowledge_base_id" in text:
            return Cursor(self.rows)
        raise AssertionError(text)


@pytest.fixture
def database(monkeypatch):
    connection = Connection()

    @contextmanager
    def borrow(settings):
        yield connection

    def refresh(self, conn, user):
        connection.refreshes.append(user)
        return replace(user, department_id=99, clearance=0)

    monkeypatch.setattr(repository_module, "database_connection", borrow)
    monkeypatch.setattr(QueryRetrievalRepository, "_refresh_identity", refresh)
    return connection


def repository():
    return StructuredQueryRepository(Settings(_env_file=None))


async def test_zero_count_and_empty_page_are_successful_and_refresh_identity(database):
    database.rows = []
    service = StructuredQueryService(repository=repository())
    for operation in ("count_documents", "list_documents"):
        result = await service.execute(identity=identity(), scope=scope(), plan=plan(operation))
        assert result.total == 0 and result.documents == ()
        assert result.provenance.knowledge_base_ids == ("3",)
        assert await service.revalidate(identity=identity(), scope=scope(), result=result)
        assert "provenance" not in result.model_dump(mode="json")
    assert len(database.refreshes) == 4
    sql_text, arguments = next(call for call in database.calls if "doc.kb_id AS" in call[0])
    assert "doc.confidentiality <= %s" in sql_text and 99 in arguments


async def test_page_total_uses_all_document_rows_and_stable_revision_order(database):
    database.rows = [document(8), document(4)]
    selected = scope(identifiers=(4, 8))
    result = await repository().execute(
        identity=identity(), scope=selected, plan=plan("list_documents", offset=1, limit=1),
    )
    assert result.total == 2 and [row.document_id for row in result.documents] == ["4"]
    assert len(result.provenance.sources) == 2
    assert result.documents[0].document_name == " 工单.pdf "
    query, arguments = database.calls[-1]
    assert "ORDER BY r.created_at DESC, doc.id DESC" in query
    assert "document_chunk" not in query and "g.id = doc.active_generation_id" in query
    assert "g.status = 'READY'" in query and "unnest(%s::bigint[]" in query
    assert arguments[-4:] == ([4, 8], [3, 3], [5, 5], MAX_STRUCTURED_DOCUMENTS + 1)
    beyond = await repository().execute(
        identity=identity(), scope=selected, plan=plan("list_documents", offset=10, limit=1),
    )
    assert beyond.total == 2 and beyond.documents == ()


async def test_filters_are_parameterized_before_count_and_use_active_revision(database):
    needle = "x' OR 1=1 --"
    filters = (ResolvedFilter(field="document_name", operator="contains", values=(needle,)),)
    result = await repository().execute(
        identity=identity(), scope=scope(filters=filters), plan=plan(),
    )
    query, arguments = database.calls[-1]
    assert needle not in query and any(needle in str(value) for value in arguments)
    assert "strpos(lower(r.original_filename), lower(%s)) > 0" in query
    assert "r.id = g.revision_id" in query and result.applied_filters == filters


@pytest.mark.parametrize("change", [
    {"updated_at": datetime(2026, 10, 12, tzinfo=UTC)},
    {"business_status": "REPEALED"}, {"document_code": "CHANGED"},
    {"metadata": {"business": "changed"}}, {"generation_id": 99},
])
async def test_changed_metadata_or_published_version_rejects_entire_count(database, change):
    result = await repository().execute(identity=identity(), scope=scope(), plan=plan())
    database.rows[0].update(change)
    assert not await repository().revalidate(identity=identity(), scope=scope(), result=result)


async def test_zero_result_still_requires_current_knowledge_base_read_grant(database):
    database.rows = []
    result = await repository().execute(identity=identity(), scope=scope(), plan=plan())
    database.anchors = []
    assert not await repository().revalidate(identity=identity(), scope=scope(), result=result)
    with pytest.raises(AuthError) as error:
        await repository().execute(identity=identity(), scope=scope(), plan=plan())
    assert error.value.code == "QUERY_SCOPE_FORBIDDEN"


async def test_zero_result_selected_document_revocation_rejects_publication(database):
    database.rows = []
    result = await repository().execute(identity=identity(), scope=scope(), plan=plan())
    assert result.provenance.selected_document_ids == ("4",)
    database.document_anchors = []
    assert not await repository().revalidate(identity=identity(), scope=scope(), result=result)


async def test_revalidation_checks_sources_outside_displayed_page(database):
    database.rows = [document(8), document(4)]
    selected = scope(identifiers=(4, 8))
    result = await repository().execute(
        identity=identity(), scope=selected, plan=plan("list_documents", limit=1),
    )
    database.rows.pop()
    assert not await repository().revalidate(identity=identity(), scope=selected, result=result)


async def test_provenance_limit_is_explicit_and_does_not_truncate_total(database):
    database.rows = [document()] * (MAX_STRUCTURED_DOCUMENTS + 1)
    with pytest.raises(AuthError) as error:
        await repository().execute(identity=identity(), scope=scope(), plan=plan())
    assert error.value.code == "STRUCTURED_SCOPE_LIMIT"


async def test_database_failure_is_not_reported_as_zero_count(database):
    database.error = psycopg.OperationalError("unavailable")
    with pytest.raises(AuthError) as error:
        await repository().execute(identity=identity(), scope=scope(), plan=plan())
    assert error.value.code == "STRUCTURED_DATABASE_UNAVAILABLE"


async def test_cross_tenant_scope_is_rejected_before_database_work(database):
    with pytest.raises(AuthError):
        await repository().execute(identity=identity(tenant_id=9), scope=scope(), plan=plan())
    assert database.calls == [] and database.refreshes == []
