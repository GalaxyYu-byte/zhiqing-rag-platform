"""可选本地 PostgreSQL 统计验证；复用每项测试独立 schema 的安全 fixture。"""

import os

import pytest
import test_chat_persistence_integration as chat_integration
from test_chat_repository import identity, response
from test_query_structured import plan

from zhiqing_rag.query_execution import structured_answer_text
from zhiqing_rag.query_metadata.models import ResolvedFilter
from zhiqing_rag.query_retrieval.repository import QueryRetrievalRepository
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument
from zhiqing_rag.query_structured.repository import StructuredQueryRepository

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CHAT_DB_TESTS") != "1",
    reason="设置 RUN_CHAT_DB_TESTS=1 才执行 PostgreSQL 隔离 schema 统计验证",
)
database_fixture = chat_integration.database


@pytest.fixture
def structured_database(database_fixture, monkeypatch):
    database = database_fixture
    database.execute("""ALTER TABLE {schema}.document
        ADD COLUMN business_status text NOT NULL DEFAULT 'UNKNOWN',
        ADD COLUMN event_date date,
        ADD COLUMN document_code text,
        ADD COLUMN metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
        ADD COLUMN updated_at timestamptz NOT NULL DEFAULT clock_timestamp()""")
    database.execute("""ALTER TABLE {schema}.document_revision
        ADD COLUMN original_filename text NOT NULL DEFAULT '工单.pdf',
        ADD COLUMN mime_type text NOT NULL DEFAULT 'application/pdf',
        ADD COLUMN created_at timestamptz NOT NULL DEFAULT clock_timestamp()""")
    database.execute("""UPDATE {schema}.document SET business_status = 'EFFECTIVE',
        event_date = '2026-10-01', document_code = 'TEST-1' WHERE id = 4""")
    # 仅替换登录会话表，ACL、JOIN、过滤和来源复核均在真实 PostgreSQL 执行。
    monkeypatch.setattr(
        QueryRetrievalRepository, "_refresh_identity", lambda self, connection, current: current,
    )
    return database


def selected(filters=(), *, second=False):
    documents = (PublishedDocument(4, 3, 5),)
    if second:
        documents += (PublishedDocument(8, 3, 10),)
    return AccessScope(
        1, 2, (3,), tuple(item.document_id for item in documents), documents,
        resolved_filters=filters,
    )


def result_response(result):
    data = response(with_evidence=False)
    data["plan"] = plan(result.operation).model_dump(mode="json")
    data["structured"] = result.model_dump(mode="json")
    data["answer"] = {
        "status": "structured", "message": structured_answer_text(result),
        "review_status": "passed", "statements": [], "model": None,
        "trace": {"input_tokens": 0, "output_tokens": 0},
    }
    return data


async def save_result(database, result):
    session_id = await chat_integration.create_session(database)
    run = await chat_integration.start(database, session_id, {
        **chat_integration.request(), "question": "知识库有多少文档", "knowledge_base_ids": ["3"],
    })
    await database.repository.complete(
        identity(), session_id, int(run["id"]), lease_owner="worker", lease_epoch=1,
        response=result_response(result), standalone_question="知识库有多少文档",
        structured_provenance=result.provenance.model_dump(mode="json"),
    )
    return session_id, int(run["id"])


async def test_real_count_filters_active_revision_and_independent_page_total(structured_database):
    database = structured_database
    database.execute("""INSERT INTO {schema}.document
        (id, tenant_id, kb_id, active_generation_id, status, confidentiality, department_id,
         created_by, title, business_status, document_code)
        VALUES (8, 1, 3, 10, 'PUBLISHED', 0, NULL, 10, '草稿制度', 'DRAFT', 'TEST-2')""")
    database.execute("""INSERT INTO {schema}.document_revision
        (id, tenant_id, document_id, original_filename, mime_type)
        VALUES (9, 1, 8, '制度.docx',
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document')""")
    database.execute("INSERT INTO {schema}.index_generation VALUES (10, 1, 8, 9, 'READY')")
    repository = StructuredQueryRepository(database.settings)
    filters = (ResolvedFilter(field="business_status", operator="eq", values=("EFFECTIVE",)),)
    count = await repository.execute(identity=identity(), scope=selected(filters, second=True),
                                     plan=plan())
    assert count.total == 1 and len(count.provenance.sources) == 1
    page = await repository.execute(
        identity=identity(), scope=selected(second=True),
        plan=plan("list_documents", offset=1, limit=1),
    )
    assert page.total == 2 and page.documents[0].document_id == "4"
    assert page.documents[0].revision_id == "6" and page.documents[0].file_type == "application/pdf"
    assert await repository.revalidate(
        identity=identity(), scope=selected(second=True), result=page,
    )


async def test_real_chat_snapshot_hides_entire_count_after_metadata_change(structured_database):
    database = structured_database
    repository = StructuredQueryRepository(database.settings)
    count = await repository.execute(identity=identity(), scope=selected(), plan=plan())
    session_id, run_id = await save_result(database, count)
    before = await database.repository.get_run(identity(), session_id, run_id)
    assert before["visibility"] == "visible" and before["response"]["structured"]["total"] == 1
    assert database.scalar("SELECT count(*) FROM {schema}.answer_citation") == 0
    database.execute("UPDATE {schema}.document SET document_code = 'CHANGED' WHERE id = 4")
    assert not await repository.revalidate(identity=identity(), scope=selected(), result=count)
    hidden = await database.repository.get_run(identity(), session_id, run_id)
    assert hidden["visibility"] == "unavailable" and hidden["response"] is None


async def test_real_zero_result_anchor_hides_chat_after_kb_revocation(structured_database):
    database = structured_database
    filters = (ResolvedFilter(field="business_status", operator="eq", values=("REPEALED",)),)
    repository = StructuredQueryRepository(database.settings)
    zero = await repository.execute(identity=identity(), scope=selected(filters), plan=plan())
    assert zero.total == 0 and zero.provenance.sources == ()
    session_id, run_id = await save_result(database, zero)
    visible = await database.repository.get_run(identity(), session_id, run_id)
    assert visible["visibility"] == "visible"
    database.execute("DELETE FROM {schema}.knowledge_base_grant")
    assert not await repository.revalidate(
        identity=identity(), scope=selected(filters), result=zero,
    )
    hidden = await database.repository.get_run(identity(), session_id, run_id)
    assert hidden["visibility"] == "unavailable" and hidden["response"] is None
