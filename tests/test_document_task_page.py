"""分页接口、日期时区和服务端统一解析配置。"""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from zhiqing_rag.core.config import Settings
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.documents.repository import UploadRepository
from zhiqing_rag.modules.documents.router import get_upload_identity, get_upload_service
from zhiqing_rag.modules.documents.schemas import DocumentTaskPage, UploadIdentity


@pytest.fixture
def page_api():
    calls = []
    identity = UploadIdentity(1, 2, 3, None, 2)

    def paginate(current, **query):
        assert current == identity
        calls.append(query)
        return DocumentTaskPage(
            items=[],
            total=0,
            overall_total=80,
            overall_size=1000,
            counts={"all": 0},
            offset=query["offset"],
            limit=query["limit"],
        )

    service = SimpleNamespace(
        repository=SimpleNamespace(paginate_tasks=paginate),
        settings=Settings(_env_file=None, upload_chunk_size=256, upload_chunk_overlap=32),
    )
    app = create_app()
    app.dependency_overrides[get_upload_service] = lambda: service
    app.dependency_overrides[get_upload_identity] = lambda: identity
    with TestClient(app) as client:
        yield client, calls


def test_pagination_query_preserves_all_filters_and_includes_entire_end_day(page_api):
    client, calls = page_api
    response = client.get(
        "/api/document-tasks/page",
        params={
            "offset": 20,
            "limit": 10,
            "knowledge_base_id": 11,
            "search": " 手册 ",
            "status": "done",
            "date_from": "2026-09-25",
            "date_to": "2026-09-27",
        },
    )
    assert response.status_code == 200 and response.json()["overall_total"] == 80
    query = calls[0]
    assert query["offset"] == 20 and query["limit"] == 10
    assert query["knowledge_base_id"] == 11 and query["search"] == "手册"
    assert query["status"] == "done"
    assert query["start"].astimezone(UTC) == datetime(2026, 9, 24, 16, tzinfo=UTC)
    assert query["end"].astimezone(UTC) == datetime(2026, 9, 27, 16, tzinfo=UTC)


@pytest.mark.parametrize(
    "params",
    [
        {"offset": -1},
        {"limit": 0},
        {"limit": 101},
        {"knowledge_base_id": 0},
        {"status": "unknown"},
        {"date_from": "invalid"},
        {"date_from": "2026-09-28", "date_to": "2026-09-27"},
    ],
)
def test_invalid_pagination_filters_do_not_query_repository(page_api, params):
    client, calls = page_api
    assert client.get("/api/document-tasks/page", params=params).status_code == 422
    assert not calls


def test_read_only_settings_come_from_server(page_api):
    response = page_api[0].get("/api/document-upload-settings")
    assert response.status_code == 200
    assert response.json() == {"chunk_size": 256, "chunk_overlap": 32}


def test_invalid_server_overlap_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, upload_chunk_size=256, upload_chunk_overlap=128)


def test_repository_limits_rows_and_keeps_tenant_clearance_and_grant_scope():
    calls = []
    summary = {
        "items": [],
        "total": 0,
        "overall_total": 80,
        "overall_size": 1000,
        "all_count": 0,
        "done_count": 0,
        "processing_count": 0,
        "error_count": 0,
        "cancelled_count": 0,
    }
    repository = UploadRepository(None, "test")

    def query(statement, params):
        calls.append((statement, params))
        return SimpleNamespace(fetchone=lambda: summary)

    repository.query = query
    result = repository.paginate_tasks(
        UploadIdentity(1, 2, 3, 4, 5),
        offset=60,
        limit=10,
        knowledge_base_id=11,
        search="%_",
        status="all",
        start=None,
        end=None,
    )
    statement, params = calls[0]
    assert "LIMIT %s OFFSET %s" in statement
    assert "task.tenant_id = %s" in statement and "doc.confidentiality <= %s" in statement
    assert "knowledge_base_grant" in statement and "g.user_id = %s" in statement
    assert params[:4] == (1, 5, 3, 4) and params[-2:] == (10, 60)
    assert "strpos(lower(original_filename)" in statement
    assert result.overall_total == 80 and not result.items
