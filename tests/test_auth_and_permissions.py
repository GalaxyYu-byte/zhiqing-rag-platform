import pytest
from fastapi import HTTPException

from zq_rag_app.api.auth import get_current_user
from zq_rag_app.core.security import (
    DEFAULT_ADMIN_USER,
    UserContext,
    clear_user_context,
    get_user_context,
    require_current_user,
    set_user_context,
)
from zq_rag_app.services.permission_service import (
    PermissionLevel,
    has_knowledge_base_permission,
    list_accessible_document_ids,
)


class _FakeSession:
    def __init__(self, scalar_value=None):
        self.scalar_value = scalar_value
        self.statement = None
        self.call_count = 0

    async def scalar(self, statement):
        self.statement = statement
        self.call_count += 1
        return self.scalar_value


def test_user_context_can_be_bound_and_restored():
    assert get_user_context() is None
    token = set_user_context(DEFAULT_ADMIN_USER)
    try:
        assert require_current_user() == DEFAULT_ADMIN_USER
    finally:
        clear_user_context(token)
    assert get_user_context() is None


def test_original_user_context_call_style_remains_supported():
    token = set_user_context(9, "DEV", "USER", username="reader")
    try:
        assert require_current_user().username == "reader"
    finally:
        clear_user_context(token)
    assert get_user_context() is None


def test_require_current_user_rejects_missing_context():
    with pytest.raises(HTTPException) as exc_info:
        require_current_user()
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_current_user_api_returns_bound_user():
    response = await get_current_user(DEFAULT_ADMIN_USER)
    assert response.user_id == 1
    assert response.username == "admin"
    assert response.role == "ADMIN"


@pytest.mark.asyncio
async def test_admin_permission_is_allowed_without_database_query():
    session = _FakeSession()
    allowed = await has_knowledge_base_permission(
        session,
        user=DEFAULT_ADMIN_USER,
        kb_id=999,
        required=PermissionLevel.WRITE,
    )
    assert allowed is True
    assert session.call_count == 0


@pytest.mark.asyncio
async def test_regular_user_permission_uses_database_access_rules():
    session = _FakeSession(scalar_value=7)
    user = UserContext(
        user_id=9,
        username="reader",
        department_id="DEV",
        role="USER",
    )
    allowed = await has_knowledge_base_permission(
        session,
        user=user,
        kb_id=7,
        required=PermissionLevel.READ,
    )
    assert allowed is True
    statement = str(session.statement)
    assert "kb_knowledge_base.is_public IS true" in statement
    assert "kb_permission.permission IN" in statement


class _RowsResult:
    def all(self):
        return [(7,), (9,)]


class _RowsSession:
    def __init__(self):
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _RowsResult()


@pytest.mark.asyncio
async def test_document_access_filter_is_applied_before_retrieval():
    session = _RowsSession()
    user = UserContext(
        user_id=9,
        username="public-reader",
        department_id="DEV",
        role="USER",
        clearance="内部公开",
    )

    doc_ids = await list_accessible_document_ids(
        session,
        user=user,
        kb_ids=[4],
        requested_doc_ids=[7, 9, 10],
    )

    assert doc_ids == [7, 9]
    sql = str(session.statement)
    assert "kb_document.confidentiality" in sql
    assert "kb_document.department_id" not in sql
    assert "kb_document.id IN" in sql


@pytest.mark.asyncio
async def test_department_clearance_adds_same_department_scope():
    session = _RowsSession()
    user = UserContext(
        user_id=10,
        username="department-reader",
        department_id="AI平台部",
        role="USER",
        clearance="部门内部",
    )

    await list_accessible_document_ids(session, user=user, kb_ids=[4])

    sql = str(session.statement)
    assert "kb_document.department_id" in sql
    assert "kb_document.confidentiality" in sql
