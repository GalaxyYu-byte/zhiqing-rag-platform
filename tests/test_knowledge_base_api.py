from datetime import datetime

import pytest
from fastapi import HTTPException

from zq_rag_app.api.knowledge_base import (
    CreateKnowledgeBaseRequest,
    create_knowledge_base,
    get_knowledge_bases,
)
from zq_rag_app.core.security import DEFAULT_ADMIN_USER, UserContext
from zq_rag_app.models.knowledge_base import KnowledgeBase


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.statement = None

    async def execute(self, statement):
        self.statement = statement
        return _FakeResult(self.rows)


class _FakeCreateSession:
    def __init__(self, duplicate_id=None):
        self.duplicate_id = duplicate_id
        self.statement = None
        self.added = None
        self.committed = False

    async def scalar(self, statement):
        self.statement = statement
        return self.duplicate_id

    def add(self, knowledge_base):
        self.added = knowledge_base

    async def commit(self):
        self.committed = True

    async def refresh(self, knowledge_base):
        knowledge_base.id = 12
        knowledge_base.created_at = datetime(2026, 9, 10, 16, 0, 0)


@pytest.mark.asyncio
async def test_knowledge_base_list_uses_database_rows():
    knowledge_base = KnowledgeBase(
        id=7,
        name="真实知识库",
        description="用于测试",
        department_id="DEV",
        is_public=True,
        created_by=1,
        created_at=datetime(2026, 9, 9, 12, 0, 0),
        is_deleted=False,
    )
    session = _FakeSession([(knowledge_base, 3)])

    response = await get_knowledge_bases(keyword=None, session=session)

    assert response.total == 1
    assert response.items[0].id == 7
    assert response.items[0].name == "真实知识库"
    assert response.items[0].document_count == 3
    assert "kb_knowledge_base.is_deleted IS false" in str(session.statement)


@pytest.mark.asyncio
async def test_admin_can_create_knowledge_base_for_current_department():
    session = _FakeCreateSession()
    request = CreateKnowledgeBaseRequest(
        name="  新知识库  ",
        description="  产品资料  ",
        is_public=True,
    )

    response = await create_knowledge_base(
        request=request,
        current_user=DEFAULT_ADMIN_USER,
        session=session,
    )

    assert response.id == 12
    assert response.name == "新知识库"
    assert response.department_id == "ADMIN"
    assert response.created_by == DEFAULT_ADMIN_USER.user_id
    assert response.document_count == 0
    assert session.added.description == "产品资料"
    assert session.committed is True


@pytest.mark.asyncio
async def test_regular_user_cannot_create_for_another_department():
    session = _FakeCreateSession()
    user = UserContext(
        user_id=9,
        username="member",
        department_id="DEV",
        role="USER",
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_knowledge_base(
            request=CreateKnowledgeBaseRequest(
                name="跨部门知识库",
                department_id="HR",
            ),
            current_user=user,
            session=session,
        )

    assert exc_info.value.status_code == 403
    assert session.added is None


@pytest.mark.asyncio
async def test_create_rejects_duplicate_name_in_same_department():
    session = _FakeCreateSession(duplicate_id=3)

    with pytest.raises(HTTPException) as exc_info:
        await create_knowledge_base(
            request=CreateKnowledgeBaseRequest(name="已有知识库"),
            current_user=DEFAULT_ADMIN_USER,
            session=session,
        )

    assert exc_info.value.status_code == 409
    assert session.added is None
    assert session.committed is False
