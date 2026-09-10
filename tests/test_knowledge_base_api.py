from datetime import datetime

import pytest

from zq_rag_app.api.knowledge_base import get_knowledge_bases
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
