"""SQLAlchemy ORM 实体模型集合。

这里集中导出项目中的所有 ORM 实体。数据库初始化、迁移工具或其他需要
访问 ``Base.metadata`` 的代码只要导入本模块，就能让 SQLAlchemy 注册全部
业务表，避免因为遗漏模型导入而导致表没有出现在元数据中的问题。
"""

from .chat import AnswerFeedback, ChatMessage, ChatSession
from .document import DocChunk, Document, IndexTask
from .evaluation import EvalDataset, EvalResult
from .knowledge_base import KnowledgeBase, Permission

__all__ = [
    "AnswerFeedback",
    "ChatMessage",
    "ChatSession",
    "DocChunk",
    "Document",
    "EvalDataset",
    "EvalResult",
    "IndexTask",
    "KnowledgeBase",
    "Permission",
]
