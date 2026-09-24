"""Knowledge base ORM entities."""

# Register tenant and department tables referenced by composite foreign keys.
from zhiqing_rag.modules import identity as _identity  # noqa: F401

from .knowledge_base import KnowledgeBase
from .knowledge_base_grant import KnowledgeBaseGrant

__all__ = [
    "KnowledgeBase",
    "KnowledgeBaseGrant",
]
