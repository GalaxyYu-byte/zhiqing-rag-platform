"""检索与图谱投影 ORM 实体。"""

# 注册复合外键引用的实体表。
from zhiqing_rag.modules import documents as _documents  # noqa: F401
from zhiqing_rag.modules import identity as _identity  # noqa: F401

from .graph_projection import GraphProjection

__all__ = [
    "GraphProjection",
]
