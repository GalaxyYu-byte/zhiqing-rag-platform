"""检索与图谱投影 ORM 实体。"""

# 注册复合外键引用的实体表。
from zhiqing_rag.modules import documents as _documents  # noqa: F401
from zhiqing_rag.modules import identity as _identity  # noqa: F401

from .chunk_access import authorized_chunk_statement
from .graph_access import active_graph_projection, authorized_graph_projection
from .graph_projection import GraphProjection

__all__ = [
    "GraphProjection",
    "active_graph_projection",
    "authorized_graph_projection",
    "authorized_chunk_statement",
]
