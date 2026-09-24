"""评估与反馈 ORM 实体。"""

# 注册复合外键引用的实体表。
from zhiqing_rag.modules import chat as _chat  # noqa: F401
from zhiqing_rag.modules import documents as _documents  # noqa: F401
from zhiqing_rag.modules import identity as _identity  # noqa: F401
from zhiqing_rag.modules import knowledge as _knowledge  # noqa: F401

from .answer_feedback import AnswerFeedback
from .evaluation_case import EvaluationCase
from .evaluation_dataset import EvaluationDataset
from .evaluation_result import EvaluationResult
from .evaluation_run import EvaluationRun

__all__ = [
    "AnswerFeedback",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationResult",
    "EvaluationRun",
]
