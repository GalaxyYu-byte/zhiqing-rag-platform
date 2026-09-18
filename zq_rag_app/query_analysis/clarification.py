"""会话待澄清任务；保存用户意图和授权范围，不保存助手事实。"""

from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import HistoryMessage, _StrictModel


class ClarificationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # 仅为可选择的文档对象，不能被当作事实证据或检索结果。
    doc_id: int = Field(gt=0)


class PendingClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    original_query: str = Field(min_length=1, max_length=2000)
    missing_slots: list[str] = Field(min_length=1, max_length=10)
    question: str = Field(min_length=1, max_length=300)
    kb_ids: list[int] = Field(min_length=1)
    doc_ids: list[int] = Field(min_length=1)
    candidates: list[ClarificationCandidate] = Field(default_factory=list, max_length=50)
    supplements: list[str] = Field(default_factory=list, max_length=6)
    user_context: list[HistoryMessage] = Field(default_factory=list, max_length=6)
    expires_at: datetime

    @model_validator(mode="after")
    def scoped_candidates(self):
        if self.expires_at.tzinfo is None:
            raise ValueError("澄清状态必须包含带时区的过期时间")
        if any(value <= 0 for value in [*self.kb_ids, *self.doc_ids]):
            raise ValueError("澄清授权范围无效")
        if any(candidate.doc_id not in self.doc_ids for candidate in self.candidates):
            raise ValueError("候选对象必须属于澄清授权范围")
        if any(not text.strip() or len(text) > 2000 for text in self.supplements):
            raise ValueError("澄清补充无效")
        if any(message.role != "user" for message in self.user_context):
            raise ValueError("待澄清任务不能保存历史助手回答")
        return self

    def revalidate_scope(self, kb_ids: list[int], doc_ids: list[int]):
        """调用方必须先用当前用户权限重新计算范围；旧快照不能授予权限。"""
        if self.expires_at <= datetime.now(timezone.utc) or set(self.kb_ids) != set(kb_ids):
            return None
        allowed = sorted(set(self.doc_ids).intersection(doc_ids))
        if not allowed:
            return None
        return self.model_copy(update={
            "doc_ids": allowed,
            "candidates": [candidate for candidate in self.candidates if candidate.doc_id in allowed],
        })

    @classmethod
    def create(cls, *, query: str, question: str, reason: str, kb_ids: list[int], doc_ids: list[int],
               history: list[HistoryMessage] | None = None):
        slot = ("entity_disambiguation" if reason == "graph_entity_ambiguous" else
                "context_reference" if "context" in reason else "required_information")
        return cls(
            original_query=query, missing_slots=[slot], question=question,
            kb_ids=kb_ids, doc_ids=doc_ids,
            candidates=[ClarificationCandidate(doc_id=value) for value in doc_ids[:50]],
            user_context=[message for message in (history or []) if message.role == "user"][-6:],
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
        )


class ClarificationCompletion(_StrictModel):
    status: Literal["completed", "clarify", "new_task"]
    query: str | None = Field(min_length=1, max_length=2000)
    clarification_question: str | None = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def consistent_status(self):
        if self.status == "clarify":
            if self.query is not None or self.clarification_question is None:
                raise ValueError("未补全不能生成检索问题")
        elif self.query is None or self.clarification_question is not None:
            raise ValueError("补全状态不一致")
        return self
