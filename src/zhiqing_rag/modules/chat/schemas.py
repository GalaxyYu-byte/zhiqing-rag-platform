"""会话 HTTP 协议：历史由服务器加载，客户端只提交问题、范围和幂等键。"""

import json
from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from zhiqing_rag.query_analysis.models import StrictModel
from zhiqing_rag.query_answering.models import QueryAnswerResponse
from zhiqing_rag.query_routing.models import PlanQueryBody, PublicId


class CreateSessionBody(StrictModel):
    title: str = Field(default="新会话", min_length=1, max_length=300)


class ChatAnswerBody(PlanQueryBody):
    client_request_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class ChatSessionView(StrictModel):
    id: PublicId
    title: str
    status: Literal["ACTIVE", "ARCHIVED", "DELETED"]
    last_active_at: datetime
    created_at: datetime


class SessionList(StrictModel):
    sessions: list[ChatSessionView]


class ChatRunView(StrictModel):
    id: PublicId
    session_id: PublicId
    client_request_id: str
    question: str
    standalone_question: str | None = None
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "ABANDONED"]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    response: QueryAnswerResponse | None = None
    visibility: Literal["visible", "unavailable", "pending"]

    @field_validator("response", mode="before")
    @classmethod
    def restore_json_response(cls, value):
        # 快照来自 JSONB，严格协议的 tuple 必须从 JSON 入口恢复。
        if isinstance(value, dict):
            return QueryAnswerResponse.model_validate_json(json.dumps(value, ensure_ascii=False))
        return value


class RunPage(StrictModel):
    runs: list[ChatRunView]
    has_more: bool
    next_before_run_id: PublicId | None
