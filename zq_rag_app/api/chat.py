"""带会话持久化、权限校验和证据引用的 RAG 问答接口。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db
from ..core.security import CurrentUser
from ..models.chat import ChatMessage, ChatSession
from ..query_analysis.models import HistoryMessage
from ..query_analysis.clarification import PendingClarification
from ..query_analysis.rewrite import RewriteSummary
from ..query_analysis.multi_query import MultiQuerySummary
from ..query_routing.models import ExecutionRecord, RoutingDecision
from ..services.permission_service import (
    PermissionLevel,
    has_knowledge_base_permission,
    list_accessible_document_ids,
)
from ..services.rag_service import RagAnswerService


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatAnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    kb_ids: list[int] = Field(min_length=1, max_length=50)
    session_id: str | None = Field(default=None, max_length=36)
    doc_ids: list[int] | None = Field(default=None, min_length=1, max_length=200)
    candidate_k: int = Field(default=30, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query 不能为空")
        return normalized

    @field_validator("kb_ids", "doc_ids")
    @classmethod
    def normalize_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(item <= 0 for item in value):
            raise ValueError("知识库和文档 ID 必须大于 0")
        return list(dict.fromkeys(value))

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(UUID(value))
        except ValueError as exc:
            raise ValueError("session_id 必须是有效 UUID") from exc


class ChatSourceResponse(BaseModel):
    citation_id: str
    rank: int
    chunk_id: int
    doc_id: int
    document: str
    chunk_index: int
    section: str | None
    page: int | None
    excerpt: str
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    graph_rank: int | None


class ChatTimingResponse(BaseModel):
    retrieval_ms: int
    generation_ms: int
    total_ms: int
    analysis_ms: int = 0
    routing_ms: int = 0
    rewrite_ms: int = 0
    expansion_ms: int = 0


class ChatAnswerResponse(BaseModel):
    session_id: str
    message_id: int
    query: str
    answer: str
    model: str
    engine: str
    token_count: int
    graph_degraded: bool
    reranked: bool
    sources: list[ChatSourceResponse]
    timing: ChatTimingResponse
    routing: RoutingDecision | None = None
    rewrite: RewriteSummary | None = None
    multi_query: MultiQuerySummary | None = None
    execution: ExecutionRecord | None = None


@router.post("/answer", response_model=ChatAnswerResponse)
async def answer_question(
    request: ChatAnswerRequest,
    current_user: CurrentUser,
    session: AsyncSession = Depends(get_db),
) -> ChatAnswerResponse:
    if request.top_k > request.candidate_k:
        raise HTTPException(status_code=400, detail="top_k 不能大于 candidate_k")
    for kb_id in request.kb_ids:
        allowed = await has_knowledge_base_permission(
            session,
            user=current_user,
            kb_id=kb_id,
            required=PermissionLevel.READ,
        )
        if not allowed:
            raise HTTPException(status_code=403, detail=f"无权读取知识库 {kb_id}")

    accessible_doc_ids = await list_accessible_document_ids(
        session,
        user=current_user,
        kb_ids=request.kb_ids,
        requested_doc_ids=request.doc_ids,
    )
    if request.doc_ids is not None and not set(request.doc_ids).issubset(
        accessible_doc_ids
    ):
        raise HTTPException(status_code=403, detail="请求包含无权读取的文档")
    if not accessible_doc_ids:
        raise HTTPException(status_code=403, detail="当前范围内没有可读取文档")

    chat_session: ChatSession | None = None
    if request.session_id is not None:
        chat_session = await session.scalar(
            select(ChatSession).where(
                ChatSession.id == request.session_id,
                ChatSession.user_id == current_user.user_id,
                ChatSession.is_deleted.is_(False),
            ).with_for_update()
        )
        if chat_session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        if set(json.loads(chat_session.kb_ids)) != set(request.kb_ids):
            raise HTTPException(status_code=409, detail="会话知识库范围不能变更")

    history: list[HistoryMessage] = []
    pending = None
    if chat_session is not None:
        stored = getattr(chat_session, "pending_clarification", None)
        if stored is not None:
            try:
                pending = PendingClarification.model_validate(stored).revalidate_scope(
                    request.kb_ids, accessible_doc_ids,
                )
            except ValueError:
                logger.warning("会话待澄清状态无效，已丢弃")
        if stored is None:
            # 普通会话仅加载用户消息作为指代依据。待澄清任务用自己的用户上下文，
            # 失效状态也不能借普通历史再次恢复，绕过过期时间或原授权范围。
            messages = (await session.execute(
                select(ChatMessage).where(
                    ChatMessage.session_id == chat_session.id, ChatMessage.role == "USER",
                ).order_by(ChatMessage.id.desc()).limit(6)
            )).scalars().all()
            history = [HistoryMessage(role="user", content=message.content) for message in reversed(messages)]

    service = RagAnswerService()
    try:
        result = await service.answer(
            session,
            query=request.query,
            kb_ids=request.kb_ids,
            doc_ids=accessible_doc_ids,
            candidate_k=request.candidate_k,
            top_k=request.top_k,
            rerank=request.rerank,
            history=history,
            pending_clarification=pending,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAG 回答生成失败")
        raise HTTPException(status_code=503, detail="问答服务暂时不可用") from exc
    finally:
        await service.aclose()

    if chat_session is None:
        chat_session = ChatSession(
            id=str(uuid4()),
            user_id=current_user.user_id,
            kb_ids=json.dumps(request.kb_ids, separators=(",", ":")),
            title=request.query[:200],
            message_count=0,
            is_deleted=False,
        )
        session.add(chat_session)

    chat_session.pending_clarification = (
        result.pending_clarification.model_dump(mode="json")
        if result.pending_clarification is not None else None
    )
    user_message = ChatMessage(
        session_id=chat_session.id,
        role="USER",
        content=request.query,
        sources=None,
        token_count=0,
        latency_ms=0,
    )
    assistant_message = ChatMessage(
        session_id=chat_session.id,
        role="ASSISTANT",
        content=result.answer,
        sources=[source.to_storage() for source in result.sources],
        retrieval_trace=json.loads(json.dumps({
            "routing": asdict(result.routing) if result.routing else None,
            "execution": asdict(result.execution) if result.execution else None,
            "rewrite": asdict(result.rewrite) if result.rewrite else None,
            "multi_query": asdict(result.multi_query) if result.multi_query else None,
        }, ensure_ascii=False)),
        token_count=result.token_count,
        latency_ms=result.timing.total_ms,
    )
    chat_session.message_count = int(chat_session.message_count or 0) + 2
    chat_session.last_active_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.add_all([user_message, assistant_message])
    try:
        await session.commit()
        await session.refresh(assistant_message)
    except Exception:
        await session.rollback()
        raise

    return ChatAnswerResponse(
        session_id=chat_session.id,
        message_id=assistant_message.id,
        query=result.query,
        answer=result.answer,
        model=result.model,
        engine=result.engine,
        token_count=result.token_count,
        graph_degraded=result.graph_degraded,
        reranked=result.reranked,
        sources=[ChatSourceResponse(**asdict(source)) for source in result.sources],
        timing=ChatTimingResponse(
            retrieval_ms=result.timing.retrieval_ms,
            generation_ms=result.timing.generation_ms,
            total_ms=result.timing.total_ms,
            analysis_ms=result.timing.analysis_ms,
            routing_ms=result.timing.routing_ms,
            rewrite_ms=result.timing.rewrite_ms,
            expansion_ms=result.timing.expansion_ms,
        ),
        routing=result.routing,
        rewrite=result.rewrite,
        multi_query=result.multi_query,
        execution=result.execution,
    )
