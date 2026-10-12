"""会话管理与多轮回答入口，发送流头前完成身份、范围和幂等领取检查。"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Response
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.identity.router import IdentityDependency
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_analysis.service import QueryAnalyzerService
from zhiqing_rag.query_answering.router import get_answerer
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_retrieval.router import get_retriever
from zhiqing_rag.query_retrieval.service import HybridRetrievalService
from zhiqing_rag.query_routing.router import get_access_scope, get_query_router
from zhiqing_rag.query_routing.scope import AccessScope
from zhiqing_rag.query_routing.service import QueryRouterService

from .repository import ChatRepository
from .rewrite import QueryRewriteService
from .schemas import (
    ChatAnswerBody,
    ChatRunView,
    ChatSessionView,
    CreateSessionBody,
    RunPage,
    SessionList,
)
from .service import ChatService
from .streaming import chat_events

router = APIRouter(prefix="/api/chat", tags=["多轮会话"])
ResourceId = Annotated[int, Path(gt=0, le=2**63 - 1)]


def get_chat_repository(settings: Annotated[Settings, Depends(get_settings)]) -> ChatRepository:
    return ChatRepository(settings)


def get_rewriter(settings: Annotated[Settings, Depends(get_settings)]) -> QueryRewriteService:
    return QueryRewriteService(settings)


def get_chat_scope(
    body: ChatAnswerBody,
    identity: IdentityDependency,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AccessScope:
    # 使用同一请求模型，避免依赖用单轮模型重复解析后拒绝幂等字段。
    return get_access_scope(body, identity, settings)


def get_chat_service(
    repository: Annotated[ChatRepository, Depends(get_chat_repository)],
    rewriter: Annotated[QueryRewriteService, Depends(get_rewriter)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    return ChatService(repository, rewriter, settings)


RepositoryDependency = Annotated[ChatRepository, Depends(get_chat_repository)]
ServiceDependency = Annotated[ChatService, Depends(get_chat_service)]


async def prepare_chat(service, identity, session_id, body):
    run = await service.begin(identity, session_id, body)
    scope = None
    if run["is_new"]:
        try:
            scope = await run_in_threadpool(get_chat_scope, body, identity, service.settings)
        except (Exception, asyncio.CancelledError) as error:
            await service.fail_before_execution(identity, session_id, run, error)
            raise
    # 已提交轮次通过仓储的当前来源复核回放，不要求旧选择的资源仍可读。
    return run, scope


@router.post("/sessions", response_model=ChatSessionView, status_code=201)
async def create_session(
    body: CreateSessionBody,
    response: Response,
    identity: IdentityDependency,
    repository: RepositoryDependency,
):
    response.headers["Cache-Control"] = "no-store"
    return await repository.create_session(identity, body.title)


@router.get("/sessions", response_model=SessionList)
async def list_sessions(
    response: Response,
    identity: IdentityDependency,
    repository: RepositoryDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    response.headers["Cache-Control"] = "no-store"
    return SessionList(
        sessions=await repository.list_sessions(identity, limit=limit, offset=offset)
    )


@router.get("/sessions/{session_id}/runs", response_model=RunPage)
async def list_runs(
    session_id: ResourceId,
    response: Response,
    identity: IdentityDependency,
    repository: RepositoryDependency,
    before_run_id: Annotated[int | None, Query(gt=0, le=2**63 - 1)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    response.headers["Cache-Control"] = "no-store"
    return await repository.list_runs(
        identity, session_id, before_run_id=before_run_id, limit=limit
    )


@router.get("/sessions/{session_id}/runs/{run_id}", response_model=ChatRunView)
async def get_run(
    session_id: ResourceId,
    run_id: ResourceId,
    response: Response,
    identity: IdentityDependency,
    repository: RepositoryDependency,
):
    response.headers["Cache-Control"] = "no-store"
    return await repository.get_run(identity, session_id, run_id)


@router.post("/sessions/{session_id}/answer", response_model=ChatRunView)
async def answer_chat(
    session_id: ResourceId,
    body: ChatAnswerBody,
    response: Response,
    identity: IdentityDependency,
    service: ServiceDependency,
    analyzer: Annotated[QueryAnalyzerService, Depends(get_analyzer)],
    planner: Annotated[QueryRouterService, Depends(get_query_router)],
    retriever: Annotated[HybridRetrievalService, Depends(get_retriever)],
    answerer: Annotated[QueryAnswerService, Depends(get_answerer)],
):
    response.headers["Cache-Control"] = "no-store"
    run, scope = await prepare_chat(service, identity, session_id, body)
    return await service.execute(
        identity=identity,
        session_id=session_id,
        body=body,
        run=run,
        scope=scope,
        analyzer=analyzer,
        planner=planner,
        retriever=retriever,
        answerer=answerer,
    )


@router.post("/sessions/{session_id}/answer/stream", response_class=EventSourceResponse)
async def stream_chat(
    session_id: ResourceId,
    body: ChatAnswerBody,
    identity: IdentityDependency,
    service: ServiceDependency,
    analyzer: Annotated[QueryAnalyzerService, Depends(get_analyzer)],
    planner: Annotated[QueryRouterService, Depends(get_query_router)],
    retriever: Annotated[HybridRetrievalService, Depends(get_retriever)],
    answerer: Annotated[QueryAnswerService, Depends(get_answerer)],
):
    run, scope = await prepare_chat(service, identity, session_id, body)
    initial = ChatRunView.model_validate(
        {key: run[key] for key in ChatRunView.model_fields if key in run}
    ).model_dump(mode="json")
    initial["response"] = None
    initial["standalone_question"] = None

    async def work(on_progress):
        return await service.execute(
            identity=identity,
            session_id=session_id,
            body=body,
            run=run,
            scope=scope,
            analyzer=analyzer,
            planner=planner,
            retriever=retriever,
            answerer=answerer,
            on_progress=on_progress,
        )

    async def revalidate():
        return ChatRunView.model_validate(
            await service.repository.get_run(identity, session_id, int(run["id"]))
        )

    return EventSourceResponse(
        chat_events(work, initial, revalidate=revalidate),
        headers={"Cache-Control": "no-store"},
        ping=10,
        send_timeout=15,
    )
