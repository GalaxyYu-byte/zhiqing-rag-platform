"""单轮回答入口只接受问题和资源选择，证据必须由当前检索链取得。"""

from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Depends, Response
from sse_starlette.sse import EventSourceResponse

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.modules.identity.router import IdentityDependency
from zhiqing_rag.query_analysis.models import QueryAnalysisRequest
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_analysis.service import QueryAnalyzerService
from zhiqing_rag.query_execution import QueryExecutionService, structured_answer_text
from zhiqing_rag.query_retrieval.repository import QueryRetrievalRepository
from zhiqing_rag.query_retrieval.router import get_retriever
from zhiqing_rag.query_retrieval.service import HybridRetrievalService
from zhiqing_rag.query_routing.models import PlanQueryBody
from zhiqing_rag.query_routing.planning import plan_with_graph
from zhiqing_rag.query_routing.router import get_access_scope, get_query_router
from zhiqing_rag.query_routing.scope import AccessScope
from zhiqing_rag.query_routing.service import QueryRouterService

from .models import AnswerProgress, AnswerResult, QueryAnswerResponse
from .providers import AnswerProvider
from .service import QueryAnswerService
from .streaming import answer_events

router = APIRouter(prefix="/api/query", tags=["证据回答"])


def get_answerer(settings: Annotated[Settings, Depends(get_settings)]) -> QueryAnswerService:
    return QueryAnswerService(
        QueryRetrievalRepository(settings), AnswerProvider(settings), settings
    )


@router.post("/answer", response_model=QueryAnswerResponse)
async def answer_query(
    body: PlanQueryBody,
    response: Response,
    identity: IdentityDependency,
    scope: Annotated[AccessScope, Depends(get_access_scope)],
    analyzer: Annotated[QueryAnalyzerService, Depends(get_analyzer)],
    planner: Annotated[QueryRouterService, Depends(get_query_router)],
    retriever: Annotated[HybridRetrievalService, Depends(get_retriever)],
    answerer: Annotated[QueryAnswerService, Depends(get_answerer)],
):
    response.headers["Cache-Control"] = "no-store"
    return await execute_answer(body, identity, scope, analyzer, planner, retriever, answerer)


async def execute_answer(
    body: PlanQueryBody,
    identity: AuthIdentity,
    scope: AccessScope,
    analyzer: QueryAnalyzerService,
    planner: QueryRouterService,
    retriever: HybridRetrievalService,
    answerer: QueryAnswerService,
    on_progress: AnswerProgress | None = None,
) -> QueryAnswerResponse:
    request = QueryAnalysisRequest(query=body.query)
    if on_progress is not None:
        await on_progress("analysis")
    analysis = await analyzer.analyze(request)
    plan = await plan_with_graph(
        request=request,
        analysis=analysis,
        identity=identity,
        scope=scope,
        planner=planner,
        graph_service=getattr(retriever, "graph_service", None),
        filters=body.filters,
        offset=body.offset,
        limit=body.limit,
    )
    if on_progress is not None:
        await on_progress("retrieval")
    executor = QueryExecutionService(retriever)
    retrieval, structured = await executor.execute(plan=plan, scope=scope, identity=identity)
    if structured is not None:
        execution_scope = (
            replace(scope, resolved_filters=plan.resolved_filters)
            if plan.resolved_filters
            else scope
        )
        if not await executor.structured_service.revalidate(
            identity=identity, scope=execution_scope, result=structured
        ):
            raise AuthError(409, "STRUCTURED_SCOPE_CHANGED", "资料范围或属性已变化，请重新查询。")
        answer = AnswerResult(
            status="structured", message=structured_answer_text(structured), review_status="passed"
        )
        return QueryAnswerResponse(
            analysis=analysis,
            plan=plan,
            scope=scope.summary(),
            retrieval=retrieval,
            structured=structured,
            answer=answer,
            request=body,
        )
    if plan.resolved_filters:
        scope = replace(scope, resolved_filters=plan.resolved_filters)
    retrieval, answer = await answerer.execute(
        identity=identity,
        scope=scope,
        plan=plan,
        retrieval=retrieval,
        on_progress=on_progress,
    )
    return QueryAnswerResponse(
        analysis=analysis,
        plan=plan,
        scope=scope.summary(),
        retrieval=retrieval,
        answer=answer,
        request=body,
    )


@router.post(
    "/answer/stream",
    response_class=EventSourceResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_answer_query(
    body: PlanQueryBody,
    identity: IdentityDependency,
    scope: Annotated[AccessScope, Depends(get_access_scope)],
    analyzer: Annotated[QueryAnalyzerService, Depends(get_analyzer)],
    planner: Annotated[QueryRouterService, Depends(get_query_router)],
    retriever: Annotated[HybridRetrievalService, Depends(get_retriever)],
    answerer: Annotated[QueryAnswerService, Depends(get_answerer)],
):
    # 鉴权、资源范围和参数校验在发送 HTTP 头之前完成，沿用 JSON 错误及刷新策略。
    async def work(on_progress: AnswerProgress):
        return await execute_answer(
            body, identity, scope, analyzer, planner, retriever, answerer, on_progress
        )

    return EventSourceResponse(
        answer_events(work),
        headers={"Cache-Control": "no-store"},
        ping=10,
        send_timeout=15,
    )
