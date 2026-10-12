"""登录用户的检索入口：只接受问题和资源选择，计划由服务器生成。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.identity.router import IdentityDependency
from zhiqing_rag.query_analysis.models import QueryAnalysisRequest
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_analysis.service import QueryAnalyzerService
from zhiqing_rag.query_execution import QueryExecutionService
from zhiqing_rag.query_routing.models import PlanQueryBody
from zhiqing_rag.query_routing.planning import get_graph_service, plan_with_graph
from zhiqing_rag.query_routing.router import get_access_scope, get_query_router
from zhiqing_rag.query_routing.scope import AccessScope
from zhiqing_rag.query_routing.service import QueryRouterService
from zhiqing_rag.query_structured.repository import StructuredQueryRepository
from zhiqing_rag.query_structured.service import StructuredQueryService

from .graph import OnlineGraphService
from .models import QuerySearchResponse
from .providers import QueryEmbedder, QueryReranker
from .repository import QueryRetrievalRepository
from .service import HybridRetrievalService

router = APIRouter(prefix="/api/query", tags=["文档检索"])


def get_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
    graph_service: Annotated[OnlineGraphService, Depends(get_graph_service)],
) -> HybridRetrievalService:
    service = HybridRetrievalService(
        QueryRetrievalRepository(settings),
        QueryEmbedder(settings),
        QueryReranker(settings),
        reranker_enabled=settings.query_reranker_enabled,
        graph_service=graph_service,
        graph_timeout_seconds=settings.query_graph_timeout_seconds,
    )
    service.structured_service = StructuredQueryService(
        repository=StructuredQueryRepository(settings)
    )
    return service


@router.post("/search", response_model=QuerySearchResponse)
async def search_query(
    body: PlanQueryBody,
    response: Response,
    identity: IdentityDependency,
    scope: Annotated[AccessScope, Depends(get_access_scope)],
    analyzer: Annotated[QueryAnalyzerService, Depends(get_analyzer)],
    planner: Annotated[QueryRouterService, Depends(get_query_router)],
    retriever: Annotated[HybridRetrievalService, Depends(get_retriever)],
):
    response.headers["Cache-Control"] = "no-store"
    request = QueryAnalysisRequest(query=body.query)
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
    retrieval, structured = await QueryExecutionService(retriever).execute(
        plan=plan, scope=scope, identity=identity
    )
    return QuerySearchResponse(
        analysis=analysis,
        plan=plan,
        scope=scope.summary(),
        retrieval=retrieval,
        structured=structured,
        request=body,
    )
