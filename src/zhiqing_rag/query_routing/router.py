"""已登录计划接口；先确认资源范围，再调用 Analyzer 和 Router。"""

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Response

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.modules.identity.router import IdentityDependency
from zhiqing_rag.query_analysis.models import QueryAnalysisRequest
from zhiqing_rag.query_analysis.router import get_analyzer
from zhiqing_rag.query_analysis.service import QueryAnalyzerService

from .models import PlanQueryBody, QueryPlanningResponse
from .scope import AccessScope, QueryScopeRepository
from .service import QueryRouterService

router = APIRouter(prefix="/api/query", tags=["查询计划"])


def get_access_scope(
    body: PlanQueryBody,
    identity: IdentityDependency,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AccessScope:
    # 此同步依赖在线程池运行，连接在模型请求前关闭，不持有长事务。
    try:
        with database_connection(settings) as connection:
            return QueryScopeRepository(connection, settings.db_schema).resolve(identity, body)
    except psycopg.Error as error:
        raise AuthError(503, "DATABASE_UNAVAILABLE", "数据库暂不可用，请稍后重试") from error


def get_query_router() -> QueryRouterService:
    return QueryRouterService()


@router.post("/plan", response_model=QueryPlanningResponse)
async def plan_query(
    body: PlanQueryBody,
    response: Response,
    scope: Annotated[AccessScope, Depends(get_access_scope)],
    analyzer: Annotated[QueryAnalyzerService, Depends(get_analyzer)],
    planner: Annotated[QueryRouterService, Depends(get_query_router)],
):
    response.headers["Cache-Control"] = "no-store"
    request = QueryAnalysisRequest(query=body.query)
    analysis = await analyzer.analyze(request)
    plan = planner.route(request=request, analysis=analysis, scope=scope)
    return QueryPlanningResponse(analysis=analysis, plan=plan, scope=scope.summary())
