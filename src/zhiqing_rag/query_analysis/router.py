"""已登录用户的问题分析调试接口；不执行检索或读取会话历史。"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.modules.identity.router import IdentityDependency

from .models import AnalyzeQueryBody, QueryAnalysisRequest, QueryAnalysisResult
from .service import QueryAnalyzerService

router = APIRouter(prefix="/api/query", tags=["问题分析"])


def get_analyzer(settings: Annotated[Settings, Depends(get_settings)]) -> QueryAnalyzerService:
    return QueryAnalyzerService(settings)


@router.post("/analyze", response_model=QueryAnalysisResult)
async def analyze_query(
    body: AnalyzeQueryBody,
    response: Response,
    identity: IdentityDependency,
    analyzer: Annotated[QueryAnalyzerService, Depends(get_analyzer)],
):
    response.headers["Cache-Control"] = "no-store"
    return await analyzer.analyze(QueryAnalysisRequest(query=body.query))
