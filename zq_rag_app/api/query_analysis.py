"""独立 Query Analyzer 接口，不读取或修改知识库和聊天会话。"""

from fastapi import APIRouter

from ..core.security import CurrentUser
from ..query_analysis.models import QueryAnalysisRequest, QueryAnalysisResult
from ..services.query_analyzer_service import QueryAnalyzerService


router = APIRouter(prefix="/query", tags=["query-analysis"])


@router.post("/analyze", response_model=QueryAnalysisResult)
async def analyze_query(
    request: QueryAnalysisRequest, current_user: CurrentUser,
) -> QueryAnalysisResult:
    service = QueryAnalyzerService()
    try:
        return await service.analyze(request)
    finally:
        await service.aclose()
