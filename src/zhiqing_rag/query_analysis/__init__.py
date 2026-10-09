"""问题分析契约；特征交给后续 Router，不能授予权限或执行查询。"""

from .models import QueryAnalysisRequest, QueryAnalysisResult
from .service import QueryAnalyzerService

__all__ = ["QueryAnalysisRequest", "QueryAnalysisResult", "QueryAnalyzerService"]
