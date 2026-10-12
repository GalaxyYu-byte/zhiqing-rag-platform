"""统一查询执行分派；目录结果与原文证据分别返回。"""

from dataclasses import replace
from time import perf_counter

from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_retrieval.models import RetrievalResult, RetrievalTrace
from zhiqing_rag.query_routing.models import StructuredPlan


class QueryExecutionService:
    def __init__(self, retriever, structured_service=None):
        self.retriever = retriever
        self.structured_service = structured_service or getattr(
            retriever, "structured_service", None
        )

    async def execute(self, *, identity, scope, plan):
        if (identity.tenant_id, identity.user_id) != (scope.tenant_id, scope.user_id):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "查询范围与当前身份不一致")
        if plan.resolved_filters:
            scope = replace(scope, resolved_filters=plan.resolved_filters)
        if isinstance(plan, StructuredPlan):
            if plan.planning_status != "planned":
                return RetrievalResult(status=plan.planning_status, message=plan.message), None
            if self.structured_service is None:
                return RetrievalResult(status="unsupported", message="目录查询执行器尚未接入"), None
            started = perf_counter()
            result = await self.structured_service.execute(
                identity=identity, scope=scope, plan=plan
            )
            return RetrievalResult(
                status="skipped",
                message="已执行精确目录查询",
                trace=RetrievalTrace(total_ms=max(0, round((perf_counter() - started) * 1000))),
            ), result
        return await self.retriever.execute(identity=identity, scope=scope, plan=plan), None


def structured_answer_text(result) -> str:
    prefix = "当前可读的已发布文档中，符合条件的共有"
    if result.operation == "count_documents":
        return f"{prefix} {result.total} 份。"
    if not result.documents:
        return f"{prefix} {result.total} 份，本页没有文档。"
    first = result.offset + 1
    last = result.offset + len(result.documents)
    return f"{prefix} {result.total} 份，当前展示第 {first}—{last} 份。"
