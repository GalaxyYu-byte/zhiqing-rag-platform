"""只执行服务端白名单目录计划，空目录是成功的精确结果。"""

from zhiqing_rag.core.config import Settings
from zhiqing_rag.query_routing.models import StructuredPlan

from .repository import StructuredQueryRepository


class StructuredQueryService:
    def __init__(self, settings: Settings | None = None, *, repository=None):
        if repository is None and settings is None:
            raise ValueError("统计服务需要配置或仓储")
        self.repository = (
            repository if repository is not None else StructuredQueryRepository(settings)
        )

    async def execute(self, *, identity, scope, plan):
        if not isinstance(plan, StructuredPlan) or plan.params.operation is None:
            raise ValueError("统计服务只接受白名单目录计划")
        if plan.planning_status == "unsupported":
            raise ValueError("不支持的计划不能执行统计")
        return await self.repository.execute(identity=identity, scope=scope, plan=plan)

    async def revalidate(self, *, identity, scope, result) -> bool:
        return await self.repository.revalidate(identity=identity, scope=scope, result=result)
