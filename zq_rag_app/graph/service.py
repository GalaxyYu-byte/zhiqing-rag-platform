"""图抽取与持久化的应用服务边界。"""

from __future__ import annotations

from ..core.config import settings
from .extractor import GraphExtractor, QwenGraphExtractor
from .models import GraphChunkContext
from .repository import GraphRepository, GraphWriteResult
from .standardization import EntityStandardizer


class GraphIndexingService:
    """把模型抽取和 Neo4j 写入组合成单 Chunk 的幂等操作。"""

    def __init__(
        self,
        *,
        extractor: GraphExtractor | None = None,
        repository: GraphRepository | None = None,
        standardizer: EntityStandardizer | None = None,
        extractor_version: str = settings.graph_extractor_version,
    ) -> None:
        if not extractor_version.strip():
            raise ValueError("extractor_version 不能为空")
        self.extractor = extractor or QwenGraphExtractor()
        self.repository = repository or GraphRepository()
        self.standardizer = standardizer or EntityStandardizer()
        self.extractor_version = extractor_version.strip()

    async def ensure_schema(self) -> None:
        await self.repository.ensure_schema()

    async def index_chunk(self, context: GraphChunkContext) -> GraphWriteResult:
        extraction = await self.extractor.extract(context)
        # 抽取器和仓储都会校验证据；标准化只允许修改身份字段，不能改写引用。
        extraction.validate_evidence(context.content)
        extraction = self.standardizer.standardize(extraction)
        extraction.validate_evidence(context.content)
        return await self.repository.upsert_extraction(
            context=context,
            extraction=extraction,
            extractor_version=self.extractor_version,
        )
