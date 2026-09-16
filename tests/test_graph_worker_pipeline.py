import pytest

from zq_rag_app.graph.models import GraphExtraction
from zq_rag_app.graph.repository import GraphWriteResult
from zq_rag_app.graph.standardization import EntityStandardizer
from zq_rag_app.services.graph_task_service import (
    GraphTaskService,
    GraphTaskStage,
    _ChunkSnapshot,
    _GraphDocumentSnapshot,
)
from zq_rag_app.workers.graph_worker import WorkerSettings


def _duplicate_entity_extraction() -> GraphExtraction:
    return GraphExtraction.model_validate(
        {
            "schema_version": "1.0",
            "entities": [
                {
                    "local_id": "e1",
                    "entity_type": "SYSTEM",
                    "name": "Redis 缓存",
                    "evidence_quote": "Redis 缓存",
                },
                {
                    "local_id": "e2",
                    "entity_type": "SYSTEM",
                    "name": "Redis",
                    "evidence_quote": "Redis",
                },
                {
                    "local_id": "e3",
                    "entity_type": "SERVICE",
                    "name": "订单服务",
                    "evidence_quote": "订单服务",
                },
            ],
            "relations": [
                {
                    "source_local_id": "e3",
                    "target_local_id": "e1",
                    "predicate": "DEPENDS_ON",
                    "confidence": 0.9,
                    "evidence_quote": "订单服务依赖 Redis 缓存",
                },
                {
                    "source_local_id": "e3",
                    "target_local_id": "e2",
                    "predicate": "DEPENDS_ON",
                    "confidence": 0.8,
                    "evidence_quote": "订单服务依赖 Redis",
                },
            ],
        }
    )


def test_standardizer_canonicalizes_and_merges_local_entities() -> None:
    result = EntityStandardizer().standardize(_duplicate_entity_extraction())

    assert len(result.entities) == 2
    redis = next(entity for entity in result.entities if entity.name == "Redis")
    assert redis.aliases == ["Redis 缓存"]
    assert len(result.relations) == 1
    assert result.relations[0].target_local_id == redis.local_id


class _Repository:
    def __init__(self) -> None:
        self.calls = []

    async def ensure_schema(self) -> None:
        self.calls.append(("schema",))

    async def begin_document_version(self, **kwargs) -> None:
        self.calls.append(("begin", kwargs))

    async def activate_document_version(self, **kwargs) -> None:
        self.calls.append(("activate", kwargs))

    async def prune_inactive_document_versions(self, **kwargs) -> None:
        self.calls.append(("prune", kwargs))


class _IndexingService:
    def __init__(self, repository: _Repository) -> None:
        self.repository = repository
        self.contexts = []

    async def index_chunk(self, context) -> GraphWriteResult:
        self.contexts.append(context)
        return GraphWriteResult(
            chunk_uid=f"chunk:{context.chunk_index}",
            entity_count=2,
            claim_count=1,
        )


class _PipelineService(GraphTaskService):
    def __init__(self, indexing_service) -> None:
        super().__init__(
            session_factory=object(),  # type: ignore[arg-type]
            indexing_service=indexing_service,
            worker_id="graph-test-worker",
        )
        self.progress = []
        self.completed = False

    async def _load_chunks(self, document):
        del document
        return [
            _ChunkSnapshot(0, "订单服务依赖 Redis。", 1, "依赖"),
            _ChunkSnapshot(1, "平台团队负责订单服务。", 2, "负责人"),
        ]

    async def _set_stage(self, task_id, stage, progress_percent):
        self.progress.append((task_id, stage, progress_percent))

    async def _update_progress(self, task_id, **values):
        self.progress.append((task_id, values))

    async def _complete_task(self, task_id, document):
        del task_id, document
        self.completed = True


@pytest.mark.asyncio
async def test_pipeline_reads_real_chunks_then_switches_before_cleanup() -> None:
    repository = _Repository()
    indexing = _IndexingService(repository)
    service = _PipelineService(indexing)
    document = _GraphDocumentSnapshot(7, 4, 2, "运行手册.md")

    await service._run_pipeline(31, document)

    assert [context.chunk_index for context in indexing.contexts] == [0, 1]
    assert indexing.contexts[0].content == "订单服务依赖 Redis。"
    assert indexing.contexts[0].doc_version == 2
    operation_names = [call[0] for call in repository.calls]
    assert operation_names == ["schema", "begin", "activate", "prune"]
    assert repository.calls[2][1]["expected_chunk_count"] == 2
    assert service.completed is True


def test_graph_worker_uses_independent_queue() -> None:
    assert WorkerSettings.queue_name == "arq:graph"
    assert WorkerSettings.max_jobs > 0


def test_pipeline_stage_order_contains_switch_and_cleanup() -> None:
    assert GraphTaskStage.SWITCHING.value == "SWITCHING"
    assert GraphTaskStage.CLEANING.value == "CLEANING"
