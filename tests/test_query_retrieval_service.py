"""混合检索排序、降级与等待期间的权限变化；替身不访问模型或数据库。"""

import asyncio
import hashlib
from dataclasses import replace
from uuid import UUID

import pytest

from zhiqing_rag.modules.identity.auth_schemas import (
    AuthError,
    AuthIdentity,
    AuthTenant,
    AuthUser,
)
from zhiqing_rag.query_analysis.models import QueryConstraint
from zhiqing_rag.query_retrieval.fusion import RRF_K, fuse_candidates
from zhiqing_rag.query_retrieval.models import ChunkRecord, RecallHit
from zhiqing_rag.query_retrieval.service import HybridRetrievalService
from zhiqing_rag.query_routing.models import (
    ChatPlan,
    ClarifyParams,
    ClarifyPlan,
    GraphParams,
    GraphPlan,
    GraphProjectionRef,
    HybridPlan,
    RetrievalParams,
    StructuredParams,
    StructuredPlan,
)
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument

QUERY = "工单关闭前需要谁确认？"


def identity(**changes):
    return AuthIdentity(
        **{
            "user_id": 2,
            "tenant_id": 1,
            "member_id": 7,
            "department_id": 8,
            "role": "USER",
            "clearance": 1,
            "auth_version": 1,
            "tenant_auth_version": 1,
            "session_key": UUID(int=1),
            "user": AuthUser(id="2", email="test@example.invalid", display_name="测试"),
            "tenant": AuthTenant(id="1", code="test", name="测试租户", role="USER"),
            **changes,
        }
    )


def scope(*, published=True, second_document=False):
    return AccessScope(
        tenant_id=1,
        user_id=2,
        knowledge_base_ids=(3,),
        document_ids=(4, 6) if second_document else (4,),
        published_documents=(
            (PublishedDocument(4, 3, 5), PublishedDocument(6, 3, 9))
            if second_document
            else (PublishedDocument(4, 3, 5),)
        )
        if published
        else (),
    )


def chunk(chunk_id=10, *, content=None, **changes):
    text = content if content is not None else f"工单原文 {chunk_id}"
    return ChunkRecord(
        **{
            "tenant_id": 1,
            "knowledge_base_id": 3,
            "document_id": 4,
            "revision_id": 11,
            "generation_id": 5,
            "chunk_id": chunk_id,
            "chunk_index": chunk_id - 10,
            "document_title": "工单管理规程",
            "document_name": "工单管理规程.pdf",
            "content": text,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "token_count": 12,
            "page_number": 3,
            "section_title": "关闭流程",
            **changes,
        }
    )


def hit(chunk_id=10, score=0.8, *, record=None, **changes):
    return RecallHit(record or chunk(chunk_id, **changes), score)


def plan(*, candidate_k=30, top_k=5, rerank=True, **changes):
    return HybridPlan(
        **{
            "original_query": QUERY,
            "query": QUERY,
            "requested_route": "hybrid",
            "reason_code": "TEST",
            "params": RetrievalParams(candidate_k=candidate_k, top_k=top_k, rerank=rerank),
            **changes,
        }
    )


class FakeRepository:
    def __init__(self, dense=(), bm25=(), *, failures=None, validations=()):
        self.hits = {"dense": dense, "bm25": bm25}
        self.failures = failures or {}
        self.validations = list(validations)
        self.calls = []
        self.validation_calls = []

    async def _recall(self, channel, user, access, query, limit):
        self.calls.append((channel, user, access, query, limit))
        error = self.failures.get(channel)
        if error is not None:
            raise error
        return self.hits[channel]

    async def dense(self, user, access, vector, limit):
        return await self._recall("dense", user, access, vector, limit)

    async def bm25(self, user, access, query, limit):
        return await self._recall("bm25", user, access, query, limit)

    async def revalidate(self, user, access, chunks):
        self.validation_calls.append((user, access, chunks))
        action = self.validations.pop(0) if self.validations else None
        if isinstance(action, BaseException):
            raise action
        if callable(action):
            return action(chunks)
        return chunks if action is None else action


class FakeEmbedder:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.calls = []
        self.vector = (1.0,) + (0.0,) * 1023

    async def embed(self, query):
        self.calls.append(query)
        if self.failure is not None:
            raise self.failure
        return self.vector


class FakeReranker:
    def __init__(self, *, failure=None, scores=None):
        self.failure = failure
        self.scores = scores
        self.calls = []

    async def rerank(self, query, documents):
        self.calls.append((query, documents))
        if self.failure is not None:
            raise self.failure
        if self.scores is not None:
            return self.scores
        return tuple((index, 1.0 / (index + 1)) for index in range(len(documents)))


def service(repository=None, embedder=None, reranker=None, **changes):
    return HybridRetrievalService(
        repository or FakeRepository(),
        embedder or FakeEmbedder(),
        reranker or FakeReranker(),
        **changes,
    )


async def execute(executor, *, query_plan=None, access=None, user=None):
    return await executor.execute(
        plan=query_plan or plan(), scope=access or scope(), identity=user or identity()
    )


def test_rrf_uses_rank_and_not_incompatible_channel_score_scales():
    dense = (hit(10, 0.99), hit(11, 0.4), hit(12, 0.3))
    bm25 = (hit(11, 1_000_000), hit(12, 500_000), hit(13, 1))
    fused = fuse_candidates(dense, bm25, limit=30)
    assert [candidate.chunk.chunk_id for candidate in fused] == [11, 12, 10, 13]
    assert fused[0].scores.fusion_score == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused[0].scores.dense_score == 0.4
    assert fused[0].scores.bm25_score == 1_000_000
    rescaled = fuse_candidates(
        tuple(replace(item, score=item.score * 1000) for item in dense),
        tuple(replace(item, score=item.score / 1000) for item in bm25),
        limit=30,
    )
    assert [item.chunk.chunk_id for item in rescaled] == [11, 12, 10, 13]
    assert [item.scores.fusion_score for item in rescaled] == [
        item.scores.fusion_score for item in fused
    ]


def test_fusion_counts_duplicate_chunk_once_per_branch_and_once_in_results():
    fused = fuse_candidates(
        (hit(10), hit(10, 0.7), hit(11, 0.6)),
        (hit(10, 20), hit(10, 10)),
        limit=30,
    )
    assert [item.chunk.chunk_id for item in fused] == [10, 11]
    assert fused[0].scores.fusion_score == pytest.approx(2 / (RRF_K + 1))
    assert fused[1].scores.dense_rank == 2


def test_fusion_deduplicates_same_document_text_but_preserves_other_document_citation():
    first = chunk(10, content="关闭前由申请人确认。")
    duplicate = chunk(11, content=first.content)
    other_document = chunk(
        12, content=first.content, document_id=6, revision_id=12, generation_id=9
    )
    fused = fuse_candidates(
        (hit(record=first), hit(record=duplicate), hit(record=other_document)), (), limit=30
    )
    assert [(item.chunk.document_id, item.chunk.chunk_id) for item in fused] == [(4, 10), (6, 12)]


def test_fusion_ties_are_stable_and_budget_applies_after_deduplication():
    fused = fuse_candidates((hit(15), hit(16), hit(17)), (hit(10), hit(11), hit(12)), limit=4)
    assert [item.chunk.chunk_id for item in fused] == [10, 15, 11, 16]


def test_fusion_rejects_conflicting_original_text_for_same_chunk_id():
    with pytest.raises(ValueError, match="不一致"):
        fuse_candidates((hit(10),), (hit(10, content="另一段原文"),), limit=30)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_fusion_rejects_nonfinite_scores(score):
    with pytest.raises(ValueError, match="有限"):
        fuse_candidates((hit(10, score),), (), limit=30)


async def test_candidate_budget_each_branch_fusion_budget_and_final_top_k():
    repo = FakeRepository(
        dense=tuple(hit(index) for index in range(10, 40)),
        bm25=tuple(hit(index, 1000) for index in range(40, 70)),
    )
    reranker = FakeReranker()
    embedder = FakeEmbedder()
    result = await execute(service(repo, embedder, reranker))
    assert {call[0]: call[4] for call in repo.calls} == {"dense": 30, "bm25": 30}
    assert next(call[3] for call in repo.calls if call[0] == "dense") is embedder.vector
    assert result.trace.dense_count == result.trace.bm25_count == 30
    assert result.trace.fused_count == result.trace.authorized_count == 30
    assert len(reranker.calls[0][1]) == 30
    assert result.trace.returned_count == len(result.evidence) == 5
    assert result.status == "ok"
    assert result.trace.rerank_status == "succeeded"
    assert len(repo.validation_calls) == 2
    first = result.evidence[0]
    assert (first.chunk_id, first.document_name, first.page_number, first.section_title) == (
        "10",
        "工单管理规程.pdf",
        3,
        "关闭流程",
    )
    assert first.content == "工单原文 10"
    assert first.generation_id == "5"
    assert [item.citation_id for item in result.evidence] == ["E1", "E2", "E3", "E4", "E5"]


async def test_reranker_result_indexes_control_order_and_keep_each_original_source():
    repo = FakeRepository(dense=(hit(10), hit(11), hit(12)))
    reranker = FakeReranker(scores=((0, 0.2), (2, 0.8), (1, 0.4)))
    result = await execute(service(repo, reranker=reranker), query_plan=plan(top_k=2))
    assert [
        (item.chunk_id, item.content, item.scores.rerank_score) for item in result.evidence
    ] == [("12", "工单原文 12", 0.8), ("11", "工单原文 11", 0.4)]


@pytest.mark.parametrize(
    ("stage", "code"),
    [
        ("embedding", "EMBEDDING_UNAVAILABLE"),
        ("dense", "DENSE_UNAVAILABLE"),
        ("bm25", "BM25_UNAVAILABLE"),
        ("rerank", "RERANKER_UNAVAILABLE"),
    ],
)
async def test_single_stage_failure_explicitly_degrades_and_retains_available_evidence(stage, code):
    failure = RuntimeError("包含敏感服务细节的错误")
    repo = FakeRepository(
        dense=(hit(10), hit(11)),
        bm25=(hit(10, 20), hit(11, 10)),
        failures={stage: failure},
    )
    embedder = FakeEmbedder(failure=failure if stage == "embedding" else None)
    reranker = FakeReranker(failure=failure if stage == "rerank" else None)
    result = await execute(service(repo, embedder, reranker))
    assert result.status == "degraded"
    assert result.degradation_codes == (code,)
    assert len(result.evidence) == 2
    assert "敏感" not in result.model_dump_json()
    if stage == "embedding":
        assert all(call[0] != "dense" for call in repo.calls)
        assert result.trace.dense_status == "failed"
    elif stage == "rerank":
        assert result.trace.rerank_status == "failed"
        assert [item.chunk_id for item in result.evidence] == ["10", "11"]
    else:
        assert getattr(result.trace, f"{stage}_status") == "failed"


@pytest.mark.parametrize("embedding_failed", [False, True])
async def test_both_recall_paths_unavailable_fail_503(embedding_failed):
    failure = RuntimeError("数据库或供应商内部详情")
    repo = FakeRepository(failures={"dense": failure, "bm25": failure})
    embedder = FakeEmbedder(failure=failure if embedding_failed else None)
    reranker = FakeReranker()
    with pytest.raises(AuthError) as raised:
        await execute(service(repo, embedder, reranker))
    assert (raised.value.status_code, raised.value.code) == (503, "RETRIEVAL_UNAVAILABLE")
    assert "内部详情" not in str(raised.value)
    assert not reranker.calls
    assert not repo.validation_calls


async def test_failed_branch_and_no_matches_preserve_failure_code_with_empty_status():
    repo = FakeRepository(failures={"bm25": RuntimeError("不可用")})
    result = await execute(service(repo))
    assert result.status == "empty"
    assert result.evidence == ()
    assert result.degradation_codes == ("BM25_UNAVAILABLE",)


@pytest.mark.parametrize("global_enabled", [False, True])
async def test_disabled_reranking_never_calls_external_provider(global_enabled):
    repo = FakeRepository(dense=(hit(10), hit(11)))
    reranker = FakeReranker()
    result = await execute(
        service(repo, reranker=reranker, reranker_enabled=global_enabled),
        query_plan=plan(rerank=not global_enabled),
    )
    assert result.trace.rerank_status == "disabled"
    assert not reranker.calls
    assert len(repo.validation_calls) == 1
    assert all(item.scores.rerank_score is None for item in result.evidence)


@pytest.mark.parametrize("count", [0, 1])
async def test_no_or_single_candidate_skips_reranking(count):
    repo = FakeRepository(dense=(hit(10),) if count else ())
    reranker = FakeReranker()
    result = await execute(service(repo, reranker=reranker))
    assert result.trace.rerank_status == "skipped"
    assert len(result.evidence) == count
    assert not reranker.calls
    assert len(repo.validation_calls) == count


@pytest.mark.parametrize("changes", [{"tenant_id": 99}, {"user_id": 99}])
async def test_identity_scope_mismatch_denies_before_any_provider_or_repository_call(changes):
    executor = service()
    with pytest.raises(AuthError) as raised:
        await execute(executor, user=identity(**changes))
    assert (raised.value.status_code, raised.value.code) == (403, "QUERY_SCOPE_FORBIDDEN")
    assert not executor.repository.calls
    assert not executor.embedder.calls
    assert not executor.reranker.calls


@pytest.mark.parametrize("channel", ["dense", "bm25"])
@pytest.mark.parametrize(
    "changes",
    [{"tenant_id": 99}, {"knowledge_base_id": 99}, {"document_id": 99}, {"generation_id": 99}],
)
async def test_out_of_scope_recall_denies_without_external_text_or_returned_evidence(
    channel, changes
):
    repo = FakeRepository(**{channel: (hit(10, **changes),)})
    reranker = FakeReranker()
    with pytest.raises(AuthError) as raised:
        await execute(service(repo, reranker=reranker))
    assert (raised.value.status_code, raised.value.code) == (403, "QUERY_SCOPE_FORBIDDEN")
    assert not reranker.calls
    assert not repo.validation_calls


async def test_first_permission_recheck_removes_revoked_text_before_reranker_request():
    repo = FakeRepository(
        dense=(hit(10, content="已撤销读取权的敏感原文"), hit(11), hit(12)),
        validations=[lambda records: tuple(item for item in records if item.chunk_id != 10)],
    )
    reranker = FakeReranker()
    result = await execute(service(repo, reranker=reranker))
    assert reranker.calls == [(QUERY, ("工单原文 11", "工单原文 12"))]
    assert [item.chunk_id for item in result.evidence] == ["11", "12"]
    assert result.degradation_codes == ("SCOPE_CHANGED",)
    assert result.trace.authorized_count == 2
    assert result.status == "degraded"


async def test_all_permissions_revoked_before_reranker_returns_empty_and_sends_no_text():
    repo = FakeRepository(dense=(hit(10), hit(11)), validations=[()])
    reranker = FakeReranker()
    result = await execute(service(repo, reranker=reranker))
    assert result.status == "empty"
    assert result.degradation_codes == ("SCOPE_CHANGED",)
    assert not result.evidence
    assert not reranker.calls


@pytest.mark.parametrize("change", ["deleted", "generation", "content", "hash", "revision"])
async def test_after_rerank_recheck_drops_changed_chunk_and_fills_top_k_from_remaining_pool(change):
    def current_records(records):
        remaining = [record for record in records if record.chunk_id != 10]
        first = next(record for record in records if record.chunk_id == 10)
        replacements = {
            "generation": replace(first, generation_id=99),
            "content": replace(first, content="等待重排期间发生变化的原文"),
            "hash": replace(first, content_sha256="变化后的摘要"),
            "revision": replace(first, revision_id=99),
        }
        return tuple(remaining) if change == "deleted" else (replacements[change], *remaining)

    repo = FakeRepository(
        dense=tuple(hit(index) for index in range(10, 17)), validations=[None, current_records]
    )
    result = await execute(service(repo))
    assert [item.chunk_id for item in result.evidence] == ["11", "12", "13", "14", "15"]
    assert result.trace.returned_count == 5
    assert result.trace.authorized_count == 6
    assert result.degradation_codes == ("SCOPE_CHANGED",)
    assert result.trace.rerank_status == "succeeded"
    assert len(repo.validation_calls[1][2]) == 7


@pytest.mark.parametrize("phase", [0, 1])
async def test_permission_recheck_failure_fails_closed_even_when_other_stages_succeeded(phase):
    repo = FakeRepository(
        dense=(hit(10), hit(11)),
        validations=[None] * phase + [RuntimeError("授权数据库内部错误")],
    )
    reranker = FakeReranker()
    with pytest.raises(AuthError) as raised:
        await execute(service(repo, reranker=reranker))
    assert (raised.value.status_code, raised.value.code) == (503, "RETRIEVAL_UNAVAILABLE")
    assert "内部错误" not in str(raised.value)
    assert len(reranker.calls) == phase


@pytest.mark.parametrize("phase", [0, 1])
async def test_session_rejected_during_revalidation_propagates_auth_failure(phase):
    repo = FakeRepository(
        dense=(hit(10), hit(11)),
        validations=[None] * phase + [AuthError(401, "SESSION_INVALID", "会话已失效")],
    )
    with pytest.raises(AuthError) as raised:
        await execute(service(repo))
    assert (raised.value.status_code, raised.value.code) == (401, "SESSION_INVALID")


@pytest.mark.parametrize(
    "scores",
    [
        ((0, 0.9),),
        ((0, 0.9), (0, 0.8)),
        ((0, 0.9), (2, 0.8)),
        ((False, 0.9), (1, 0.8)),
        ((0, True), (1, 0.8)),
        ((0, float("nan")), (1, 0.8)),
        ((0, 0.9), (1, float("inf"))),
    ],
)
async def test_invalid_rerank_response_degrades_to_fusion_order_and_rechecks_permissions(scores):
    repo = FakeRepository(dense=(hit(10), hit(11)))
    result = await execute(service(repo, reranker=FakeReranker(scores=scores)))
    assert result.status == "degraded"
    assert result.degradation_codes == ("RERANKER_UNAVAILABLE",)
    assert [item.chunk_id for item in result.evidence] == ["10", "11"]
    assert all(item.scores.rerank_score is None for item in result.evidence)
    assert len(repo.validation_calls) == 2


@pytest.mark.parametrize("stage", ["embedding", "dense", "bm25", "rerank", "authorization"])
async def test_cancellation_is_never_converted_to_degradation(stage):
    canceled = asyncio.CancelledError()
    repo = FakeRepository(
        dense=(hit(10), hit(11)),
        bm25=(hit(10, 20),),
        failures={stage: canceled},
        validations=[canceled] if stage == "authorization" else [],
    )
    embedder = FakeEmbedder(failure=canceled if stage == "embedding" else None)
    reranker = FakeReranker(failure=canceled if stage == "rerank" else None)
    with pytest.raises(asyncio.CancelledError):
        await execute(service(repo, embedder, reranker))


async def test_bm25_starts_while_embedding_waits_and_both_recalls_can_run_in_parallel():
    embedding_started = asyncio.Event()
    bm25_started = asyncio.Event()
    dense_started = asyncio.Event()

    class BlockingEmbedder(FakeEmbedder):
        async def embed(self, query):
            embedding_started.set()
            await asyncio.wait_for(bm25_started.wait(), timeout=1)
            return await super().embed(query)

    class ParallelRepository(FakeRepository):
        async def dense(self, user, access, vector, limit):
            dense_started.set()
            await asyncio.wait_for(bm25_started.wait(), timeout=1)
            return await super().dense(user, access, vector, limit)

        async def bm25(self, user, access, query, limit):
            await asyncio.wait_for(embedding_started.wait(), timeout=1)
            bm25_started.set()
            await asyncio.wait_for(dense_started.wait(), timeout=1)
            return await super().bm25(user, access, query, limit)

    result = await execute(
        service(ParallelRepository(dense=(hit(10),), bm25=(hit(10, 20),)), BlockingEmbedder())
    )
    assert result.status == "ok"
    assert result.trace.dense_status == result.trace.bm25_status == "succeeded"


def nonhybrid_plans():
    common = {"original_query": QUERY, "query": QUERY, "reason_code": "TEST"}
    return [
        (ChatPlan(**common, requested_route="none"), "skipped"),
        (
            ClarifyPlan(
                **common, requested_route="clarify", params=ClarifyParams(question="请指定工单。")
            ),
            "skipped",
        ),
        (
            StructuredPlan(
                **common,
                requested_route="structured",
                params=StructuredParams(operation="count_documents"),
            ),
            "unsupported",
        ),
        (
            GraphPlan(
                **common,
                requested_route="hybrid_graph",
                params=GraphParams(
                    seed_entity_uids=("工单_UID",),
                    projections=(
                        GraphProjectionRef(
                            tenant_id="1",
                            document_id="4",
                            generation_id="5",
                            projection_id="6",
                            namespace="test",
                        ),
                    ),
                ),
            ),
            "unsupported",
        ),
        (plan(planning_status="unsupported", message="元数据过滤未就绪"), "unsupported"),
        (plan(planning_status="empty", message="范围没有文档"), "empty"),
    ]


@pytest.mark.parametrize(("query_plan", "status"), nonhybrid_plans())
async def test_nonexecuting_plans_do_not_call_database_embedding_or_reranker(query_plan, status):
    executor = service()
    result = await execute(executor, query_plan=query_plan)
    assert result.status == status
    assert not result.evidence
    assert not executor.repository.calls
    assert not executor.repository.validation_calls
    assert not executor.embedder.calls
    assert not executor.reranker.calls


async def test_empty_published_scope_short_circuits_before_all_external_calls():
    executor = service()
    result = await execute(executor, access=scope(published=False))
    assert result.status == "empty"
    assert not executor.repository.calls
    assert not executor.embedder.calls
    assert not executor.reranker.calls


async def test_unimplemented_metadata_filter_is_not_silently_dropped():
    constraint = QueryConstraint.model_validate(
        {
            "field": "department",
            "operator": "eq",
            "value": "客服部",
            "evidence": {"source": "query", "history_index": None, "quote": "客服部"},
        }
    )
    executor = service()
    result = await execute(executor, query_plan=plan(constraints=(constraint,)))
    assert result.status == "unsupported"
    assert not executor.repository.calls
    assert not executor.embedder.calls


async def test_semantic_other_constraint_keeps_the_query_and_allows_hybrid_execution():
    constraint = QueryConstraint.model_validate(
        {
            "field": "other",
            "operator": "unspecified",
            "value": "关闭前",
            "evidence": {"source": "query", "history_index": None, "quote": "关闭前"},
        }
    )
    executor = service(FakeRepository(dense=(hit(10),)))
    result = await execute(executor, query_plan=plan(constraints=(constraint,)))
    assert result.status == "ok"
    assert executor.embedder.calls == [QUERY]
    assert next(call[3] for call in executor.repository.calls if call[0] == "bm25") == QUERY
