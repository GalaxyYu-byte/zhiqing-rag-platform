"""会话编排的行为测试：重复生成、上下文撤权、取消、超时与发布顺序。"""

import asyncio
import copy
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from test_query_answer_api import Answerer
from test_query_search_api import SCOPE, Analyzer, Retriever

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.chat.rewrite import RewriteResult
from zhiqing_rag.modules.chat.schemas import ChatAnswerBody, ChatRunView
from zhiqing_rag.modules.chat.service import ChatService, canonical_request, request_digest
from zhiqing_rag.modules.chat.streaming import chat_events
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_routing.service import QueryRouterService

IDENTITY = SimpleNamespace(tenant_id=1, user_id=3, member_id=10)


class MemoryRepository:
    """替代外部存储，模拟可见性及终态；真实SQL由仓储/集成测试覆盖。"""

    def __init__(self):
        self.rows = {}
        self.completed = []
        self.terminated = []
        self.hidden = set()
        self.complete_error = None
        self.heartbeat_error = None

    async def start_run(self, identity, session_id, request, digest, owner, lease_seconds):
        for row in self.rows.values():
            if row["client_request_id"] == request["client_request_id"]:
                if row["request_sha256"] != digest:
                    raise AuthError(409, "CHAT_IDEMPOTENCY_CONFLICT", "冲突")
                return copy.deepcopy(row) | {"is_new": False}
        if any(row["status"] == "RUNNING" for row in self.rows.values()):
            raise AuthError(409, "CHAT_RUN_ACTIVE", "已有活动提问")
        run_id = str(len(self.rows) + 1)
        row = {
            "id": run_id,
            "session_id": str(session_id),
            "client_request_id": request["client_request_id"],
            "question": request["question"],
            "standalone_question": None,
            "status": "RUNNING",
            "created_at": datetime.now(UTC),
            "started_at": datetime.now(UTC),
            "finished_at": None,
            "error_code": None,
            "response": None,
            "visibility": "pending",
            "request_sha256": digest,
            "lease_owner": owner,
            "lease_epoch": 1,
        }
        self.rows[run_id] = row
        return copy.deepcopy(row) | {"is_new": True}

    async def get_run(self, identity, session_id, run_id):
        row = self.rows[str(run_id)]
        result = {key: copy.deepcopy(row[key]) for key in ChatRunView.model_fields}
        if str(run_id) in self.hidden:
            result |= {"response": None, "standalone_question": None, "visibility": "unavailable"}
        return result

    async def list_runs(self, identity, session_id, before_run_id=None, limit=20):
        rows = [
            row
            for key, row in self.rows.items()
            if before_run_id is None or int(key) < before_run_id
        ]
        return {
            "runs": [
                await self.get_run(identity, session_id, int(row["id"])) for row in rows[-limit:]
            ],
            "has_more": len(rows) > limit,
            "next_before_run_id": None,
        }

    async def heartbeat(self, *args, **kwargs):
        if self.heartbeat_error:
            raise self.heartbeat_error

    async def complete(self, identity, session_id, run_id, **kwargs):
        if self.complete_error:
            raise self.complete_error
        self.completed.append(copy.deepcopy(kwargs))
        self.rows[str(run_id)] |= {
            "status": "SUCCEEDED",
            "visibility": "visible",
            "finished_at": datetime.now(UTC),
            "response": kwargs["response"],
            "standalone_question": kwargs["standalone_question"],
        }

    async def terminate(self, identity, session_id, run_id, **kwargs):
        self.terminated.append(kwargs)
        if self.rows[str(run_id)]["status"] == "RUNNING":
            self.rows[str(run_id)] |= {
                "status": kwargs["status"],
                "error_code": kwargs["error_code"],
                "visibility": "visible",
                "finished_at": datetime.now(UTC),
            }


class Rewriter:
    def __init__(self):
        self.calls = []
        self.result = None
        self.after = None

    async def rewrite(self, query, history):
        self.calls.append(copy.deepcopy(history))
        if self.after:
            self.after()
        return self.result or RewriteResult(question=query)


def setup(*, settings=None):
    repository, rewriter = MemoryRepository(), Rewriter()
    service = ChatService(repository, rewriter, settings or Settings(_env_file=None))
    return service, repository, rewriter, Analyzer(), Retriever(), Answerer()


async def perform(parts, query="工单如何关闭？", request_id="request-1", *, scope=SCOPE):
    service, _, _, analyzer, retriever, answerer = parts
    body = ChatAnswerBody(query=query, client_request_id=request_id)
    run = await service.begin(IDENTITY, 1, body)
    return await service.execute(
        identity=IDENTITY,
        session_id=1,
        body=body,
        run=run,
        scope=scope,
        analyzer=analyzer,
        planner=QueryRouterService(),
        retriever=retriever,
        answerer=answerer,
    )


def test_request_hash_binds_question_and_scope_but_normalizes_selection_order():
    first = ChatAnswerBody(
        query=" 工单如何关闭？ ", client_request_id="one", knowledge_base_ids=["12", "2"]
    )
    second = first.model_copy(update={"knowledge_base_ids": ["2", "12"]})
    assert request_digest(canonical_request(first)) == request_digest(canonical_request(second))
    assert request_digest(canonical_request(first)) != request_digest(
        canonical_request(first.model_copy(update={"knowledge_base_ids": ["2"]}))
    )


async def test_success_and_same_request_replay_generate_and_commit_once():
    parts = setup()
    first = await perform(parts)
    second = await perform(parts)
    assert first.id == second.id and first.status == second.status == "SUCCEEDED"
    assert first.response == second.response
    assert len(parts[1].completed) == len(parts[3].calls) == len(parts[5].calls) == 1


async def test_same_key_with_different_question_conflicts_without_generation():
    parts = setup()
    await perform(parts)
    with pytest.raises(AuthError) as error:
        await perform(parts, "工单如何创建？")
    assert error.value.code == "CHAT_IDEMPOTENCY_CONFLICT"
    assert len(parts[5].calls) == 1


async def test_active_identical_request_returns_in_progress_not_second_execution():
    parts = setup()
    service = parts[0]
    body = ChatAnswerBody(query="工单如何关闭？", client_request_id="one")
    await service.begin(IDENTITY, 1, body)
    with pytest.raises(AuthError) as error:
        await service.begin(IDENTITY, 1, body)
    assert error.value.code == "CHAT_RUN_IN_PROGRESS"
    with pytest.raises(AuthError) as error:
        await service.begin(IDENTITY, 1, body.model_copy(update={"client_request_id": "two"}))
    assert error.value.code == "CHAT_RUN_ACTIVE"


async def test_followup_uses_authorized_answer_and_records_only_actual_dependencies():
    parts = setup()
    await perform(parts)
    parts[2].result = RewriteResult(
        "工单由谁确认？", input_tokens=10, output_tokens=5, used_run_ids=("1",)
    )
    second = await perform(parts, "它由谁确认？", "request-2")
    assert parts[2].calls[-1][0]["answer"]["review_status"] == "passed"
    assert second.question == "它由谁确认？"
    assert second.standalone_question == second.response.plan.query == "工单由谁确认？"
    assert second.response.plan.original_query == "它由谁确认？"
    assert parts[1].completed[-1]["history_run_ids"] == [1]
    assert second.response.answer.trace.input_tokens == 10


async def test_independent_question_has_no_unnecessary_old_source_dependency():
    parts = setup()
    await perform(parts)
    await perform(parts, "工单如何创建？", "request-2")
    assert parts[1].completed[-1]["history_run_ids"] == []


async def test_revoke_context_during_rewrite_prevents_generation_and_save():
    parts = setup()
    await perform(parts)
    parts[2].result = RewriteResult("工单由谁确认？", used_run_ids=("1",))
    parts[2].after = lambda: parts[1].hidden.add("1")
    with pytest.raises(AuthError) as error:
        await perform(parts, "它由谁确认？", "request-2")
    assert error.value.code == "CHAT_CONTEXT_CHANGED"
    assert len(parts[5].calls) == len(parts[1].completed) == 1
    assert parts[1].rows["2"]["status"] == "FAILED"


async def test_unavailable_latest_round_cannot_redirect_second_item_to_older_round():
    parts = setup()
    await perform(parts)
    await perform(parts, "工单如何创建？", "request-2")
    parts[1].hidden.add("2")
    calls = len(parts[2].calls)
    result = await perform(parts, "第二条展开一下", "request-3")
    assert len(parts[2].calls) == calls
    assert result.response.answer.status == "skipped"


async def test_scope_change_does_not_feed_out_of_scope_answer_to_rewriter():
    from zhiqing_rag.query_routing.scope import AccessScope

    parts = setup()
    await perform(parts)
    calls = len(parts[2].calls)
    result = await perform(parts, "它由谁确认？", "request-2", scope=AccessScope(1, 3, (), (), ()))
    assert len(parts[2].calls) == calls and result.response.answer.status == "skipped"


async def test_clarification_persists_explicit_result_without_retrieval_or_generation():
    parts = setup()
    parts[2].result = RewriteResult(None, clarification="请补充对象名称。", input_tokens=7)
    result = await perform(parts)
    assert result.status == "SUCCEEDED" and result.response.answer.status == "skipped"
    assert not parts[3].calls and not parts[4].calls and not parts[5].calls
    assert result.response.answer.trace.input_tokens == 7


async def test_final_authorization_failure_never_publishes_draft():
    parts = setup()
    parts[1].complete_error = AuthError(403, "CHAT_SOURCE_UNAVAILABLE", "来源变化")
    with pytest.raises(AuthError):
        await perform(parts)
    assert not parts[1].completed and parts[1].rows["1"]["response"] is None
    assert parts[1].rows["1"]["status"] == "FAILED"


async def test_cancel_stops_work_and_persists_cancelled_state():
    parts = setup()
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def stalled(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    parts[5].execute = stalled
    task = asyncio.create_task(perform(parts))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set() and not parts[1].completed
    assert parts[1].rows["1"]["status"] == "CANCELLED"


async def test_timeout_finishes_run_and_releases_generation_task():
    parts = setup(
        settings=SimpleNamespace(
            chat_history_turns=6,
            chat_run_lease_seconds=30,
            chat_run_timeout_seconds=0.02,
        )
    )

    async def stalled(**kwargs):
        await asyncio.Event().wait()

    parts[5].execute = stalled
    with pytest.raises(AuthError) as error:
        await perform(parts)
    assert error.value.code == "CHAT_TIMEOUT" and parts[1].rows["1"]["status"] == "FAILED"


async def test_lost_heartbeat_cancels_inflight_work_before_publication():
    parts = setup(
        settings=SimpleNamespace(
            chat_history_turns=6,
            chat_run_lease_seconds=0.03,
            chat_run_timeout_seconds=1,
        )
    )
    heartbeat_calls = 0

    async def heartbeat(*args, **kwargs):
        nonlocal heartbeat_calls
        heartbeat_calls += 1
        if heartbeat_calls > 1:
            raise AuthError(409, "CHAT_LEASE_LOST", "租约失效")

    parts[1].heartbeat = heartbeat

    async def stalled(**kwargs):
        await asyncio.Event().wait()

    parts[5].execute = stalled
    with pytest.raises(AuthError) as error:
        await perform(parts)
    assert error.value.code == "CHAT_LEASE_LOST" and not parts[1].completed


async def test_expired_lease_after_scope_loading_prevents_model_work():
    parts = setup()
    parts[1].heartbeat_error = AuthError(409, "CHAT_LEASE_LOST", "租约失效")
    with pytest.raises(AuthError) as error:
        await perform(parts)
    assert error.value.code == "CHAT_LEASE_LOST"
    assert not parts[2].calls and not parts[3].calls and not parts[5].calls
    assert parts[1].rows["1"]["status"] == "FAILED"


async def test_stream_sends_committed_cumulative_statements_once_and_no_snapshot_in_metadata():
    parts = setup()
    committed = await perform(parts)
    committed.response.answer.statements.append(committed.response.answer.statements[0])

    async def work(progress):
        assert parts[1].completed
        await progress("review")
        return committed

    initial = committed.model_dump(mode="json") | {"response": None}
    frames = [event async for event in chat_events(work, initial)]
    assert [event.event for event in frames] == ["run", "progress", "answer", "answer", "done"]
    answers = [json.loads(event.data) for event in frames if event.event == "answer"]
    assert [len(frame["response"]["answer"]["statements"]) for frame in answers] == [1, 2]
    assert all(frame["run"]["response"] is None for frame in answers)
    assert len(parts[1].completed) == 1


async def test_unavailable_replay_has_terminal_metadata_but_no_answer_event():
    parts = setup()
    committed = await perform(parts)
    parts[1].hidden.add("1")
    replay = await perform(parts)

    async def work(progress):
        return replay

    frames = [
        event
        async for event in chat_events(work, committed.model_dump(mode="json") | {"response": None})
    ]
    assert [frame.event for frame in frames] == ["run", "done"]
    final = json.loads(frames[-1].data)
    assert final["response"] is None and final["run"]["standalone_question"] is None


async def test_stream_disconnect_cancels_producer():
    started, cleaned = asyncio.Event(), asyncio.Event()

    async def work(progress):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    iterator = chat_events(work, {"id": "1", "session_id": "1"})
    assert (await anext(iterator)).event == "run"
    await asyncio.wait_for(started.wait(), 1)
    await iterator.aclose()
    assert cleaned.is_set()


async def test_stream_rechecks_sources_between_statements_and_stops_after_revocation():
    parts = setup()
    committed = await perform(parts)
    committed.response.answer.statements.append(committed.response.answer.statements[0])
    parts[1].rows["1"]["response"] = committed.response.model_dump(mode="json")
    checks = 0

    async def work(progress):
        return committed

    async def revalidate():
        nonlocal checks
        checks += 1
        if checks == 2:
            parts[1].hidden.add("1")
        return ChatRunView.model_validate(await parts[1].get_run(IDENTITY, 1, 1))

    frames = [
        event
        async for event in chat_events(
            work,
            committed.model_dump(mode="json") | {"response": None},
            revalidate=revalidate,
        )
    ]
    assert [frame.event for frame in frames] == ["run", "answer", "error"]
    assert json.loads(frames[-1].data)["code"] == "CHAT_SOURCE_UNAVAILABLE"
    assert parts[1].rows["1"]["status"] == "SUCCEEDED"


@pytest.mark.parametrize(
    "error",
    [
        AuthError(401, "AUTH_SESSION_REVOKED", "登录已失效"),
        RuntimeError("dependency failed"),
    ],
)
async def test_stream_publication_failure_sends_no_saved_answer(error):
    parts = setup()
    committed = await perform(parts)

    async def work(progress):
        return committed

    async def revalidate():
        raise error

    frames = [
        event
        async for event in chat_events(
            work,
            committed.model_dump(mode="json") | {"response": None},
            revalidate=revalidate,
        )
    ]
    assert [frame.event for frame in frames] == ["run", "error"]
    assert parts[1].rows["1"]["status"] == "SUCCEEDED"


async def test_disconnect_after_commit_keeps_result_available_for_recovery():
    parts = setup()
    committed = await perform(parts)

    async def work(progress):
        return committed

    iterator = chat_events(work, committed.model_dump(mode="json") | {"response": None})
    assert (await anext(iterator)).event == "run"
    assert (await anext(iterator)).event == "answer"
    await iterator.aclose()
    replay = await perform(parts)
    assert replay.status == "SUCCEEDED" and replay.response is not None
    assert len(parts[5].calls) == len(parts[1].completed) == 1
