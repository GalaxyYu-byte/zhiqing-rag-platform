from typing import Any

import pytest

from zq_rag_app.core import task_queue


class _FakeQueue:
    def __init__(self, effects: list[Any]) -> None:
        self.effects = effects
        self.calls: list[tuple[str, int, str]] = []
        self.queue_names: list[str | None] = []
        self.closed = False

    async def enqueue_job(
        self,
        function: str,
        task_id: int,
        *,
        _job_id: str,
        _queue_name: str | None = None,
    ) -> Any:
        self.calls.append((function, task_id, _job_id))
        self.queue_names.append(_queue_name)
        effect = self.effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_enqueue_rebuilds_broken_arq_client_and_retries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_queue = _FakeQueue([TypeError("'NoneType' object is not callable")])
    new_queue = _FakeQueue([object()])
    created = 0

    async def fake_create_pool(*_: Any, **__: Any) -> _FakeQueue:
        nonlocal created
        created += 1
        return new_queue

    monkeypatch.setattr(task_queue, "_pool", old_queue)
    monkeypatch.setattr(task_queue, "create_pool", fake_create_pool)

    queued = await task_queue.enqueue_document_index(5)

    assert queued is True
    assert created == 1
    assert old_queue.closed is True
    assert old_queue.calls == [("execute_document_index", 5, "document-index:5")]
    assert new_queue.calls == [("execute_document_index", 5, "document-index:5")]


@pytest.mark.asyncio
async def test_idempotent_retry_counts_existing_job_as_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_queue = _FakeQueue([ConnectionError("connection lost")])
    # 第一次请求可能已经写入 Redis；重试时 ARQ 通过相同 Job ID 返回 None。
    new_queue = _FakeQueue([None])

    async def fake_create_pool(*_: Any, **__: Any) -> _FakeQueue:
        return new_queue

    monkeypatch.setattr(task_queue, "_pool", old_queue)
    monkeypatch.setattr(task_queue, "create_pool", fake_create_pool)

    assert await task_queue.enqueue_document_index(9) is True


@pytest.mark.asyncio
async def test_initial_duplicate_still_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _FakeQueue([None])
    monkeypatch.setattr(task_queue, "_pool", queue)

    assert await task_queue.enqueue_document_index(12) is False


@pytest.mark.asyncio
async def test_enqueue_reraises_when_reconnect_attempt_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_queue = _FakeQueue([ConnectionError("first failure")])
    new_queue = _FakeQueue([ConnectionError("second failure")])

    async def fake_create_pool(*_: Any, **__: Any) -> _FakeQueue:
        return new_queue

    monkeypatch.setattr(task_queue, "_pool", old_queue)
    monkeypatch.setattr(task_queue, "create_pool", fake_create_pool)

    with pytest.raises(ConnectionError, match="second failure"):
        await task_queue.enqueue_document_index(15)

    assert old_queue.closed is True
    assert new_queue.closed is True


@pytest.mark.asyncio
async def test_graph_task_uses_dedicated_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _FakeQueue([object()])
    monkeypatch.setattr(task_queue, "_pool", queue)

    assert await task_queue.enqueue_graph_index(21) is True
    assert queue.calls == [("execute_graph_index", 21, "graph-index:21")]
    assert queue.queue_names == [task_queue.settings.graph_task_queue_name]
