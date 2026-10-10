"""进度实时发送；只有完成引用、语义和授权核验的回答才能逐段发布。"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress

import anyio
from sse_starlette.event import JSONServerSentEvent

from zhiqing_rag.modules.identity.auth_schemas import AuthError

from .models import AnswerProgress, AnswerStage, QueryAnswerResponse

logger = logging.getLogger(__name__)


async def answer_events(
    work: Callable[[AnswerProgress], Awaitable[QueryAnswerResponse]],
) -> AsyncIterator[JSONServerSentEvent]:
    # 有界队列让慢客户端产生背压，断开后必须取消生成和复核任务。
    queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue(maxsize=1)

    async def progress(stage: AnswerStage):
        await queue.put(("progress", {"stage": stage}))

    async def produce():
        try:
            response = await work(progress)
            await queue.put(("result", response.model_dump(mode="json")))
        except AuthError as error:
            await queue.put(
                ("error", {"status": error.status_code, "code": error.code, "message": str(error)})
            )
        except Exception as error:
            # HTTP 头已发送，只能使用流内错误；不暴露模型输出或底层异常文本。
            logger.warning("回答流失败: %s", type(error).__name__)
            await queue.put(
                (
                    "error",
                    {
                        "status": 503,
                        "code": "ANSWER_STREAM_UNAVAILABLE",
                        "message": "回答暂不可用，请稍后重试。",
                    },
                )
            )

    task = asyncio.create_task(produce())
    try:
        while True:
            event, data = await queue.get()
            if event == "result":
                # 不拆分尚未完整核验的句子。每个 answer 携带累计的完整陈述和引用。
                statements = data["answer"]["statements"]
                for count in range(1, len(statements) + 1):
                    partial = {
                        **data,
                        "answer": {**data["answer"], "statements": statements[:count]},
                    }
                    yield JSONServerSentEvent(event="answer", data=partial)
                yield JSONServerSentEvent(event="done", data=data)
                return
            yield JSONServerSentEvent(event=event, data=data)
            if event == "error":
                return
    finally:
        task.cancel()
        # EventSourceResponse 使用 AnyIO 取消域；屏蔽重复取消，等待上游清理完成。
        with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
            await task
