"""会话流只发送已持久化并复查授权的结果，断连遵循取消当前轮次的契约。"""

import asyncio
import logging
from contextlib import suppress

import anyio
from sse_starlette.event import JSONServerSentEvent

from zhiqing_rag.modules.identity.auth_schemas import AuthError

logger = logging.getLogger(__name__)


async def chat_events(work, initial_run, *, revalidate=None):
    queue = asyncio.Queue(maxsize=1)

    async def progress(stage):
        await queue.put(("progress", {"stage": stage}))

    async def produce():
        try:
            run = await work(progress)
            await queue.put(("result", run))
        except AuthError as error:
            await queue.put(
                (
                    "error",
                    {
                        "status": error.status_code,
                        "code": error.code,
                        "message": str(error),
                    },
                )
            )
        except Exception as error:
            logger.warning("会话回答流失败: %s", type(error).__name__)
            await queue.put(
                (
                    "error",
                    {
                        "status": 503,
                        "code": "CHAT_EXECUTION_FAILED",
                        "message": "本轮暂不可用，请恢复会话。",
                    },
                )
            )

    task = asyncio.create_task(produce())
    try:
        yield JSONServerSentEvent(event="run", data=initial_run)
        while True:
            event, data = await queue.get()
            if event == "result":
                meta = data.model_dump(mode="json", exclude={"response"}) | {"response": None}
                response = data.response.model_dump(mode="json") if data.response else None

                async def check_publication():
                    if revalidate is None:
                        return None
                    try:
                        fresh = await revalidate()
                        current = fresh.response.model_dump(mode="json") if fresh.response else None
                        if current != response:
                            return {
                                "status": 403,
                                "code": "CHAT_SOURCE_UNAVAILABLE",
                                "message": "回答来源已变化，请恢复会话。",
                            }
                    except AuthError as error:
                        return {
                            "status": error.status_code,
                            "code": error.code,
                            "message": str(error),
                        }
                    except Exception as error:
                        logger.warning("会话发布复核失败: %s", type(error).__name__)
                        return {
                            "status": 503,
                            "code": "CHAT_PUBLICATION_UNAVAILABLE",
                            "message": "无法确认本轮资料权限，请恢复会话。",
                        }
                    return None

                if response is not None:
                    statements = response["answer"]["statements"]
                    if response["answer"]["status"] == "structured":
                        error = await check_publication()
                        if error:
                            yield JSONServerSentEvent(event="error", data=error)
                            return
                        yield JSONServerSentEvent(
                            event="answer", data={"run": meta, "response": response}
                        )
                    for count in range(1, len(statements) + 1):
                        error = await check_publication()
                        if error:
                            yield JSONServerSentEvent(event="error", data=error)
                            return
                        partial = response | {
                            "answer": response["answer"] | {"statements": statements[:count]}
                        }
                        yield JSONServerSentEvent(
                            event="answer", data={"run": meta, "response": partial}
                        )
                error = await check_publication()
                if error:
                    yield JSONServerSentEvent(event="error", data=error)
                    return
                yield JSONServerSentEvent(event="done", data={"run": meta, "response": response})
                return
            if event == "error":
                data |= {"run_id": initial_run["id"], "session_id": initial_run["session_id"]}
            yield JSONServerSentEvent(event=event, data=data)
            if event == "error":
                return
    finally:
        task.cancel()
        with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
            await task
