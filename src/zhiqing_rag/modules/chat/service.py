"""短事务领取轮次，授权历史改写后复用单轮回答，提交结果后才允许发布。"""

import asyncio
import hashlib
import json
from contextlib import suppress
from uuid import uuid4

import anyio

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.query_analysis.models import QueryAnalysisResult, QueryFeatures
from zhiqing_rag.query_answering.models import AnswerProgress, AnswerResult, QueryAnswerResponse
from zhiqing_rag.query_answering.router import execute_answer
from zhiqing_rag.query_retrieval.models import RetrievalResult
from zhiqing_rag.query_routing.models import PlanQueryBody

from .rewrite import RewriteResult, depends_on_history
from .schemas import ChatAnswerBody, ChatRunView


def canonical_request(body: ChatAnswerBody) -> dict:
    """省略与 null 等价，选择顺序不改变请求语义；范围参与幂等摘要。"""
    return {
        "question": body.query,
        "client_request_id": body.client_request_id,
        "knowledge_base_ids": sorted(body.knowledge_base_ids, key=int)
        if body.knowledge_base_ids
        else None,
        "document_ids": sorted(body.document_ids, key=int) if body.document_ids else None,
        "filters": sorted(
            (item.model_dump(mode="json") for item in body.filters),
            key=lambda item: (item["field"], item["operator"], item["value"]),
        ),
        "offset": body.offset,
        "limit": body.limit,
    }


def request_digest(request: dict) -> str:
    return hashlib.sha256(
        json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ChatService:
    def __init__(self, repository, rewriter, settings: Settings):
        self.repository = repository
        self.rewriter = rewriter
        self.settings = settings

    async def begin(self, identity: AuthIdentity, session_id: int, body: ChatAnswerBody) -> dict:
        request = canonical_request(body)
        run = await self.repository.start_run(
            identity,
            session_id,
            request,
            request_digest(request),
            str(uuid4()),
            self.settings.chat_run_lease_seconds,
        )
        if not run["is_new"] and run["status"] in ("PENDING", "RUNNING"):
            raise AuthError(409, "CHAT_RUN_IN_PROGRESS", "该轮正在执行，请恢复会话查看结果。")
        return run

    async def execute(
        self,
        *,
        identity,
        session_id,
        body,
        run,
        scope,
        analyzer,
        planner,
        retriever,
        answerer,
        on_progress: AnswerProgress | None = None,
    ) -> ChatRunView:
        run_id = int(run["id"])
        if not run["is_new"]:
            return ChatRunView.model_validate(
                await self.repository.get_run(identity, session_id, run_id)
            )
        lease = {"lease_owner": run["lease_owner"], "lease_epoch": run["lease_epoch"]}
        try:
            # 范围加载可能等待连接；开始任何模型工作前确认领取租约仍然有效。
            await self.repository.heartbeat(
                identity,
                session_id,
                run_id,
                **lease,
                lease_seconds=self.settings.chat_run_lease_seconds,
            )
        except (Exception, asyncio.CancelledError) as error:
            await self.fail_before_execution(identity, session_id, run, error)
            raise
        reported_stages = set()

        async def progress(stage):
            if on_progress is not None and stage not in reported_stages:
                reported_stages.add(stage)
                await on_progress(stage)

        async def heartbeat():
            while True:
                await asyncio.sleep(self.settings.chat_run_lease_seconds / 3)
                await self.repository.heartbeat(
                    identity,
                    session_id,
                    run_id,
                    **lease,
                    lease_seconds=self.settings.chat_run_lease_seconds,
                )

        async def work():
            async with asyncio.timeout(self.settings.chat_run_timeout_seconds):
                history = await self._history(identity, session_id, run_id, scope)
                await progress("analysis")
                if not history and depends_on_history(body.query):
                    rewritten = RewriteResult(
                        question=None, clarification="请补充本轮要讨论的对象名称和条件。"
                    )
                else:
                    rewritten = await self.rewriter.rewrite(body.query, history)
                used_ids = set(getattr(rewritten, "used_run_ids", ()))
                used_history = [item for item in history if str(item["run_id"]) in used_ids]
                # 改写等待期间发生撤权时，不能继续使用历史中揭示的对象。
                for item in used_history:
                    current = await self.repository.get_run(
                        identity, session_id, int(item["run_id"])
                    )
                    if current["visibility"] != "visible":
                        raise AuthError(
                            409, "CHAT_CONTEXT_CHANGED", "上文资料已变化，请重新确认问题。"
                        )
                if rewritten.clarification is not None:
                    response = self._clarification(body.query, rewritten.clarification, scope)
                    standalone_question = None
                else:
                    standalone_question = rewritten.question
                    if not standalone_question:
                        raise AuthError(
                            503, "CHAT_REWRITE_UNAVAILABLE", "暂时无法理解追问，请补充对象。"
                        )
                    effective = PlanQueryBody(
                        query=standalone_question,
                        knowledge_base_ids=body.knowledge_base_ids,
                        document_ids=body.document_ids,
                        filters=body.filters,
                        offset=body.offset,
                        limit=body.limit,
                    )
                    response = await execute_answer(
                        effective,
                        identity,
                        scope,
                        analyzer,
                        planner,
                        retriever,
                        answerer,
                        progress if on_progress is not None else None,
                    )
                    response = response.model_copy(
                        update={
                            "plan": response.plan.model_copy(update={"original_query": body.query})
                        }
                    )
                trace = response.answer.trace
                response = response.model_copy(
                    update={
                        "answer": response.answer.model_copy(
                            update={
                                "trace": trace.model_copy(
                                    update={
                                        "input_tokens": trace.input_tokens + rewritten.input_tokens,
                                        "output_tokens": trace.output_tokens
                                        + rewritten.output_tokens,
                                    }
                                )
                            }
                        )
                    }
                )
                # 完成事务同时复查历史来源及本轮证据；所有 SSE 段落共用这一份已提交结果。
                await self.repository.complete(
                    identity,
                    session_id,
                    run_id,
                    **lease,
                    response=response.model_dump(mode="json"),
                    standalone_question=standalone_question,
                    history_run_ids=[int(item["run_id"]) for item in used_history],
                    **(
                        {
                            "structured_provenance": response.structured.provenance.model_dump(
                                mode="json"
                            )
                        }
                        if response.structured is not None
                        and response.structured.provenance is not None
                        else {}
                    ),
                )
            return ChatRunView.model_validate(
                await self.repository.get_run(identity, session_id, run_id)
            )

        task = asyncio.create_task(work())
        renewal = asyncio.create_task(heartbeat())
        try:
            done, _ = await asyncio.wait((task, renewal), return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return await task
            await renewal  # 心跳失败，终止工作；完成提交仍会校验租约代次。
            raise AuthError(409, "CHAT_LEASE_LOST", "该轮执行已失效，请恢复会话。")
        except asyncio.CancelledError:
            task.cancel()
            renewal.cancel()
            await self._terminate(
                identity, session_id, run_id, lease, "CANCELLED", "CHAT_CANCELLED"
            )
            raise
        except TimeoutError as error:
            task.cancel()
            renewal.cancel()
            await self._terminate(identity, session_id, run_id, lease, "FAILED", "CHAT_TIMEOUT")
            raise AuthError(503, "CHAT_TIMEOUT", "本轮等待超时，请恢复会话查看状态。") from error
        except Exception as error:
            task.cancel()
            renewal.cancel()
            code = error.code if isinstance(error, AuthError) else "CHAT_EXECUTION_FAILED"
            await self._terminate(identity, session_id, run_id, lease, "FAILED", code)
            raise
        finally:
            task.cancel()
            renewal.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, renewal, return_exceptions=True)

    async def _terminate(self, identity, session_id, run_id, lease, status, code):
        with anyio.CancelScope(shield=True), suppress(Exception):
            # 取消后仅做有界短事务清理；进程崩溃由下一次访问的租约回收兜底。
            async with asyncio.timeout(10):
                await self.repository.terminate(
                    identity, session_id, run_id, **lease, status=status, error_code=code
                )

    async def fail_before_execution(self, identity, session_id, run, error):
        await self._terminate(
            identity,
            session_id,
            int(run["id"]),
            {"lease_owner": run["lease_owner"], "lease_epoch": run["lease_epoch"]},
            "CANCELLED" if isinstance(error, asyncio.CancelledError) else "FAILED",
            error.code if isinstance(error, AuthError) else "CHAT_SCOPE_UNAVAILABLE",
        )

    async def _history(self, identity, session_id, run_id, scope):
        page = await self.repository.list_runs(
            identity, session_id, before_run_id=run_id, limit=self.settings.chat_history_turns
        )
        allowed = {
            (str(doc.document_id), str(doc.knowledge_base_id), str(doc.generation_id))
            for doc in scope.published_documents
        }
        history = []
        for previous in page["runs"]:
            response = previous.get("response")
            if (
                previous["status"] != "SUCCEEDED"
                or previous["visibility"] != "visible"
                or not response
            ):
                # 不把跳过的最近轮次偷换成更早的“刚才/第二条”。
                history.clear()
                continue
            if response.get(
                "structured"
            ) is not None and not await self.repository.structured_history_in_scope(
                identity, session_id, int(previous["id"]), scope
            ):
                history.clear()
                continue
            if any(
                (item["document_id"], item["knowledge_base_id"], item["generation_id"])
                not in allowed
                for item in response["retrieval"]["evidence"]
            ):
                history.clear()
                continue
            history.append(
                {
                    "run_id": previous["id"],
                    "question": previous["question"],
                    "answer": response["answer"],
                }
            )
        return history

    @staticmethod
    def _clarification(question, message, scope):
        from zhiqing_rag.query_routing.models import ClarifyParams, ClarifyPlan

        return QueryAnswerResponse(
            analysis=QueryAnalysisResult(
                status="ok",
                original_query=question,
                analysis=QueryFeatures(
                    intent="other",
                    entities=[],
                    keywords=[],
                    constraints=[],
                    relation_required=False,
                    needs_context=True,
                    clarification_question=message,
                ),
                fallback_hint=None,
                message=None,
                degradation_reason=None,
                model="context-rewrite",
                latency_ms=0,
            ),
            plan=ClarifyPlan(
                original_query=question,
                query=question,
                requested_route="clarify",
                reason_code="CHAT_CONTEXT_CLARIFICATION",
                params=ClarifyParams(question=message),
            ),
            scope=scope.summary(),
            retrieval=RetrievalResult(status="skipped", message=message),
            answer=AnswerResult(status="skipped", message=message),
        )
