"""缓冲草稿，核对引用和语义支持，等待期间撤权则丢弃整份回答。"""

import hashlib
import json
import logging
import re
import time

import tiktoken

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.query_retrieval.models import ChunkRecord, RetrievalResult
from zhiqing_rag.query_retrieval.service import elapsed_ms, in_scope
from zhiqing_rag.query_routing.models import QueryPlan
from zhiqing_rag.query_routing.scope import AccessScope

from .models import AnswerProgress, AnswerResult, AnswerReview, AnswerTrace, DraftAnswer
from .providers import InvalidAnswer, evidence_payload, table_context_complete

logger = logging.getLogger(__name__)


class QueryAnswerService:
    def __init__(self, repository, provider, settings: Settings):
        self.repository = repository
        self.provider = provider
        self.settings = settings

    async def execute(
        self,
        *,
        identity: AuthIdentity,
        scope: AccessScope,
        plan: QueryPlan,
        retrieval: RetrievalResult,
        on_progress: AnswerProgress | None = None,
    ) -> tuple[RetrievalResult, AnswerResult]:
        started = time.perf_counter()
        metrics = {}

        def result(status, message=None, *, codes=(), statements=None, review_status="skipped"):
            metrics["total_ms"] = elapsed_ms(started)
            return AnswerResult(
                status=status,
                message=message,
                codes=codes,
                statements=statements or [],
                review_status=review_status,
                model=self.settings.chat_model,
                trace=AnswerTrace(**metrics),
            )

        if (identity.user_id, identity.tenant_id) != (scope.user_id, scope.tenant_id):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "回答范围与当前身份不一致")
        if retrieval.status in ("unsupported", "skipped"):
            await self._revalidate((), identity, scope, metrics)
            return retrieval, result(retrieval.status, plan.message or retrieval.message)
        evidence = await self._revalidate(retrieval.evidence, identity, scope, metrics)
        retrieval = self._with_evidence(retrieval, evidence)
        if not evidence:
            return retrieval, result("insufficient", "当前可读资料不足以回答这个问题。")
        if not self.settings.query_answer_enabled:
            return retrieval, result(
                "unavailable", "回答生成暂未启用。", codes=("ANSWER_DISABLED",)
            )
        context = json.dumps(
            {"question": plan.query, "evidence": evidence_payload(evidence)}, ensure_ascii=False
        )
        metrics["context_tokens"] = len(
            tiktoken.get_encoding("cl100k_base").encode(context, disallowed_special=())
        )
        if metrics["context_tokens"] > self.settings.query_answer_context_tokens:
            return retrieval, result(
                "insufficient", "资料范围过大，请缩小问题后重试。", codes=("ANSWER_CONTEXT_LIMIT",)
            )

        stage = "generation"
        try:
            if on_progress is not None:
                await on_progress("generation")
            start = time.perf_counter()
            try:
                generated = await self.provider.generate(plan.query, evidence)
            finally:
                metrics["generation_ms"] = elapsed_ms(start)
            self._usage(metrics, generated)
            draft = generated.value
            if not isinstance(draft, DraftAnswer):
                raise InvalidAnswer("草稿类型无效")
            current = await self._revalidate(evidence, identity, scope, metrics)
            if len(current) != len(evidence):
                return self._with_evidence(retrieval, current), result(
                    "rejected",
                    "资料或权限在处理期间发生变化，请重新提问。",
                    codes=("ANSWER_SCOPE_CHANGED",),
                    review_status="failed",
                )
            evidence = current
            retrieval = self._with_evidence(retrieval, evidence)
            if draft.status == "insufficient":
                return retrieval, result("insufficient", "当前可读资料不足以完整回答这个问题。")
            draft = self._validate_citations(draft, evidence)
            stage = "review"
            if on_progress is not None:
                await on_progress("review")
            start = time.perf_counter()
            try:
                reviewed = await self.provider.review(plan.query, evidence, draft)
            finally:
                metrics["review_ms"] = elapsed_ms(start)
            self._usage(metrics, reviewed)
            review = reviewed.value
            if not isinstance(review, AnswerReview):
                raise InvalidAnswer("复核类型无效")
            indexes = [item.index for item in review.statements]
            if len(indexes) != len(draft.statements) or set(indexes) != set(range(len(indexes))):
                raise InvalidAnswer("复核响应缺少陈述或索引重复")
            current = await self._revalidate(evidence, identity, scope, metrics)
            retrieval = self._with_evidence(retrieval, current)
            if len(current) != len(evidence):
                return retrieval, result(
                    "rejected",
                    "资料或权限在处理期间发生变化，请重新提问。",
                    codes=("ANSWER_SCOPE_CHANGED",),
                    review_status="failed",
                )
            if (
                review.missing_aspects
                or not review.sufficient
                or not all(item.supported for item in review.statements)
            ):
                return retrieval, result(
                    "insufficient",
                    "检索资料尚不足以支持完整回答。",
                    codes=("ANSWER_REVIEW_FAILED",),
                    review_status="failed",
                )
            return retrieval, result(
                "answered", statements=draft.statements, review_status="passed"
            )
        except AuthError:
            raise
        except Exception as exc:
            logger.warning(
                "回答阶段 %s 失败: %s code=%s",
                stage,
                type(exc).__name__,
                exc.code if isinstance(exc, InvalidAnswer) else "MODEL_UNAVAILABLE",
            )
            current = await self._revalidate(evidence, identity, scope, metrics)
            retrieval = self._with_evidence(retrieval, current)
            invalid = isinstance(exc, InvalidAnswer)
            code = exc.code if invalid else f"ANSWER_{stage.upper()}_UNAVAILABLE"
            incomplete_table = code == "ANSWER_TABLE_CONTEXT_INCOMPLETE"
            return retrieval, result(
                "insufficient" if incomplete_table else "rejected" if invalid else "unavailable",
                "续表缺少列名，尚不能确认完整答案，请查看原文。"
                if incomplete_table
                else "本次回答未能通过检查，请重试或查看原文。",
                codes=(code,),
                review_status="failed" if stage == "review" or invalid else "skipped",
            )

    @staticmethod
    def _usage(metrics, generated):
        metrics["input_tokens"] = metrics.get("input_tokens", 0) + generated.input_tokens
        metrics["output_tokens"] = metrics.get("output_tokens", 0) + generated.output_tokens

    async def _revalidate(self, evidence, identity, scope, metrics):
        start = time.perf_counter()
        records = tuple(
            ChunkRecord(
                tenant_id=identity.tenant_id,
                knowledge_base_id=int(item.knowledge_base_id),
                document_id=int(item.document_id),
                revision_id=int(item.revision_id),
                generation_id=int(item.generation_id),
                chunk_id=int(item.chunk_id),
                chunk_index=item.chunk_index,
                document_title=item.document_title,
                document_name=item.document_name,
                content=item.content,
                token_count=item.token_count,
                content_sha256=hashlib.sha256(item.content.encode()).hexdigest(),
                page_number=item.page_number,
                section_title=item.section_title,
            )
            for item in evidence
        )
        if any(not in_scope(record, scope) for record in records):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "回答证据超出当前范围")
        try:
            current = await self.repository.revalidate(identity, scope, records)
            by_key = {
                (
                    row.chunk_id,
                    row.document_id,
                    row.knowledge_base_id,
                    row.revision_id,
                    row.generation_id,
                    row.content,
                    row.content_sha256,
                ): row
                for row in current
                if in_scope(row, scope)
            }
            allowed = []
            for item, record in zip(evidence, records, strict=True):
                key = (
                    record.chunk_id,
                    record.document_id,
                    record.knowledge_base_id,
                    record.revision_id,
                    record.generation_id,
                    record.content,
                    record.content_sha256,
                )
                fresh = by_key.get(key)
                if fresh is not None:
                    allowed.append(
                        item.model_copy(
                            update={
                                "document_title": fresh.document_title,
                                "document_name": fresh.document_name,
                                "page_number": fresh.page_number,
                                "section_title": fresh.section_title,
                            }
                        )
                    )
            return tuple(allowed)
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError(
                503, "ANSWER_AUTHORIZATION_UNAVAILABLE", "无法确认资料权限，请稍后重试"
            ) from exc
        finally:
            metrics["authorization_ms"] = metrics.get("authorization_ms", 0) + elapsed_ms(start)

    @staticmethod
    def _with_evidence(retrieval, evidence):
        changed = len(evidence) != len(retrieval.evidence)
        return retrieval.model_copy(
            update={
                "evidence": evidence,
                "status": ("degraded" if changed else retrieval.status) if evidence else "empty",
                "degradation_codes": tuple(
                    sorted(
                        set(retrieval.degradation_codes) | ({"SCOPE_CHANGED"} if changed else set())
                    )
                ),
                "trace": retrieval.trace.model_copy(update={"returned_count": len(evidence)}),
            }
        )

    @staticmethod
    def _validate_citations(draft, evidence):
        by_id = {item.citation_id: item for item in evidence}
        if len(by_id) != len(evidence):
            raise InvalidAnswer("证据编号重复")
        statements = []
        for statement in draft.statements:
            if re.search(r"\[\s*[Ee][0-9]+\s*\]", statement.text):
                raise InvalidAnswer("引用标签必须由服务器展示")
            citations = []
            for citation in statement.citations:
                source = by_id.get(citation.citation_id)
                if source is None:
                    raise InvalidAnswer("引用不存在", "ANSWER_REFERENCE_UNKNOWN")
                if not table_context_complete(source):
                    raise InvalidAnswer("续表缺少列名", "ANSWER_TABLE_CONTEXT_INCOMPLETE")
                if citation.quote not in source.content:
                    quote = QueryAnswerService._restore_line_breaks(citation.quote, source.content)
                    if quote is None:
                        raise InvalidAnswer("引文不属于原文", "ANSWER_QUOTE_NOT_VERBATIM")
                    citation = citation.model_copy(update={"quote": quote})
                citations.append(citation)
            statements.append(statement.model_copy(update={"citations": citations}))
        return draft.model_copy(update={"statements": statements})

    @staticmethod
    def _restore_line_breaks(quote: str, content: str) -> str | None:
        # 只恢复 PDF 等解析器的 CR/LF；不改空格、标点、词语、否定或实体名。
        needle = quote.replace("\r", "").replace("\n", "")
        if len(needle) < 8:
            return None
        offsets = [index for index, char in enumerate(content) if char not in "\r\n"]
        flat = "".join(content[index] for index in offsets)
        start = flat.find(needle)
        if start < 0 or flat.find(needle, start + 1) >= 0:
            return None
        return content[offsets[start] : offsets[start + len(needle) - 1] + 1]
