"""缓冲草稿，核对引用和语义支持，等待期间撤权则丢弃整份回答。"""

import asyncio
import hashlib
import json
import logging
import re
import time

import tiktoken

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.query_retrieval.fusion import FusedCandidate
from zhiqing_rag.query_retrieval.graph import OnlineGraphService
from zhiqing_rag.query_retrieval.models import ChunkRecord, RetrievalResult
from zhiqing_rag.query_retrieval.paths import complete_paths
from zhiqing_rag.query_retrieval.service import elapsed_ms, in_scope
from zhiqing_rag.query_routing.models import GraphPlan, QueryPlan
from zhiqing_rag.query_routing.scope import AccessScope

from .models import AnswerProgress, AnswerResult, AnswerReview, AnswerTrace, DraftAnswer
from .providers import (
    InvalidAnswer,
    build_repair_payload,
    chain_payload,
    evidence_payload,
    review_evidence_payload,
    table_context_complete,
)
from .support import (
    confirmation_guard_failure,
    expand_selected_path_quotes,
    graph_condition_diagnostics,
    path_citations_complete,
    semantic_guard_failure,
    source_scope_ambiguous,
)

logger = logging.getLogger(__name__)


class QueryAnswerService:
    def __init__(self, repository, provider, settings: Settings, *, graph_service=None):
        self.repository = repository
        self.provider = provider
        self.settings = settings
        self.graph_service = (
            graph_service if graph_service is not None else OnlineGraphService(settings)
        )

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
        retrieval = self._with_evidence(retrieval, evidence, tenant_id=identity.tenant_id)
        retrieval, evidence = await self._revalidate_paths(
            retrieval, evidence, identity, scope, plan, metrics
        )
        if not evidence:
            return retrieval, result("insufficient", "当前可读资料不足以回答这个问题。")
        if not self.settings.query_answer_enabled:
            return retrieval, result(
                "unavailable", "回答生成暂未启用。", codes=("ANSWER_DISABLED",)
            )
        context_payload = {"question": plan.query,
                           "evidence": evidence_payload(evidence, plan.query)}
        if retrieval.graph_paths:
            context_payload["evidence_chains"] = chain_payload(evidence, retrieval.graph_paths)
        context = json.dumps(context_payload, ensure_ascii=False)
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
                # SSE 队列可背压等待；等待结束后出网前再次授权，而不是沿用旧快照。
                current = await self._revalidate(evidence, identity, scope, metrics)
                retrieval = self._with_evidence(retrieval, current, tenant_id=identity.tenant_id)
                retrieval, current = await self._revalidate_paths(
                    retrieval, current, identity, scope, plan, metrics
                )
                if len(current) != len(evidence):
                    return retrieval, result(
                        "rejected", "资料或权限在处理期间发生变化，请重新提问。",
                        codes=("ANSWER_SCOPE_CHANGED",), review_status="failed",
                    )
                evidence = current
                context_payload = {"question": plan.query,
                                   "evidence": evidence_payload(evidence, plan.query)}
                if retrieval.graph_paths:
                    context_payload["evidence_chains"] = chain_payload(
                        evidence, retrieval.graph_paths
                    )
                tokens = len(tiktoken.get_encoding("cl100k_base").encode(
                    json.dumps(context_payload, ensure_ascii=False), disallowed_special=()
                ))
                metrics["context_tokens"] = max(metrics["context_tokens"], tokens)
                if tokens > self.settings.query_answer_context_tokens:
                    return retrieval, result("insufficient", "资料范围过大，请缩小问题后重试。",
                                             codes=("ANSWER_CONTEXT_LIMIT",))
            start = time.perf_counter()
            try:
                method = getattr(self.provider, "generate_with_paths", None)
                generated = await method(plan.query, evidence, retrieval.graph_paths) if (
                    retrieval.graph_paths and callable(method)
                ) else await self.provider.generate(plan.query, evidence)
            finally:
                metrics["generation_ms"] = elapsed_ms(start)
            self._usage(metrics, generated)
            draft = generated.value
            if not isinstance(draft, DraftAnswer):
                raise InvalidAnswer("草稿类型无效")
            current = await self._revalidate(evidence, identity, scope, metrics)
            retrieval = self._with_evidence(retrieval, current, tenant_id=identity.tenant_id)
            retrieval, current = await self._revalidate_paths(
                retrieval, current, identity, scope, plan, metrics
            )
            if len(current) != len(evidence):
                return retrieval, result(
                    "rejected",
                    "资料或权限在处理期间发生变化，请重新提问。",
                    codes=("ANSWER_SCOPE_CHANGED",),
                    review_status="failed",
                )
            evidence = current
            retrieval = self._with_evidence(retrieval, evidence, tenant_id=identity.tenant_id)
            if draft.status == "insufficient":
                return retrieval, result("insufficient", "当前可读资料不足以完整回答这个问题。")
            try:
                draft = self._validate_draft(draft, evidence, plan.query, retrieval.graph_paths)
            except InvalidAnswer as error:
                repair = getattr(self.provider, "repair", None)
                if not callable(repair) or error.code not in {
                    "ANSWER_QUOTE_NOT_VERBATIM", "ANSWER_SEMANTIC_GUARD_FAILED",
                    "ANSWER_SOURCE_SCOPE_AMBIGUOUS", "ANSWER_GRAPH_PATH_NOT_CITED",
                    "ANSWER_CONFIRMATION_GUARD_FAILED",
                    "ANSWER_GRAPH_CONDITION_NOT_PRESERVED",
                }:
                    raise
                # 修复是另一次外发；先重新授权，不能沿用上次模型调用前的快照。
                current = await self._revalidate(evidence, identity, scope, metrics)
                retrieval = self._with_evidence(retrieval, current, tenant_id=identity.tenant_id)
                retrieval, current = await self._revalidate_paths(
                    retrieval, current, identity, scope, plan, metrics
                )
                if len(current) != len(evidence):
                    return retrieval, result(
                        "rejected", "资料或权限在处理期间发生变化，请重新提问。",
                        codes=("ANSWER_SCOPE_CHANGED",), review_status="failed",
                    )
                evidence = current
                repair_payload = build_repair_payload(
                    plan.query, evidence, draft, error.code, retrieval.graph_paths
                )
                tokens = len(tiktoken.get_encoding("cl100k_base").encode(
                    json.dumps(repair_payload, ensure_ascii=False), disallowed_special=()
                ))
                metrics["context_tokens"] = max(metrics["context_tokens"], tokens)
                if tokens > self.settings.query_answer_context_tokens:
                    raise
                metrics["repair_count"] = 1
                start = time.perf_counter()
                try:
                    method = getattr(self.provider, "repair_with_paths", None)
                    generated = await method(
                        plan.query, evidence, draft, error.code, retrieval.graph_paths
                    ) if retrieval.graph_paths and callable(method) else await repair(
                        plan.query, evidence, draft, error.code
                    )
                finally:
                    metrics["generation_ms"] += elapsed_ms(start)
                self._usage(metrics, generated)
                draft = generated.value
                if not isinstance(draft, DraftAnswer):
                    raise InvalidAnswer("修复草稿类型无效")
                current = await self._revalidate(evidence, identity, scope, metrics)
                retrieval = self._with_evidence(retrieval, current, tenant_id=identity.tenant_id)
                retrieval, current = await self._revalidate_paths(
                    retrieval, current, identity, scope, plan, metrics
                )
                if len(current) != len(evidence):
                    return retrieval, result(
                        "rejected", "资料或权限在处理期间发生变化，请重新提问。",
                        codes=("ANSWER_SCOPE_CHANGED",), review_status="failed",
                    )
                evidence = current
                if draft.status == "insufficient":
                    return retrieval, result("insufficient", "当前可读资料不足以完整回答这个问题。")
                # 一次受限修复后再次走全部检查，不再重试，不能因模型修复而豁免。
                draft = self._validate_draft(draft, evidence, plan.query, retrieval.graph_paths)
            stage = "review"
            if on_progress is not None:
                await on_progress("review")
                current = await self._revalidate(evidence, identity, scope, metrics)
                retrieval = self._with_evidence(retrieval, current, tenant_id=identity.tenant_id)
                retrieval, current = await self._revalidate_paths(
                    retrieval, current, identity, scope, plan, metrics
                )
                if len(current) != len(evidence):
                    return retrieval, result(
                        "rejected", "资料或权限在处理期间发生变化，请重新提问。",
                        codes=("ANSWER_SCOPE_CHANGED",), review_status="failed",
                    )
                evidence = current
                draft = self._validate_draft(draft, evidence, plan.query, retrieval.graph_paths)
            review_payload = {
                "question": plan.query, "evidence": review_evidence_payload(evidence, plan.query),
                "draft": draft.model_dump(),
            }
            if retrieval.graph_paths:
                review_payload["evidence_chains"] = chain_payload(evidence, retrieval.graph_paths)
            tokens = len(tiktoken.get_encoding("cl100k_base").encode(
                json.dumps(review_payload, ensure_ascii=False), disallowed_special=()
            ))
            metrics["context_tokens"] = max(metrics["context_tokens"], tokens)
            if tokens > self.settings.query_answer_context_tokens:
                return retrieval, result(
                    "insufficient", "资料引用范围过大，请缩小问题后重试。",
                    codes=("ANSWER_CONTEXT_LIMIT",),
                )
            start = time.perf_counter()
            try:
                method = getattr(self.provider, "review_with_paths", None)
                reviewed = await method(plan.query, evidence, draft, retrieval.graph_paths) if (
                    retrieval.graph_paths and callable(method)
                ) else await self.provider.review(plan.query, evidence, draft)
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
            retrieval = self._with_evidence(retrieval, current, tenant_id=identity.tenant_id)
            retrieval, current = await self._revalidate_paths(
                retrieval, current, identity, scope, plan, metrics
            )
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
            if retrieval.graph_paths and (
                not path_citations_complete(draft, current, retrieval.graph_paths)
                or graph_condition_diagnostics(draft, current, retrieval.graph_paths)
            ):
                # 回答等待时已使用的路径可能退休，其他未引用路径不能冒充其出处。
                # 当前可读原文及已核实陈述仍可保留，剥离路径元数据即可。
                retrieval = self._with_evidence(
                    retrieval.model_copy(update={
                        "graph_paths": (), "status": "degraded",
                        "degradation_codes": tuple(sorted(
                            set(retrieval.degradation_codes) | {"GRAPH_PROVENANCE_CHANGED"}
                        )),
                    }), current, tenant_id=identity.tenant_id,
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
            retrieval = self._with_evidence(retrieval, current, tenant_id=identity.tenant_id)
            retrieval, current = await self._revalidate_paths(
                retrieval, current, identity, scope, plan, metrics
            )
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

    @staticmethod
    def _records(evidence, tenant_id):
        return tuple(
            ChunkRecord(
                tenant_id=tenant_id,
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

    async def _revalidate(self, evidence, identity, scope, metrics):
        start = time.perf_counter()
        records = self._records(evidence, identity.tenant_id)
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
    def _with_evidence(retrieval, evidence, *, tenant_id=None):
        changed = len(evidence) != len(retrieval.evidence)
        paths = ()
        if tenant_id is not None and retrieval.graph_paths:
            records = QueryAnswerService._records(evidence, tenant_id)
            candidates = tuple(
                FusedCandidate(
                    record, item.scores, item.graph_relations,
                    tuple(path for path in retrieval.graph_paths
                          if item.chunk_id in path.required_chunk_ids),
                )
                for item, record in zip(evidence, records, strict=True)
            )
            paths = complete_paths(candidates)
        paths_changed = len(paths) != len(retrieval.graph_paths)
        return retrieval.model_copy(
            update={
                "evidence": evidence,
                "graph_paths": paths,
                "status": (
                    "degraded" if changed or paths_changed else retrieval.status
                ) if evidence else "empty",
                "degradation_codes": tuple(
                    sorted(
                        set(retrieval.degradation_codes)
                        | ({"SCOPE_CHANGED"} if changed else set())
                        | ({"GRAPH_PROVENANCE_CHANGED"} if paths_changed else set())
                    )
                ),
                "trace": retrieval.trace.model_copy(update={
                    "returned_count": len(evidence), "graph_path_count": len(paths),
                    "graph_hops": 2 if paths else (
                        1 if any(item.graph_relations for item in evidence) else 0
                    ),
                }),
            }
        )

    async def _revalidate_paths(self, retrieval, evidence, identity, scope, plan, metrics):
        if not retrieval.graph_paths:
            return retrieval, evidence
        started = time.perf_counter()
        sources = [source for path in retrieval.graph_paths for source in (
            *(hop.source for hop in path.hops),
            *(source for bridge in path.bridges for source in bridge.sources),
        )]
        valid_plan = isinstance(plan, GraphPlan) and plan.params.max_hops == 2
        if valid_plan:
            refs = {(
                item.tenant_id, item.document_id, item.generation_id,
                item.projection_id, item.namespace,
            ) for item in plan.params.projections}
            valid_plan = all(
                (source.tenant_id, source.document_id, source.generation_id,
                 source.projection_id, source.namespace) in refs
                and int(source.tenant_id) == identity.tenant_id
                for source in sources
            ) and all(
                path.hops[0].relation.seed_entity_uid in plan.params.seed_entity_uids
                for path in retrieval.graph_paths
            )
        code = None
        allowed = frozenset()
        try:
            if valid_plan:
                needed = {source.chunk_id for source in sources}
                records = self._records(
                    tuple(item for item in evidence if item.chunk_id in needed), identity.tenant_id
                )
                async with asyncio.timeout(self.settings.query_graph_timeout_seconds):
                    allowed = await self.graph_service.revalidate_relations(
                        identity, scope, plan.params, records
                    )
            if not valid_plan or any(int(source.chunk_id) not in allowed for source in sources):
                code = "GRAPH_PROVENANCE_CHANGED"
        except AuthError:
            raise
        except Exception as exc:
            logger.warning("回答图出处复核失败: %s", type(exc).__name__)
            code = "GRAPH_PROVENANCE_UNAVAILABLE"
        finally:
            duration = elapsed_ms(started)
            metrics["authorization_ms"] = metrics.get("authorization_ms", 0) + duration
        paths = tuple(
            path for path in retrieval.graph_paths
            if valid_plan and all(int(value) in allowed for value in path.required_chunk_ids)
        )
        updates = {"graph_paths": paths}
        if code:
            evidence = tuple(
                item if int(item.chunk_id) in allowed else item.model_copy(update={
                    "graph_relations": (),
                    "scores": item.scores.model_copy(update={
                        "graph_rank": None, "graph_score": None,
                    }),
                })
                for item in evidence
            )
            updates.update(
                evidence=evidence,
                status="degraded" if evidence else "empty",
                degradation_codes=tuple(sorted(set(retrieval.degradation_codes) | {code})),
            )
        updates["trace"] = retrieval.trace.model_copy(update={
            "graph_ms": retrieval.trace.graph_ms + duration,
            "graph_status": "failed" if code == "GRAPH_PROVENANCE_UNAVAILABLE" else (
                "succeeded" if any(item.graph_relations for item in evidence) else "empty"
            ),
            "graph_path_count": len(paths),
            "graph_count": sum(bool(item.graph_relations) for item in evidence),
            "graph_assertion_count": len({relation.assertion_id
                for item in evidence for relation in item.graph_relations}),
            "graph_hops": 2 if paths else (
                1 if any(item.graph_relations for item in evidence) else 0
            ),
        })
        retrieval = retrieval.model_copy(update=updates)
        # 图投影复查会等待数据库；之后再次核实原文，不能错过等待期间的撤权。
        current = await self._revalidate(evidence, identity, scope, metrics)
        return self._with_evidence(retrieval, current, tenant_id=identity.tenant_id), current

    @staticmethod
    def _validate_draft(draft, evidence, question="", paths=()):
        draft = QueryAnswerService._validate_citations(draft, evidence)
        if paths:
            draft = expand_selected_path_quotes(draft, evidence, paths)
            draft = QueryAnswerService._validate_citations(draft, evidence)
        if semantic_guard_failure(draft, evidence):
            raise InvalidAnswer(
                "风险陈述未保留原文条件、责任主体、结论与否定",
                "ANSWER_SEMANTIC_GUARD_FAILED",
            )
        if paths and not path_citations_complete(draft, evidence, paths):
            raise InvalidAnswer("回答缺少完整路径原文引用", "ANSWER_GRAPH_PATH_NOT_CITED")
        if paths and graph_condition_diagnostics(draft, evidence, paths):
            raise InvalidAnswer("回答正文没有保留已引用路径的原文条件",
                                "ANSWER_GRAPH_CONDITION_NOT_PRESERVED")
        if source_scope_ambiguous(question, draft, evidence, paths):
            raise InvalidAnswer("不同主题的独立清单被混合", "ANSWER_SOURCE_SCOPE_AMBIGUOUS")
        if confirmation_guard_failure(question, draft, evidence):
            raise InvalidAnswer("确认阶段原句缺失或混入其他状态/否定条件",
                                "ANSWER_CONFIRMATION_GUARD_FAILED")
        return draft

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
