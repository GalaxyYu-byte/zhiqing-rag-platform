"""执行 Router 计划，保留规则决策并记录实际降级及最终检索参数。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ..query_analysis.multi_query import MultiQuerySummary, expansion_block_reason
from ..query_analysis.rewrite import RewriteSummary
from ..query_routing.models import ExecutionRecord, RetrievalPlan, RoutingDecision
from .graph_retrieval_service import GraphRetrievalResult
from .hybrid_retrieval_service import HybridRetrievalResult, fuse_dense_bm25_rrf, fuse_dense_bm25_graph_rrf
from .multi_query_retrieval_service import retrieve_expansions
from .query_preprocessing_service import PreparedQuery
from .reranker_service import RerankerService
from .retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalExecutionResult:
    hybrid: HybridRetrievalResult
    chunks: list[RetrievedChunk]
    graph: GraphRetrievalResult
    routing: RoutingDecision
    rewrite: RewriteSummary
    multi_query: MultiQuerySummary
    execution: ExecutionRecord
    engine: str
    reranked: bool
    retrieval_ms: int
    expansion_ms: int


class RetrievalExecutor:
    def __init__(
        self, *, dense_search: Callable[..., Any], bm25_search: Callable[..., Any],
        graph_search: Callable[..., Any], reranker_factory: Callable[..., Any],
        multi_query_factory: Callable[..., Any], multi_query_enabled: bool, empty_retrieval_expansion: bool,
        expansion_retrieval_timeout_seconds: float,
    ) -> None:
        self.dense_search = dense_search
        self.bm25_search = bm25_search
        self.graph_search = graph_search
        self.reranker_factory = reranker_factory
        self.multi_query_factory = multi_query_factory
        self.multi_query_enabled = multi_query_enabled
        self.empty_retrieval_expansion = empty_retrieval_expansion
        self.expansion_retrieval_timeout_seconds = expansion_retrieval_timeout_seconds

    async def execute(self, session: Any, *, plan: RetrievalPlan, prepared: PreparedQuery) -> RetrievalExecutionResult:
        if plan.decision.action != "retrieve":
            raise ValueError("Executor 只执行检索计划")
        if not plan.kb_ids or any(kb_id <= 0 for kb_id in plan.kb_ids):
            raise ValueError("执行计划缺少有效知识库范围")
        if plan.doc_ids is not None and (not plan.doc_ids or any(doc_id <= 0 for doc_id in plan.doc_ids)):
            raise ValueError("执行计划包含无效授权文档范围")
        if plan.query != prepared.request.query:
            raise ValueError("执行计划不属于当前预处理问题")
        if not 1 <= plan.top_k <= plan.candidate_k <= 100:
            raise ValueError("执行计划的候选或结果预算无效")
        if plan.decision.path == "hybrid_graph" and (not plan.doc_ids or not plan.seed_entity_uids):
            raise ValueError("Graph 执行计划必须包含授权文档和已解析种子")
        requested_plan = plan
        hypothetical_document = prepared.hypothetical_document if plan.use_hyde else None
        candidate_k, top_k, rerank = plan.candidate_k, plan.top_k, plan.rerank
        kb_ids = list(plan.kb_ids)
        doc_ids = list(plan.doc_ids) if plan.doc_ids is not None else None
        expansion = MultiQuerySummary("disabled" if not self.multi_query_enabled else "skipped", (plan.query,),
                                      reason="disabled" if not self.multi_query_enabled else "no_trigger")
        expansion_ms, expansion_inside_retrieval_ms = 0, 0
        block = expansion_block_reason(prepared.analysis, hyde_selected=prepared.rewrite.method == "hyde")
        if self.multi_query_enabled and block:
            expansion = replace(expansion, reason=block)

        async def expand(trigger):
            nonlocal expansion_ms
            started = time.perf_counter()
            service = self.multi_query_factory()
            try:
                return await service.expand(prepared.request, prepared.analysis, trigger=trigger,
                                            hyde_selected=prepared.rewrite.method == "hyde")
            finally:
                await service.aclose()
                expansion_ms += round((time.perf_counter() - started) * 1000)

        if self.multi_query_enabled and not block and plan.use_multi_query:
            expansion = await expand("analyzer")
        retrieval_started = time.perf_counter()
        dense = await self.dense_search(
            session,
            query=plan.query,
            kb_ids=kb_ids,
            doc_ids=doc_ids,
            top_k=candidate_k,
            min_score=plan.min_dense_score,
            **({"embedding_text": hypothetical_document} if hypothetical_document else {}),
        )
        bm25 = await self.bm25_search(
            session,
            query=plan.query,
            kb_ids=kb_ids,
            doc_ids=doc_ids,
            top_k=candidate_k,
        )
        graph = GraphRetrievalResult(
            query=plan.query, engine="graph_skipped", latency_ms=0, results=[], matches=(),
        )
        routing = plan.decision
        rewrite = prepared.rewrite
        hyde_degraded = getattr(dense, "hyde_degraded", False)
        if hyde_degraded:
            rewrite = replace(
                rewrite, status="degraded", degradation_reason="hyde_embedding_failed",
                warnings=(*rewrite.warnings, "HyDE 向量生成失败，已使用问题本身生成向量。"),
            )
            routing = replace(routing, warnings=(*routing.warnings, *rewrite.warnings))
        if routing.path == "hybrid_graph":
            graph_failure = None
            try:
                async with asyncio.timeout(plan.graph_timeout_seconds):
                    graph = await self.graph_search(
                        session, query=plan.query, kb_ids=kb_ids, doc_ids=doc_ids,
                        top_k=candidate_k, max_hops=plan.max_hops,
                        seed_entity_uids=list(plan.seed_entity_uids),
                    )
                if not graph.results:
                    graph_failure = "graph_execution_empty"
            except TimeoutError:
                graph_failure = "graph_execution_timeout"
            except Exception:
                graph_failure = "graph_execution_unavailable"
            if graph_failure:
                routing = replace(
                    routing, path="hybrid", reason=graph_failure, graph_degraded=True,
                    warnings=(*routing.warnings, "图谱召回失败或没有当前证据，本次使用 Dense + BM25。"),
                )
                plan = replace(
                    plan, decision=routing,
                    dense_weight=plan.hybrid_fallback_weights[0],
                    bm25_weight=plan.hybrid_fallback_weights[1], graph_weight=0.0,
                )
                graph = GraphRetrievalResult(
                    query=plan.query, engine=graph_failure, latency_ms=0, results=[], matches=(),
                )

        def fuse(current_dense, current_bm25):
            if routing.path == "hybrid_graph":
                return fuse_dense_bm25_graph_rrf(
                    current_dense, current_bm25, graph, top_k=candidate_k,
                    dense_weight=plan.dense_weight, bm25_weight=plan.bm25_weight,
                    graph_weight=plan.graph_weight,
                )
            result = fuse_dense_bm25_rrf(
                current_dense, current_bm25, top_k=candidate_k,
                dense_weight=plan.dense_weight, bm25_weight=plan.bm25_weight,
            )
            return replace(result, engine=f"{result.engine}+graph_fallback") if routing.graph_degraded else result

        hybrid = fuse(dense, bm25)
        if (
            self.multi_query_enabled and self.empty_retrieval_expansion and not block
            and not hybrid.results and expansion.trigger is None
        ):
            prior_expansion_ms = expansion_ms
            expansion = await expand("empty_retrieval")
            expansion_inside_retrieval_ms = expansion_ms - prior_expansion_ms
        dense, bm25, expansion = await retrieve_expansions(
            session, dense=dense, bm25=bm25, expansion=expansion,
            dense_search=self.dense_search, bm25_search=self.bm25_search,
            kb_ids=kb_ids, doc_ids=doc_ids, candidate_k=candidate_k,
            timeout_seconds=self.expansion_retrieval_timeout_seconds,
        )
        if len(expansion.applied_queries) > 1:
            hybrid = fuse(dense, bm25)
            hybrid = replace(hybrid, engine=f"{hybrid.engine}+multi_query_rrf")
        if expansion.warnings:
            routing = replace(routing, warnings=(*routing.warnings, *expansion.warnings))
        hybrid = HybridRetrievalResult(
            query=hybrid.query,
            engine=hybrid.engine,
            latency_ms=hybrid.latency_ms,
            results=self._deduplicate_chunks(hybrid.results),
            candidate_scores=hybrid.candidate_scores,
        )
        result_chunks = hybrid.results[:top_k]
        engine = hybrid.engine
        reranked = False
        rerank_degraded = False
        if rerank and hybrid.results:
            reranker: RerankerService | None = None
            try:
                reranker = self.reranker_factory()
                ranked = await reranker.rerank(
                    hybrid,
                    top_n=min(top_k, len(hybrid.results)),
                )
                result_chunks = ranked.results
                engine = ranked.engine
                reranked = True
            except Exception:
                rerank_degraded = True
                logger.exception("Reranker 不可用，保留融合排序")
            finally:
                if reranker is not None:
                    await reranker.aclose()

        retrieval_ms = max(
            0, round((time.perf_counter() - retrieval_started) * 1000) - expansion_inside_retrieval_ms
        )
        degradations = []
        if prepared.analysis.status == "degraded":
            degradations.append("analysis_degraded")
        if routing.graph_degraded:
            degradations.append(routing.reason)
        if rewrite.status == "degraded" and rewrite.degradation_reason:
            degradations.append(rewrite.degradation_reason)
        if expansion.degradation_reason:
            degradations.append(expansion.degradation_reason)
        if rerank_degraded:
            degradations.append("reranker_unavailable")
        graph_attempted = requested_plan.decision.path == "hybrid_graph"
        execution = ExecutionRecord(
            path=routing.path, query=plan.query, kb_ids=plan.kb_ids, doc_ids=plan.doc_ids,
            candidate_k=candidate_k, top_k=top_k,
            weights=(plan.dense_weight, plan.bm25_weight, plan.graph_weight),
            min_dense_score=plan.min_dense_score, graph_attempted=graph_attempted,
            graph_max_hops=requested_plan.max_hops if graph_attempted else None,
            graph_timeout_seconds=requested_plan.graph_timeout_seconds if graph_attempted else None,
            graph_seed_entity_uids=requested_plan.seed_entity_uids if graph_attempted else (),
            rerank_requested=rerank, reranked=reranked,
            dense_input="hypothetical_document" if hypothetical_document and not hyde_degraded else "query",
            applied_queries=expansion.applied_queries or (plan.query,),
            degradation_reasons=tuple(dict.fromkeys(degradations)),
        )
        return RetrievalExecutionResult(
            hybrid, result_chunks, graph, routing, rewrite, expansion, execution,
            engine, reranked, retrieval_ms, expansion_ms,
        )

    @staticmethod
    def _deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        unique: list[RetrievedChunk] = []
        seen_content: set[str] = set()
        for chunk in chunks:
            content_key = " ".join(chunk.content.split()).casefold()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            unique.append(chunk)
        return unique
