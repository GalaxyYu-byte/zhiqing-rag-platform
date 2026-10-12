"""执行混合与图增强计划；独立召回、明确降级、返回前复查权限。"""

import asyncio
import logging
import time
from dataclasses import replace
from typing import Protocol

from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.query_routing.models import GraphPlan, HybridPlan, QueryPlan
from zhiqing_rag.query_routing.scope import AccessScope

from .facets import cover_facets, fuse_facet_rankings, rerank_queries
from .fusion import FusedCandidate, fuse_candidates
from .models import ChunkRecord, Evidence, RecallHit, RetrievalResult, RetrievalTrace
from .paths import complete_paths, retain_complete_paths, select_path_bundles

logger = logging.getLogger(__name__)


class RetrievalRepository(Protocol):
    async def dense(self, identity, scope, vector, limit) -> tuple[RecallHit, ...]: ...

    async def bm25(self, identity, scope, query, limit) -> tuple[RecallHit, ...]: ...

    async def revalidate(self, identity, scope, chunks) -> tuple[ChunkRecord, ...]: ...


class Embedder(Protocol):
    async def embed(self, query: str) -> tuple[float, ...]: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, documents: tuple[str, ...]
    ) -> tuple[tuple[int, float], ...]: ...


def elapsed_ms(start: float) -> int:
    return max(0, round((time.perf_counter() - start) * 1000))


def in_scope(chunk: ChunkRecord, scope: AccessScope) -> bool:
    return chunk.tenant_id == scope.tenant_id and any(
        (chunk.document_id, chunk.knowledge_base_id, chunk.generation_id)
        == (doc.document_id, doc.knowledge_base_id, doc.generation_id)
        for doc in scope.published_documents
    )


class HybridRetrievalService:
    def __init__(
        self,
        repository: RetrievalRepository,
        embedder: Embedder,
        reranker: Reranker,
        *,
        reranker_enabled: bool = True,
        graph_service=None,
        graph_timeout_seconds: float = 3.0,
    ):
        self.repository = repository
        self.embedder = embedder
        self.reranker = reranker
        self.reranker_enabled = reranker_enabled
        self.graph_service = graph_service
        self.graph_timeout_seconds = graph_timeout_seconds

    async def execute(
        self, *, plan: QueryPlan, scope: AccessScope, identity: AuthIdentity
    ) -> RetrievalResult:
        started = time.perf_counter()
        if (identity.tenant_id, identity.user_id) != (scope.tenant_id, scope.user_id):
            raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "检索范围与当前身份不一致")
        if plan.planning_status != "planned":
            return RetrievalResult(status=plan.planning_status, message=plan.message)
        if plan.resolved_filters:
            if scope.resolved_filters and scope.resolved_filters != plan.resolved_filters:
                raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "计划过滤条件与执行范围不一致")
            scope = replace(scope, resolved_filters=plan.resolved_filters)
        if not isinstance(plan, (HybridPlan, GraphPlan)):
            return RetrievalResult(
                status="skipped" if plan.route in ("clarify", "none") else "unsupported",
                message="此计划无需文档检索"
                if plan.route in ("clarify", "none")
                else "该路径的执行器尚未接入",
            )
        if any(constraint.field != "other" for constraint in plan.constraints) and not (
            plan.resolved_filters or scope.resolved_filters
        ):
            return RetrievalResult(status="unsupported", message="元数据过滤尚未接入")
        if not scope.published_documents:
            return RetrievalResult(status="empty", message="当前范围没有可检索的已发布文档")
        if isinstance(plan, GraphPlan):
            approved = {
                (str(doc.document_id), str(doc.generation_id)) for doc in scope.published_documents
            }
            if any(
                ref.tenant_id != str(scope.tenant_id)
                or (ref.document_id, ref.generation_id) not in approved
                for ref in plan.params.projections
            ):
                raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "图谱计划超出授权范围")

        metrics: dict = {}
        degradation = []

        async def graph_branch():
            if not isinstance(plan, GraphPlan):
                return ()
            start = time.perf_counter()
            try:
                if self.graph_service is None:
                    raise RuntimeError("图检索未配置")
                async with asyncio.timeout(self.graph_timeout_seconds):
                    result = await self.graph_service.recall(
                        identity, scope, plan.params, plan.params.candidate_k
                    )
                if any(not in_scope(hit.chunk, scope) for hit in result.hits):
                    raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "图召回结果超出授权范围")
                metrics["graph_count"] = len(result.hits)
                metrics["graph_seed_count"] = result.seed_count
                metrics["graph_assertion_count"] = result.assertion_count
                metrics["graph_path_candidate_count"] = getattr(result, "path_count", 0)
                metrics["graph_hops"] = 2 if getattr(result, "path_count", 0) else (
                    1 if result.hits else 0
                )
                metrics["graph_status"] = "succeeded" if result.hits else "empty"
                if not result.hits:
                    degradation.append("GRAPH_NO_EVIDENCE")
                return result.hits
            except AuthError:
                raise
            except Exception as error:
                self._log_failure("graph", error)
                degradation.append("GRAPH_UNAVAILABLE")
                metrics["graph_status"] = "failed"
                return ()
            finally:
                metrics["graph_ms"] = elapsed_ms(start)

        async def dense_branch():
            start = time.perf_counter()
            try:
                vector = await self.embedder.embed(plan.query)
            except AuthError:
                raise
            except Exception as error:
                self._log_failure("embedding", error)
                degradation.append("EMBEDDING_UNAVAILABLE")
                metrics["dense_status"] = "failed"
                return ()
            finally:
                metrics["embedding_ms"] = elapsed_ms(start)
            return await recall("dense", vector)

        async def recall(channel: str, query):
            start = time.perf_counter()
            try:
                method = getattr(self.repository, channel)
                hits = await method(identity, scope, query, plan.params.candidate_k)
                # 仓储执行实时 SQL 授权，此处再保证结果没有扩大服务器原定范围。
                if any(not in_scope(hit.chunk, scope) for hit in hits):
                    raise AuthError(403, "QUERY_SCOPE_FORBIDDEN", "检索结果超出授权范围")
                metrics[f"{channel}_status"] = "succeeded"
                metrics[f"{channel}_count"] = len(hits)
                return hits
            except AuthError:
                raise
            except Exception as error:
                self._log_failure(channel, error)
                degradation.append(f"{channel.upper()}_UNAVAILABLE")
                metrics[f"{channel}_status"] = "failed"
                return ()
            finally:
                metrics[f"{channel}_ms"] = elapsed_ms(start)

        # 每路在仓储中使用自己的连接，失败事务不会污染另一条召回链。
        branches = await asyncio.gather(
            dense_branch(), recall("bm25", plan.query), graph_branch(), return_exceptions=True
        )
        for branch in branches:
            if isinstance(branch, BaseException):
                raise branch
        dense, bm25, graph_hits = branches
        if metrics["dense_status"] == metrics["bm25_status"] == "failed" and not graph_hits:
            raise AuthError(503, "RETRIEVAL_UNAVAILABLE", "检索服务暂不可用，请稍后重试")
        start = time.perf_counter()
        try:
            fused = fuse_candidates(dense, bm25, graph=graph_hits, limit=plan.params.candidate_k)
        except Exception as error:
            self._log_failure("fusion", error)
            raise AuthError(503, "RETRIEVAL_UNAVAILABLE", "检索结果暂不可用，请稍后重试") from error
        metrics["fusion_ms"] = elapsed_ms(start)
        metrics["fused_count"] = len(fused)
        if fused:
            fused = await self._revalidate(fused, identity, scope, metrics)
            if len(fused) < metrics["fused_count"]:
                degradation.append("SCOPE_CHANGED")
        metrics["authorized_count"] = len(fused)

        ranked = fused
        facet_rankings = ()
        facet_anchors = ()
        if not plan.params.rerank or not self.reranker_enabled:
            metrics["rerank_status"] = "disabled"
        elif len(fused) > 1:
            start = time.perf_counter()
            try:
                queries = rerank_queries(plan.query)
                if plan.params.top_k < len(queries) - 1:
                    queries = (plan.query,)
                metrics["rerank_query_count"] = len(queries)
                documents = tuple(self._rerank_text(item.chunk) for item in fused)
                results = await asyncio.gather(
                    *(self.reranker.rerank(query, documents) for query in queries),
                    return_exceptions=True,
                )
                rankings = []
                for scores in results:
                    if isinstance(scores, BaseException):
                        raise scores
                    rankings.append(self._apply_rerank(fused, scores))
                ranked = rankings[0]
                facet_rankings = fuse_facet_rankings(fused, tuple(rankings[1:]))
                facet_anchors = tuple(items[0] for items in rankings[1:])
                metrics["rerank_status"] = "succeeded"
            except Exception as error:
                self._log_failure("rerank", error)
                metrics["rerank_status"] = "failed"
                degradation.append("RERANKER_UNAVAILABLE")
            finally:
                metrics["rerank_ms"] = elapsed_ms(start)
            # 重排可能等待网络：再次检查会话、ACL、删除/撤回、活动代次。
            ranked = await self._revalidate(ranked, identity, scope, metrics)
            if len(ranked) < len(fused):
                degradation.append("SCOPE_CHANGED")
            authorized_ids = {item.chunk.chunk_id for item in ranked}
            facet_rankings = tuple(
                tuple(item for item in items if item.chunk.chunk_id in authorized_ids)
                for items in facet_rankings
            )
            facet_anchors = tuple(
                item for item in facet_anchors if item.chunk.chunk_id in authorized_ids
            )

        # 图投影可能在重排等待期间被替换；已授权原文仍可用，旧关系出处须剥离。
        if isinstance(plan, GraphPlan):
            ranked = await self._revalidate_graph_provenance(
                ranked, identity, scope, plan, metrics, degradation
            )
            # 最后的投影检查可能已观察到撤权；其后必须再次过滤原文，不能只删关系。
            before = len(ranked)
            ranked = await self._revalidate(ranked, identity, scope, metrics)
            if len(ranked) < before:
                degradation.append("SCOPE_CHANGED")
                metrics["graph_count"] = sum(bool(item.graph_relations) for item in ranked)
                metrics["graph_assertion_count"] = len(
                    {relation.assertion_id for item in ranked for relation in item.graph_relations}
                )
                if not metrics["graph_count"]:
                    metrics["graph_hops"] = 0
                    if metrics.get("graph_status") == "succeeded":
                        metrics["graph_status"] = "empty"
            authorized_ids = {item.chunk.chunk_id for item in ranked}
            facet_rankings = tuple(
                tuple(item for item in items if item.chunk.chunk_id in authorized_ids)
                for items in facet_rankings
            )
            facet_anchors = tuple(
                item for item in facet_anchors if item.chunk.chunk_id in authorized_ids
            )
        metrics["authorized_count"] = len(ranked)
        covered = cover_facets(
            ranked, facet_rankings, limit=plan.params.top_k, anchors=facet_anchors
        )
        covered_ids = {item.chunk.chunk_id for item in covered}
        ordered = (*covered, *(item for item in ranked if item.chunk.chunk_id not in covered_ids))
        selected = select_path_bundles(ordered, limit=plan.params.top_k)
        paths = complete_paths(selected)
        metrics["graph_path_count"] = len(paths)
        if isinstance(plan, GraphPlan) and plan.params.max_hops == 2 and not paths:
            degradation.append("GRAPH_PATH_INCOMPLETE")
        if paths:
            metrics["graph_hops"] = 2
        elif metrics.get("graph_hops") == 2:
            metrics["graph_hops"] = 1 if any(item.graph_relations for item in selected) else 0
        evidence = tuple(
            self._evidence(candidate, rank)
            for rank, candidate in enumerate(selected, start=1)
        )
        metrics["returned_count"] = len(evidence)
        metrics["total_ms"] = elapsed_ms(started)
        return RetrievalResult(
            status=("degraded" if degradation else "ok") if evidence else "empty",
            evidence=evidence,
            graph_paths=paths,
            degradation_codes=tuple(sorted(set(degradation))),
            message=None if evidence else "当前授权范围内没有找到匹配原文",
            trace=RetrievalTrace(**metrics),
        )

    async def _revalidate_graph_provenance(
        self, candidates, identity, scope, plan, metrics, degradation
    ):
        method = getattr(self.graph_service, "revalidate_relations", None)
        graph_chunks = tuple(
            item.chunk for item in candidates if item.graph_relations or item.graph_paths
        )
        if method is None or not graph_chunks:
            return candidates
        start = time.perf_counter()
        try:
            async with asyncio.timeout(self.graph_timeout_seconds):
                allowed = await method(identity, scope, plan.params, graph_chunks)
            if all(chunk.chunk_id in allowed for chunk in graph_chunks):
                return candidates
            degradation.append("GRAPH_PROVENANCE_CHANGED")
        except AuthError:
            raise
        except Exception as error:
            self._log_failure("graph_authorization", error)
            degradation.append("GRAPH_PROVENANCE_UNAVAILABLE")
            allowed = frozenset()
            metrics["graph_status"] = "failed"
        finally:
            metrics["graph_ms"] = metrics.get("graph_ms", 0) + elapsed_ms(start)
        current = tuple(
            replace(
                item,
                graph_relations=(),
                graph_paths=(),
                scores=item.scores.model_copy(update={"graph_rank": None, "graph_score": None}),
            )
            if (item.graph_relations or item.graph_paths) and item.chunk.chunk_id not in allowed
            else item
            for item in candidates
        )
        current = retain_complete_paths(current, allowed_chunk_ids=allowed)
        metrics["graph_count"] = sum(bool(item.graph_relations) for item in current)
        metrics["graph_assertion_count"] = len(
            {relation.assertion_id for item in current for relation in item.graph_relations}
        )
        if not metrics["graph_count"]:
            metrics["graph_hops"] = 0
            if metrics["graph_status"] != "failed":
                metrics["graph_status"] = "empty"
        return current

    async def _revalidate(self, candidates, identity, scope, metrics):
        start = time.perf_counter()
        try:
            records = await self.repository.revalidate(
                identity, scope, tuple(item.chunk for item in candidates)
            )
            by_id = {record.chunk_id: record for record in records if in_scope(record, scope)}
            filtered = []
            for item in candidates:
                current = by_id.get(item.chunk.chunk_id)
                # 已发布分块应不可变；如原文在等待期间变化，不沿用旧排序或出处。
                if current is not None and (
                    current.revision_id,
                    current.content,
                    current.content_sha256,
                ) == (item.chunk.revision_id, item.chunk.content, item.chunk.content_sha256):
                    filtered.append(replace(item, chunk=current))
            return retain_complete_paths(tuple(filtered))
        except AuthError:
            raise
        except Exception as error:
            self._log_failure("authorization", error)
            raise AuthError(503, "RETRIEVAL_UNAVAILABLE", "无法确认原文权限，请稍后重试") from error
        finally:
            metrics["authorization_ms"] = metrics.get("authorization_ms", 0) + elapsed_ms(start)

    @staticmethod
    def _apply_rerank(candidates, scores):
        # 边界也复查供应商结果，保证测试替身/后续新 provider 不会混淆索引。
        import math

        if len(scores) != len(candidates) or {item[0] for item in scores} != set(
            range(len(candidates))
        ):
            raise ValueError("重排响应索引不完整")
        if any(
            type(index) is not int or type(score) not in (int, float) or not math.isfinite(score)
            for index, score in scores
        ):
            raise ValueError("重排响应含无效索引或分数")
        return tuple(
            replace(
                candidates[index],
                scores=candidates[index].scores.model_copy(update={"rerank_score": float(score)}),
            )
            for index, score in sorted(scores, key=lambda item: (-item[1], item[0]))
        )

    @staticmethod
    def _rerank_text(chunk: ChunkRecord) -> str:
        # 步骤分块未必重复主题名称；授权后的标题与章节帮助重排理解正文对象。
        return (
            f"文档标题：{chunk.document_title}\n文件名：{chunk.document_name}\n"
            f"章节：{chunk.section_title or ''}\n原文：\n{chunk.content}"
        )

    @staticmethod
    def _evidence(candidate: FusedCandidate, rank: int) -> Evidence:
        chunk = candidate.chunk
        return Evidence(
            citation_id=f"E{rank}",
            rank=rank,
            chunk_id=str(chunk.chunk_id),
            knowledge_base_id=str(chunk.knowledge_base_id),
            document_id=str(chunk.document_id),
            revision_id=str(chunk.revision_id),
            generation_id=str(chunk.generation_id),
            document_title=chunk.document_title,
            document_name=chunk.document_name,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            token_count=chunk.token_count,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            scores=candidate.scores,
            graph_relations=candidate.graph_relations,
        )

    @staticmethod
    def _log_failure(stage: str, error: Exception):
        logger.warning("检索阶段 %s 失败: %s", stage, type(error).__name__)
