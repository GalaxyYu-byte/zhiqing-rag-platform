"""隔离实图组件对照：冻结虚构原文，比较 hybrid / 一跳 / 两跳。

使用真实 Neo4j；--models 使用现有 DashScope 向量、重排和回答服务。
授权仓储与查询实体采用固定组件适配器，不替代真实 HTTP/PG 权限验收。
不创建正式知识库文档；每份来源使用本次随机 namespace，并定向清理。
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path
from uuid import UUID, uuid4

from zhiqing_rag.core.config import PROJECT_ROOT, get_settings
from zhiqing_rag.modules.identity.auth_schemas import AuthIdentity, AuthTenant, AuthUser
from zhiqing_rag.query_answering.providers import (
    CHAIN_PROMPT,
    GENERATE_PROMPT,
    REPAIR_PROMPT,
    REVIEW_PROMPT,
    AnswerProvider,
)
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_retrieval.graph import AuthorizedGraphProjection, OnlineGraphService
from zhiqing_rag.query_retrieval.models import ChunkRecord, RecallHit
from zhiqing_rag.query_retrieval.providers import QueryEmbedder, QueryReranker
from zhiqing_rag.query_retrieval.service import HybridRetrievalService, in_scope
from zhiqing_rag.query_routing.models import GraphPlan, HybridPlan, RetrievalParams
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument
from zhiqing_rag.workers.graph_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    Extraction,
    validate_relation_evidence,
)
from zhiqing_rag.workers.graph_storage import GraphStore

DATASET = PROJECT_ROOT / "eval/query_graph/v2/cases.json"
DATASET_SHA256 = "76a3e94c34de01350496d20e2c8bf02aa4d26fb6402c552898ea3012f1470ecc"


class RecordingProvider(AnswerProvider):
    """仅本地已授权虚构验收记录草稿/复核，正式应用 provider 不记录原文。"""

    def __init__(self, settings):
        super().__init__(settings)
        self.events = []

    async def _call(self, prompt, payload, schema, temperature):
        stage = ("repair" if "error_code" in payload else
                 "review" if schema.__name__ == "AnswerReview" else "generate")
        try:
            result = await super()._call(prompt, payload, schema, temperature)
            self.events.append({"stage": stage, "value": result.value.model_dump(),
                                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
            return result
        except Exception as error:
            self.events.append(
                {"stage": stage, "error_type": type(error).__name__,
                 "error_code": getattr(error, "code", None)}
            )
            raise


def assess_result(case, mode, plan, retrieval, answer=None):
    """结构/原句检查与回答字段覆盖分开记；模型复核不是人工语义结论。"""
    evidence = {item["chunk_id"]: item for item in retrieval["evidence"]}
    paths = retrieval.get("graph_paths", ())
    trace = retrieval["trace"]

    def source_valid(source):
        item = evidence.get(source["chunk_id"])
        if not item or source["tenant_id"] != "1":
            return False
        if any(source[key] != item[key] for key in (
            "knowledge_base_id", "document_id", "revision_id", "generation_id", "chunk_id"
        )):
            return False
        quote, start, end = source["source_quote"], source["quote_start"], source["quote_end"]
        return (
            source["content_sha256"] == hashlib.sha256(item["content"].encode()).hexdigest()
            and isinstance(start, int) and isinstance(end, int) and 0 <= start < end
            and end <= len(item["content"]) and item["content"][start:end] == quote
            and any(all(source[key] == projection[key] for key in (
                "tenant_id", "document_id", "generation_id", "projection_id", "namespace"
            )) for projection in plan["params"].get("projections", ()))
        )

    def path_valid(path):
        if len(path["hops"]) != 2 or len(path["bridges"]) != 1:
            return False
        first, second = path["hops"]
        if (first["relation"]["predicate"], second["relation"]["predicate"]) != (
            "REQUIRES_MATERIAL", "HAS_FIELD"
        ):
            return False
        bridge = path["bridges"][0]
        sources = [hop["source"] for hop in path["hops"]] + list(bridge["sources"])
        return (
            first["source"]["document_id"] != second["source"]["document_id"]
            and first["source"]["namespace"] != second["source"]["namespace"]
            and first["relation"]["subject_entity_uid"] in plan["params"]["seed_entity_uids"]
            and bridge["source_entity_uid"] == first["relation"]["object_entity_uid"]
            and bridge["target_entity_uid"] == second["relation"]["subject_entity_uid"]
            and all(source_valid(source) for source in sources)
            and set(path["required_chunk_ids"]) == {source["chunk_id"] for source in sources}
            and all(hop["relation"]["source_quote"] == hop["source"]["source_quote"]
                    and hop["relation"] in evidence[hop["source"]["chunk_id"]]["graph_relations"]
                    for hop in path["hops"])
        )

    if mode == "two_hop":
        if case["expect_path"]:
            docs = {int(item["document_id"]) for item in evidence.values()}
            conditions = {
                hop["relation"]["condition_text"] for path in paths for hop in path["hops"]
            }
            path_passed = (
                bool(paths) and plan["route"] == "hybrid_graph"
                and plan["params"]["max_hops"] == 2 and trace["graph_hops"] == 2
                and trace["graph_path_count"] == len(paths)
                and all(path_valid(path) for path in paths)
                and set(case["required_documents"]) <= docs
                and case["required_condition"] in conditions
            )
        else:
            path_passed = not paths and trace["graph_hops"] <= 1
    elif mode == "one_hop":
        path_passed = not paths and trace["graph_hops"] <= 1
    else:
        path_passed = not paths and plan["route"] == "hybrid" and trace["graph_status"] == "skipped"

    answer_checks = None
    if answer is not None:
        statements = answer.get("statements", ())
        citations = [cite for statement in statements for cite in statement["citations"]]
        by_citation = {item["citation_id"]: item for item in evidence.values()}
        citation_valid = all(
            cite["citation_id"] in by_citation and bool(cite["quote"])
            and cite["quote"] in by_citation[cite["citation_id"]]["content"] for cite in citations
        )
        text = "\n".join(item["text"] for item in statements)
        quotes = "\n".join(cite["quote"] for cite in citations)
        cited_docs = {int(by_citation[cite["citation_id"]]["document_id"])
                      for cite in citations if cite["citation_id"] in by_citation}
        if case["expect_path"]:
            answer_checks = (
                answer["status"] == "answered" and bool(statements) and citation_valid
                and set(case["required_documents"]) <= cited_docs
                and all(field in text and field in quotes for field in case["expected_fields"])
                and case["required_condition"] in text and case["required_condition"] in quotes
            )
            # 题中使用原文明示的别名时，最终引用也必须包含该对应依据。
            for item in evidence.values():
                for clause in item["content"].splitlines():
                    if ("又称" in clause
                            and clause.split("又称", 1)[1].rstrip("。") in case["query"]):
                        answer_checks = answer_checks and clause in quotes
        else:
            answer_checks = answer["status"] != "answered" and not statements
    return {"path_passed": path_passed, "answer_evidence_checks_passed": answer_checks,
            "semantic_correct": None}


def load_dataset(path=DATASET):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA256:
        raise ValueError("冻结二版语料 SHA 不匹配，禁止模型调用")
    data = json.loads(raw)
    if data["dataset_kind"] != "synthetic_development":
        raise ValueError("只接受本地隔离虚构组件语料")
    return data


def fixture_record(document):
    content = document["content"]
    return ChunkRecord(
        tenant_id=1,
        knowledge_base_id=document["kb_id"],
        document_id=document["id"],
        revision_id=document["id"],
        generation_id=document["id"],
        chunk_id=document["id"],
        chunk_index=0,
        document_title=document["title"],
        document_name=document["title"] + ".txt",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        token_count=len(content),
        page_number=None,
        section_title=document["title"],
    )


def fixture_extraction(document):
    """从冻结原句确定性构建明确断言，避免抽取模型变化混入检索对照。"""
    entities, relations = [], []

    def entity(key, type_, name, quote, canonical=None):
        entities.append(
            ExtractedEntity(
                id=key,
                type=type_,
                name=name,
                canonical_name=canonical or name,
                evidence_quote=quote,
            )
        )

    for line in document["content"].splitlines():
        if match := re.fullmatch(r"(.+流程)包含(.+步骤)。", line):
            process, step = match.groups()
            entity("process", "Process", process, line)
            entity("step", "Step", step, line)
            relations.append(
                ExtractedRelation(
                    type="HAS_STEP",
                    source_id="process",
                    target_id="step",
                    evidence_quote=line,
                )
            )
        if match := re.fullmatch(r"(.+后)，(.+步骤)需要(.+表)。", line):
            condition, _, material = match.groups()
            entity("material", "Material", material, line)
            relations.append(
                ExtractedRelation(
                    type="REQUIRES_MATERIAL",
                    source_id="step",
                    target_id="material",
                    evidence_quote=line,
                    condition_text=condition,
                )
            )
        if match := re.fullmatch(r"(.+表)包含(.+)。", line):
            material, fields = match.groups()
            entity("material", "Material", material, line)
            for index, field in enumerate(re.split("[、和]", fields)):
                entity(f"field{index}", "Field", field, line)
                relations.append(
                    ExtractedRelation(
                        type="HAS_FIELD",
                        source_id="material",
                        target_id=f"field{index}",
                        evidence_quote=line,
                    )
                )
        if match := re.fullmatch(r"(.+表)又称(.+)。", line):
            canonical, alias = match.groups()
            entity("alias", "Material", alias, line, canonical)
        if match := re.fullmatch(r"(.+表)用于(.+流程)。", line):
            _, process = match.groups()
            if not any(item.id == "process" for item in entities):
                entity("process", "Process", process, line)
    value = Extraction(entities=entities, relations=relations)
    by_id = {item.id: item for item in value.entities}
    for relation in value.relations:
        validate_relation_evidence(relation, by_id, document["content"])
    return value


class FixtureRepository:
    def __init__(self, records, projections, vectors):
        self.records, self.visible_projections, self.vectors = records, projections, vectors

    async def projections(self, identity, scope):
        return tuple(p for p in self.visible_projections if p.document_id in scope.document_ids)

    async def chunks(self, identity, scope, projections, evidence):
        keys = {item["chunk_id"] for item in evidence}
        return tuple(
            item for item in self.records if item.chunk_id in keys and in_scope(item, scope)
        )

    async def revalidate(self, identity, scope, records):
        current = {item.chunk_id: item for item in self.records if in_scope(item, scope)}
        return tuple(current[item.chunk_id] for item in records if item.chunk_id in current)

    async def dense(self, identity, scope, vector, limit):
        ranked = []
        for record in self.records:
            if in_scope(record, scope):
                other = self.vectors[record.chunk_id]
                score = sum(a * b for a, b in zip(vector, other, strict=True)) / (
                    math.sqrt(sum(a * a for a in vector)) * math.sqrt(sum(b * b for b in other))
                )
                ranked.append(RecallHit(record, score))
        return tuple(sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit])

    async def bm25(self, identity, scope, query, limit):
        # 固定本地词面通道，三组完全相同；这是组件对照，不声称复现PG全文索引。
        pairs = {query[index : index + 2] for index in range(len(query) - 1)}
        hits = (
            RecallHit(item, float(sum(pair in item.content for pair in pairs)))
            for item in self.records
            if in_scope(item, scope)
        )
        return tuple(sorted(hits, key=lambda item: (-item.score, item.chunk.chunk_id))[:limit])


class FixedEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    async def embed(self, query):
        return self.vectors[query]


class FixedReranker:
    async def rerank(self, query, documents):
        return tuple((index, 1.0 / (index + 1)) for index in range(len(documents)))


def fixture_identity():
    return AuthIdentity(
        user_id=2,
        tenant_id=1,
        member_id=7,
        department_id=8,
        role="USER",
        clearance=1,
        auth_version=1,
        tenant_auth_version=1,
        session_key=UUID(int=1),
        user=AuthUser(id="2", email="fixture@example.invalid", display_name="组件验收"),
        tenant=AuthTenant(id="1", code="fixture", name="虚构组件语料", role="USER"),
    )


def fixture_scope(case):
    return AccessScope(
        tenant_id=1,
        user_id=2,
        knowledge_base_ids=(990,),
        document_ids=tuple(case["document_ids"]),
        published_documents=tuple(
            PublishedDocument(value, 990, value) for value in case["document_ids"]
        ),
    )


def checked_output(output):
    path = output.resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve() / "eval/query_graph/runs"):
        raise ValueError("输出必须在本项目 eval/query_graph/runs 内")
    if path.exists():
        raise ValueError("使用新运行目录，保留历史失败记录")
    path.mkdir(parents=True)
    return path


async def evaluate(args):
    data = load_dataset()
    output = checked_output(args.output)
    settings = get_settings().model_copy(update={"neo4j_enabled": True})
    records = tuple(fixture_record(document) for document in data["documents"])
    extractions = [fixture_extraction(document) for document in data["documents"]]
    namespace = "zhiqing-rag-platform:eval2:" + uuid4().hex
    stores, projections, results = [], [], []
    report = {
        "dataset_sha256": DATASET_SHA256,
        "dataset_kind": data["dataset_kind"],
        "real_neo4j": True,
        "real_models": args.models,
        "authorization_repository": "fixed_component_adapter",
        "entity_analysis": "frozen_grounded_mentions",
        "results": results,
        "cleanup": False,
        "namespace_prefix": namespace,
        "implementation_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/zhiqing_rag/query_retrieval/graph.py",
                "src/zhiqing_rag/query_retrieval/graph_identity.py",
                "src/zhiqing_rag/query_retrieval/paths.py",
                "src/zhiqing_rag/query_retrieval/fusion.py",
                "src/zhiqing_rag/query_retrieval/models.py",
                "src/zhiqing_rag/query_retrieval/service.py",
                "src/zhiqing_rag/query_answering/models.py",
                "src/zhiqing_rag/query_answering/providers.py",
                "src/zhiqing_rag/query_answering/service.py",
                "src/zhiqing_rag/query_answering/support.py",
                "src/zhiqing_rag/query_routing/models.py",
                "src/zhiqing_rag/workers/graph_storage.py",
                "src/zhiqing_rag/core/config.py",
            )
        },
    }

    def save():
        (output / "results.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for name, prompt in (
        ("generate", GENERATE_PROMPT), ("review", REVIEW_PROMPT), ("repair", REPAIR_PROMPT),
        ("chain", CHAIN_PROMPT)
    ):
        (output / (name + "_prompt.txt")).write_text(prompt, encoding="utf-8")

    try:
        for index, (record, extraction) in enumerate(zip(records, extractions, strict=True)):
            config = settings.model_copy(
                update={"graph_namespace": f"{namespace}:{record.document_id}"}
            )
            store = GraphStore(config)
            stores.append((store, record))
            await store.prepare()
            scope = {
                "tenant_id": 1,
                "kb_id": 990,
                "document_id": record.document_id,
                "revision_id": record.revision_id,
                "generation_id": record.generation_id,
            }
            await store.write_chunk(
                scope,
                {
                    "id": record.chunk_id,
                    "chunk_index": record.chunk_index,
                    "content": record.content,
                    "content_sha256": record.content_sha256,
                    "section_title": record.section_title,
                    "page_number": record.page_number,
                    "start_offset": 0,
                    "end_offset": len(record.content),
                    "metadata_": {},
                },
                extraction,
            )
            projections.append(
                AuthorizedGraphProjection(
                    1,
                    990,
                    record.document_id,
                    record.revision_id,
                    record.generation_id,
                    index + 1,
                    config.graph_namespace,
                )
            )
        embedder = QueryEmbedder(settings)
        doc_vectors = {
            record.chunk_id: await embedder.embed(record.content) if args.models else (1.0,)
            for record in records
        }
        query_vectors = {
            query: await embedder.embed(query) if args.models else (1.0,)
            for query in dict.fromkeys(case["query"] for case in data["cases"])
        }
        repository = FixtureRepository(records, projections, doc_vectors)
        identity = fixture_identity()
        for case in data["cases"]:
            for mode in ("hybrid", "one_hop", "two_hop"):
                started = time.perf_counter()
                config = settings.model_copy(
                    update={"query_graph_max_hops": 2 if mode == "two_hop" else 1}
                )
                graph = OnlineGraphService(config, repository=repository)
                scope = fixture_scope(case)
                params = RetrievalParams(top_k=2)
                ready = (
                    await graph.readiness(
                        identity, scope, case["query"], tuple(case["entities"]), params
                    )
                    if mode != "hybrid"
                    else None
                )
                common = {
                    "original_query": case["query"],
                    "query": case["query"],
                    "requested_route": "hybrid_graph" if mode != "hybrid" else "hybrid",
                    "reason_code": "GRAPH_APPROVED" if ready else "GRAPH_NOT_READY",
                }
                plan = (
                    GraphPlan(**common, params=ready.params)
                    if ready
                    else HybridPlan(**common, params=params)
                )
                retrieval = await HybridRetrievalService(
                    repository,
                    FixedEmbedder(query_vectors),
                    QueryReranker(config) if args.models else FixedReranker(),
                    graph_service=graph,
                    graph_timeout_seconds=config.query_graph_timeout_seconds,
                ).execute(plan=plan, scope=scope, identity=identity)
                answer = None
                provider = None
                if args.models:
                    provider = RecordingProvider(config)
                    retrieval, answer = await QueryAnswerService(
                        repository,
                        provider,
                        config,
                        graph_service=graph,
                    ).execute(plan=plan, retrieval=retrieval, scope=scope, identity=identity)
                plan_data, retrieval_data = plan.model_dump(), retrieval.model_dump()
                answer_data = answer.model_dump() if answer else None
                item = {
                    "id": case["id"],
                    "mode": mode,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "plan": plan_data,
                    "retrieval": retrieval_data,
                    "answer": answer_data,
                    "model_events": provider.events if provider else [],
                    **assess_result(case, mode, plan_data, retrieval_data, answer_data),
                }
                results.append(item)
                save()
                print(
                    json.dumps({key: item[key] for key in (
                        "id", "mode", "path_passed", "answer_evidence_checks_passed", "elapsed_ms"
                    )})
                )
    except Exception as error:
        # 保存中断原因，不保存异常消息，避免服务响应泄露连接或认证数据。
        report["run_error"] = {
            "error_type": type(error).__name__,
            "error_code": getattr(error, "code", None),
        }
        raise
    finally:
        cleanup_errors = []
        for store, record in stores:
            try:
                await store.delete_document(1, record.document_id, [store.settings.graph_namespace])
                # 隔离组件不保留已删除文档的 worker fence；只清本次新建 namespace。
                if not store.settings.graph_namespace.startswith(namespace + ":"):
                    raise ValueError("隔离图清理范围不匹配")
                async with store.driver.session(database=settings.neo4j_database) as session:
                    await session.run(
                        "MATCH (f:GraphFence {namespace:$namespace, tenant_id:1, "
                        "document_id:$document_id}) DETACH DELETE f",
                        namespace=store.settings.graph_namespace,
                        document_id=record.document_id,
                    )
                    result = await session.run(
                        "MATCH (n {namespace:$namespace}) RETURN count(n) AS count",
                        namespace=store.settings.graph_namespace,
                    )
                    if (await result.single())["count"]:
                        raise RuntimeError("隔离图节点清理不完整")
            except Exception as error:
                cleanup_errors.append(type(error).__name__)
            finally:
                await store.close()
        report["cleanup"] = not cleanup_errors
        report["cleanup_errors"] = cleanup_errors
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    return 0 if report_passed(report) else 1


def report_passed(report):
    """二跳是验收对象；基线回答只作对照，不把其结果改成成功。"""
    return report["cleanup"] and len(report["results"]) == 15 and all(
        item["path_passed"] and (
            item["mode"] != "two_hop" or item["answer_evidence_checks_passed"] is not False
        ) for item in report["results"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
