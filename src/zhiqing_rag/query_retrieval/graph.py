"""授权投影上的在线有界图召回；映射和每条业务断言都须回读当前原文。"""

import asyncio
import hashlib
import re
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import get_args

from neo4j import AsyncGraphDatabase, Query
from starlette.concurrency import run_in_threadpool

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.modules.identity.auth_schemas import AuthIdentity
from zhiqing_rag.modules.knowledge.permissions import document_access_sql
from zhiqing_rag.query_metadata.compiler import compile_filters
from zhiqing_rag.query_routing.models import GraphParams, GraphProjectionRef, RetrievalParams
from zhiqing_rag.query_routing.scope import AccessScope
from zhiqing_rag.query_routing.service import GraphReadiness
from zhiqing_rag.workers.graph_extraction import (
    ENDPOINTS,
    EntityType,
    ExtractedRelation,
    normalize_name,
    stable_key,
    validate_relation_evidence,
)

from .graph_identity import (
    VerifiedMention,
    explicit_aliases,
    explicit_contexts,
    identity_bridge,
    identity_conflict,
)
from .models import (
    ChunkRecord,
    GraphHopEvidence,
    GraphPathEvidence,
    GraphRelationEvidence,
    GraphSourceEvidence,
    RecallHit,
)
from .repository import (
    _CHUNK_COLUMNS,
    QueryRetrievalRepository,
    _check_limit,
    _check_scope,
    _chunk_key,
    _record,
)

_UID = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True)
class AuthorizedGraphProjection:
    tenant_id: int
    knowledge_base_id: int
    document_id: int
    revision_id: int
    generation_id: int
    projection_id: int
    namespace: str

    def ref(self) -> GraphProjectionRef:
        return GraphProjectionRef(
            tenant_id=str(self.tenant_id),
            document_id=str(self.document_id),
            generation_id=str(self.generation_id),
            projection_id=str(self.projection_id),
            namespace=self.namespace,
        )

    def parameters(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "kb_id": self.knowledge_base_id,
            "document_id": self.document_id,
            "revision_id": self.revision_id,
            "generation_id": self.generation_id,
            "namespace": self.namespace,
        }


@dataclass(frozen=True, slots=True)
class GraphRecallResult:
    hits: tuple[RecallHit, ...]
    seed_count: int = 0
    assertion_count: int = 0
    path_count: int = 0


class OnlineGraphRepository(QueryRetrievalRepository):
    """独立连接刷新身份；图范围和原文范围都取初始授权与当前授权的交集。"""

    async def projections(
        self, identity: AuthIdentity, scope: AccessScope
    ) -> tuple[AuthorizedGraphProjection, ...]:
        _check_scope(identity, scope)
        return await run_in_threadpool(self._projections, identity, scope)

    def _projections(self, identity, scope):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            if not scope.published_documents:
                return ()
            predicate, arguments = document_access_sql(fresh, published_only=True)
            metadata_predicate, metadata_arguments = compile_filters(scope.resolved_filters)
            initial = scope.published_documents
            rows = self._query(
                connection,
                """SELECT gp.tenant_id, doc.kb_id AS knowledge_base_id, gp.document_id,
                          gp.revision_id, gp.generation_id, gp.id AS projection_id,
                          gp.graph_namespace AS namespace
                   FROM {schema}.graph_projection gp
                   JOIN {schema}.document doc ON doc.tenant_id = gp.tenant_id
                     AND doc.id = gp.document_id AND doc.active_generation_id = gp.generation_id
                   JOIN {schema}.knowledge_base kb ON kb.tenant_id = doc.tenant_id
                     AND kb.id = doc.kb_id
                   JOIN {schema}.index_generation g ON g.tenant_id = gp.tenant_id
                     AND g.document_id = gp.document_id AND g.revision_id = gp.revision_id
                     AND g.id = gp.generation_id
                    JOIN {schema}.document_revision r ON r.tenant_id = gp.tenant_id
                      AND r.document_id = gp.document_id AND r.id = gp.revision_id
                   WHERE """
                + predicate
                + " AND " + metadata_predicate
                + """ AND gp.status = 'READY' AND g.status = 'READY'
                      AND gp.tenant_id = %s
                      AND (doc.id, doc.kb_id, gp.generation_id) IN (
                        SELECT * FROM unnest(%s::bigint[], %s::bigint[], %s::bigint[])
                      ) ORDER BY gp.id""",
                (
                    *arguments,
                    *metadata_arguments,
                    scope.tenant_id,
                    [item.document_id for item in initial],
                    [item.knowledge_base_id for item in initial],
                    [item.generation_id for item in initial],
                ),
            )
            allowed = {
                (item.document_id, item.knowledge_base_id, item.generation_id) for item in initial
            }
            return tuple(
                AuthorizedGraphProjection(**row)
                for row in rows
                if row["tenant_id"] == scope.tenant_id
                and (row["document_id"], row["knowledge_base_id"], row["generation_id"]) in allowed
            )

    async def chunks(self, identity, scope, projections, evidence) -> tuple[ChunkRecord, ...]:
        _check_scope(identity, scope)
        return await run_in_threadpool(self._chunks, identity, scope, projections, evidence)

    def _chunks(self, identity, scope, projections, evidence):
        with database_connection(self.settings) as connection:
            fresh = self._refresh_identity(connection, identity)
            if not projections or not evidence:
                return ()
            authorized, parameters = self._authorized(fresh, scope)
            self._configure_planner(connection)
            statement = (
                "SELECT "
                + _CHUNK_COLUMNS
                + authorized
                + """ AND EXISTS (
                    SELECT 1 FROM {schema}.graph_projection gp
                    WHERE gp.tenant_id = c.tenant_id AND gp.document_id = c.document_id
                      AND gp.revision_id = c.revision_id AND gp.generation_id = c.generation_id
                      AND gp.status = 'READY'
                      AND (gp.id, gp.graph_namespace, gp.document_id, gp.generation_id) IN (
                        SELECT * FROM unnest(%s::bigint[], %s::text[], %s::bigint[], %s::bigint[])
                      )
                  ) AND (c.id, c.document_id, c.kb_id, c.revision_id, c.generation_id) IN (
                    SELECT * FROM unnest(
                      %s::bigint[], %s::bigint[], %s::bigint[], %s::bigint[], %s::bigint[]
                    )
                  )"""
            )
            keys = tuple(dict.fromkeys(_evidence_key(item) for item in evidence))
            arrays = tuple([key[index] for key in keys] for index in (5, 2, 1, 3, 4))
            rows = self._query(
                connection,
                statement,
                (
                    *parameters,
                    [item.projection_id for item in projections],
                    [item.namespace for item in projections],
                    [item.document_id for item in projections],
                    [item.generation_id for item in projections],
                    *arrays,
                ),
            )
            return tuple(record for record in map(_record, rows) if _chunk_key(record) in keys)


def _projection_predicate(alias: str, *, revision=False) -> str:
    columns = ("namespace", "tenant_id", "kb_id", "document_id", "generation_id")
    if revision:
        columns += ("revision_id",)
    return (
        "any(p IN $projections WHERE "
        + " AND ".join(f"{alias}.{column}=p.{column}" for column in columns)
        + ")"
    )


_SEEDS_QUERY = (
    "MATCH (n:Entity)-[:MENTIONED_IN]->(e:Evidence) "
    "WHERE (n.normalized_name IN $names OR n.key IN $uids) "
    "AND n.namespace=e.namespace AND n.tenant_id=e.tenant_id AND n.kb_id=e.kb_id AND "
    + _projection_predicate("e", revision=True)
    + " RETURN DISTINCT n.normalized_name AS normalized_name, n.key AS entity_uid, "
    "n.type AS entity_type ORDER BY normalized_name, entity_uid LIMIT $row_limit"
)
_RELATIONS_QUERY = (
    "MATCH (a:Assertion)-[:SUBJECT]->(s:Entity), (a)-[:OBJECT]->(o:Entity), "
    "(a)-[:SUPPORTED_BY]->(e:Evidence) "
    "WHERE (s.key IN $uids OR o.key IN $uids) AND a.predicate IN $predicates AND "
    + _projection_predicate("a")
    + " AND "
    + _projection_predicate("e", revision=True)
    + " AND a.namespace=e.namespace AND a.tenant_id=e.tenant_id AND a.kb_id=e.kb_id "
    "AND a.document_id=e.document_id AND a.generation_id=e.generation_id "
    "AND s.namespace=a.namespace AND s.tenant_id=a.tenant_id AND s.kb_id=a.kb_id "
    "AND o.namespace=a.namespace AND o.tenant_id=a.tenant_id AND o.kb_id=a.kb_id "
    "AND EXISTS { MATCH (s)-[:MENTIONED_IN]->(se:Evidence) WHERE "
    + _projection_predicate("se", revision=True)
    + " AND se.document_id=a.document_id AND se.generation_id=a.generation_id } "
    "AND EXISTS { MATCH (o)-[:MENTIONED_IN]->(oe:Evidence) WHERE "
    + _projection_predicate("oe", revision=True)
    + " AND oe.document_id=a.document_id AND oe.generation_id=a.generation_id } "
    "RETURN DISTINCT a.key AS assertion_id, "
    "CASE WHEN s.key IN $uids THEN s.key ELSE o.key END AS seed_entity_uid, "
    "s.key AS subject_entity_uid, o.key AS object_entity_uid, "
    "s.type AS subject_type, o.type AS object_type, a.predicate AS predicate, "
    "s.name AS subject_name, o.name AS object_name, "
    "coalesce(a.condition_text, '') AS condition_text, properties(e) AS evidence "
    "ORDER BY assertion_id LIMIT $row_limit"
)

_MENTIONS_QUERY = (
    "MATCH (n:Entity)-[m:MENTIONED_IN]->(e:Evidence) "
    "WHERE (n.key IN $uids OR n.normalized_name IN $names "
    "OR any(candidate_name IN coalesce(m.normalized_names, []) WHERE candidate_name IN $names)) "
    "AND n.namespace=e.namespace AND n.tenant_id=e.tenant_id AND n.kb_id=e.kb_id AND "
    + _projection_predicate("e", revision=True)
    + " RETURN DISTINCT n.key AS entity_uid, n.type AS entity_type, n.name AS name, "
    "coalesce(m.context_names, []) AS context_names, properties(e) AS evidence "
    "ORDER BY entity_uid, evidence.key LIMIT $row_limit"
)


class Neo4jGraphReader:
    """固定只读模板、服务端超时与客户端超时，逐次关闭驱动及会话。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._operation_driver = ContextVar("graph_operation_driver", default=None)

    def _driver(self):
        timeout = self.settings.query_graph_timeout_seconds
        return AsyncGraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password.get_secret_value()),
            connection_timeout=timeout,
            connection_acquisition_timeout=timeout,
            max_transaction_retry_time=0,
            max_connection_pool_size=4,
        )

    @asynccontextmanager
    async def operation(self):
        """驱动仅在当前预检查/召回内复用，任务上下文隔离并在结束时关闭。"""
        driver = self._driver()
        token = self._operation_driver.set(driver)
        try:
            yield
        finally:
            self._operation_driver.reset(token)
            await driver.close()

    async def _read(self, statement, parameters):
        timeout = self.settings.query_graph_timeout_seconds
        driver = self._operation_driver.get()
        own_driver = driver is None
        if own_driver:
            driver = self._driver()
        try:
            async with asyncio.timeout(timeout):
                async with driver.session(
                    database=self.settings.neo4j_database, default_access_mode="READ"
                ) as session:
                    result = await session.run(Query(statement, timeout=timeout), parameters)
                    return tuple([dict(row) async for row in result])
        finally:
            if own_driver:
                await driver.close()

    async def seeds(self, projections, *, names=(), uids=(), limit):
        return await self._read(
            _SEEDS_QUERY,
            {
                "projections": [item.parameters() for item in projections],
                "names": list(names),
                "uids": list(uids),
                "row_limit": limit,
            },
        )

    async def relations(self, projections, uids, limit):
        return await self._read(
            _RELATIONS_QUERY,
            {
                "projections": [item.parameters() for item in projections],
                "uids": list(uids),
                "predicates": list(ENDPOINTS),
                "row_limit": limit,
            },
        )

    async def mentions(self, projections, *, uids=(), names=(), limit):
        return await self._read(
            _MENTIONS_QUERY,
            {
                "projections": [item.parameters() for item in projections],
                "uids": list(uids),
                "names": list(names),
                "row_limit": limit,
            },
        )


def _evidence_key(evidence):
    return tuple(
        evidence[name]
        for name in (
            "tenant_id",
            "kb_id",
            "document_id",
            "revision_id",
            "generation_id",
            "chunk_id",
        )
    )


def _quote_matches(chunk, evidence):
    return (
        chunk is not None
        and chunk.content_sha256 == evidence.get("content_sha256")
        and hashlib.sha256(chunk.content.encode()).hexdigest() == chunk.content_sha256
        and chunk.content[evidence["quote_start"] : evidence["quote_end"]] == evidence["quote"]
    )


def _source(evidence, projections):
    projection = next(
        item
        for item in projections
        if all(evidence.get(key) == value for key, value in item.parameters().items())
    )
    return GraphSourceEvidence(
        tenant_id=str(evidence["tenant_id"]),
        knowledge_base_id=str(evidence["kb_id"]),
        document_id=str(evidence["document_id"]),
        revision_id=str(evidence["revision_id"]),
        generation_id=str(evidence["generation_id"]),
        chunk_id=str(evidence["chunk_id"]),
        projection_id=str(projection.projection_id),
        namespace=projection.namespace,
        content_sha256=evidence["content_sha256"],
        source_quote=evidence["quote"],
        quote_start=evidence["quote_start"],
        quote_end=evidence["quote_end"],
    )


def _relation_model(row):
    evidence = row["evidence"]
    return GraphRelationEvidence(
        **{
            name: row[name]
            for name in (
                "assertion_id",
                "seed_entity_uid",
                "subject_entity_uid",
                "object_entity_uid",
                "predicate",
                "condition_text",
            )
        },
        source_quote=evidence["quote"],
        quote_start=evidence["quote_start"],
        quote_end=evidence["quote_end"],
    )


def _strict_path_relation(row, chunk):
    """旧投影也重新核实两端、否定与条件，以及首批路径关系的明确触发。"""
    try:
        subject, object_ = row.get("subject_name"), row.get("object_name")
        if not subject or not object_:
            return False
        relation = ExtractedRelation(
            type=row["predicate"],
            source_id="s",
            target_id="o",
            evidence_quote=row["evidence"]["quote"],
            condition_text=row["condition_text"] or None,
        )
        validate_relation_evidence(
            relation,
            {
                "s": SimpleNamespace(name=subject, type=row["subject_type"]),
                "o": SimpleNamespace(name=object_, type=row["object_type"]),
            },
            chunk.content,
        )
        quote = relation.evidence_quote
        start, end = quote.find(subject), quote.find(object_)
        if start < 0 or end < 0 or start >= end:
            return False
        between = quote[start + len(subject) : end]
        # 判定关系动作本身的否定；句首的‘若尚未确认…’属于条件，不在此范围。
        if re.search(
            r"(?:不得|不能|不要|不应|不可|禁止|无需|无须|不必|不需要|不用|尚未|并未|未曾|不曾|没有|不再|未|不)"
            r"[ \t]*(?:(?:直接|另行|重复|擅自|自行|再次)[ \t]*){0,3}"
            r"(?:(?:提交|提供|填写|填入|填报|使用|需要|需|包含|包括|含有|负责|执行|办理|审核|复核|审查|确认|核实)|$)",
            between,
        ):
            return False
        # 完整保留原句中位于关系主体前的连续条件，不能用‘预算’等子串替代。
        absolute_subject = row["evidence"]["quote_start"] + start
        prefix = chunk.content[:absolute_subject]
        clause_start = max(
            (match.end() for match in re.finditer(r"[。！？!?；;\n]", prefix)), default=0
        )
        preceding = prefix[clause_start:].strip()
        pieces = list(re.finditer(r"[^，,]+", preceding))
        conditions = [
            part
            for part in pieces
            if re.search(
                r"^(?:如果|若(?!干)|如(?:有|改变|无法|不能|发现)|只有|只要|仅在|除非|必要时|尚缺少)|(?:时|后)$",
                part.group().strip(),
            )
        ]
        if conditions:
            expected = preceding[conditions[0].start() : conditions[-1].end()].strip()
            if row["condition_text"] != expected:
                return False
        triggers = {
            "REQUIRES_MATERIAL": r"需要|需|提交|提供|填写|使用|填报|必须",
            "HAS_FIELD": r"包含|包括|含有|字段|填写|填入|表头|栏[目位]|[|\t]",
            "HAS_STEP": r"包含|包括|步骤|进行|办理|先|首先|然后|：|:",
            "RESPONSIBLE_FOR": r"负责|执行|办理",
            "REVIEWS": r"审核|复核|审查",
            "CONFIRMS": r"确认|核实",
        }
        if re.search(triggers[row["predicate"]], between):
            return True
        return (
            row["predicate"] in {"REVIEWS", "CONFIRMS"}
            and re.match(triggers[row["predicate"]], object_) is not None
        )
    except (KeyError, TypeError, ValueError):
        return False


def _valid_relation(row, projections, uids) -> bool:
    """拒绝越界、未知类型及缺失条件的图记录，不把脏投影转换成新事实。"""
    try:
        evidence = row["evidence"]
        if not isinstance(evidence, dict):
            return False
        if not any(
            all(evidence.get(key) == value for key, value in item.parameters().items())
            for item in projections
        ):
            return False
        if not all(type(value) is int and value > 0 for value in _evidence_key(evidence)):
            return False
        if any(
            not isinstance(row[key], str) or not _UID.fullmatch(row[key])
            for key in (
                "assertion_id",
                "seed_entity_uid",
                "subject_entity_uid",
                "object_entity_uid",
            )
        ):
            return False
        if row["seed_entity_uid"] not in uids or row["seed_entity_uid"] not in (
            row["subject_entity_uid"],
            row["object_entity_uid"],
        ):
            return False
        subject, object_ = ENDPOINTS[row["predicate"]]
        if row["subject_type"] not in subject or row["object_type"] not in object_:
            return False
        quote, condition = evidence["quote"], row["condition_text"]
        if not isinstance(quote, str) or not quote or len(quote) > 500:
            return False
        if not isinstance(condition, str) or len(condition) > 500:
            return False
        if condition and condition not in quote:
            return False
        return (
            type(evidence["quote_start"]) is int
            and type(evidence["quote_end"]) is int
            and evidence["quote_start"] >= 0
            and evidence["quote_end"] == evidence["quote_start"] + len(quote)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _inline_mention(row, endpoint, chunk, projections, context_names):
    name = row[endpoint + "_name"]
    aliases = explicit_aliases(name, chunk.content)
    source = _source(row["evidence"], projections).model_copy(
        update={
            "source_quote": chunk.content,
            "quote_start": 0,
            "quote_end": len(chunk.content),
        }
    )
    return VerifiedMention(
        row[endpoint + "_entity_uid"],
        row[endpoint + "_type"],
        name,
        frozenset(normalize_name(value) for value in (name, *aliases)),
        frozenset(
            explicit_contexts(context_names, normalize_name(name), normalize_name(chunk.content))
        ),
        source,
        chunk,
    )


class OnlineGraphService:
    def __init__(self, settings: Settings, *, repository=None, reader=None):
        self.settings = settings
        self.repository = repository or OnlineGraphRepository(settings)
        self.reader = reader or Neo4jGraphReader(settings)

    @property
    def enabled(self):
        return self.settings.neo4j_enabled and self.settings.query_graph_enabled

    @asynccontextmanager
    async def _operation(self):
        if hasattr(self.reader, "operation"):
            async with self.reader.operation():
                yield
        else:
            yield

    async def _mentions(self, identity, scope, projections, *, names=(), uids=()):
        if not hasattr(self.reader, "mentions"):
            return ()
        budget = self.settings.query_graph_mention_limit
        rows = await self.reader.mentions(projections, names=names, uids=uids, limit=budget)
        if len(rows) >= budget:
            return ()
        valid = []
        for row in rows:
            evidence = row.get("evidence", {})
            if not isinstance(evidence, dict) or not any(
                all(evidence.get(key) == value for key, value in item.parameters().items())
                for item in projections
            ):
                continue
            if not all(
                type(evidence.get(key)) is int and evidence[key] > 0
                for key in (
                    "tenant_id",
                    "kb_id",
                    "document_id",
                    "revision_id",
                    "generation_id",
                    "chunk_id",
                )
            ):
                continue
            if not isinstance(row.get("entity_uid"), str) or not _UID.fullmatch(row["entity_uid"]):
                continue
            if row.get("entity_type") not in get_args(EntityType):
                continue
            quote = evidence.get("quote")
            name = row.get("name")
            if (
                not isinstance(quote, str)
                or not quote
                or len(quote) > 500
                or not isinstance(name, str)
            ):
                continue
            if normalize_name(name) not in normalize_name(quote):
                continue
            if (
                type(evidence.get("quote_start")) is not int
                or type(evidence.get("quote_end")) is not int
            ):
                continue
            if evidence["quote_start"] < 0 or evidence["quote_end"] != evidence[
                "quote_start"
            ] + len(quote):
                continue
            valid.append(row)
        if not valid:
            return ()
        chunks = await self.repository.chunks(
            identity, scope, projections, tuple(row["evidence"] for row in valid)
        )
        records = {_chunk_key(chunk): chunk for chunk in chunks}
        result = []
        for row in valid:
            evidence = row["evidence"]
            chunk = records.get(_evidence_key(evidence))
            if not _quote_matches(chunk, evidence):
                continue
            source = _source(evidence, projections).model_copy(
                update={
                    "source_quote": chunk.content,
                    "quote_start": 0,
                    "quote_end": len(chunk.content),
                }
            )
            aliases = explicit_aliases(row["name"], chunk.content)
            contexts = explicit_contexts(
                row.get("context_names", ()),
                normalize_name(row["name"]),
                normalize_name(chunk.content),
            )
            result.append(
                VerifiedMention(
                    row["entity_uid"],
                    row["entity_type"],
                    row["name"],
                    frozenset(normalize_name(name) for name in (row["name"], *aliases)),
                    frozenset(contexts),
                    source,
                    chunk,
                )
            )
        return tuple(result)

    async def revalidate_relations(
        self,
        identity: AuthIdentity,
        scope: AccessScope,
        params: GraphParams,
        chunks: tuple[ChunkRecord, ...],
    ) -> frozenset[int]:
        """输出前复查图出处是否仍是当前 READY 投影；原文复核由检索主链完成。"""
        _check_scope(identity, scope)
        if not self.enabled:
            return frozenset()
        requested = {item.model_dump_json() for item in params.projections}
        current = await self.repository.projections(identity, scope)
        allowed = {
            (
                item.tenant_id,
                item.knowledge_base_id,
                item.document_id,
                item.revision_id,
                item.generation_id,
            )
            for item in current
            if item.ref().model_dump_json() in requested
        }
        return frozenset(
            chunk.chunk_id
            for chunk in chunks
            if (
                chunk.tenant_id,
                chunk.knowledge_base_id,
                chunk.document_id,
                chunk.revision_id,
                chunk.generation_id,
            )
            in allowed
        )

    async def readiness(
        self,
        identity: AuthIdentity,
        scope: AccessScope,
        query: str,
        entities: tuple[str, ...],
        retrieval_params: RetrievalParams,
    ) -> GraphReadiness | None:
        async with self._operation():
            return await self._readiness(identity, scope, query, entities, retrieval_params)

    async def _readiness(self, identity, scope, query, entities, retrieval_params):
        _check_scope(identity, scope)
        if not self.enabled or not entities:
            return None
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("GRAPH_QUERY_INVALID")
        names = tuple(dict.fromkeys(normalize_name(item) for item in entities))
        if not all(names) or len(names) > self.settings.query_graph_seed_limit:
            return None
        projections = await self.repository.projections(identity, scope)
        if not projections:
            return None
        budget = self.settings.query_graph_seed_limit * 2 + 1
        rows = await self.reader.seeds(projections, names=names, limit=budget)
        if len(rows) >= budget:
            return None
        matches = {name: set() for name in names}
        for row in rows:
            name, uid = row.get("normalized_name"), row.get("entity_uid")
            if (
                name in matches
                and isinstance(uid, str)
                and _UID.fullmatch(uid)
                and row.get("entity_type") in get_args(EntityType)
            ):
                matches[name].add(uid)
        # 同名跨知识库、类型或流程产生多个节点时不猜测或强行合并。
        if any(len(values) != 1 for values in matches.values()):
            mentions = await self._mentions(identity, scope, projections, names=names)
            matches = {
                name: (
                    values
                    if len(values) == 1
                    else {mention.uid for mention in mentions if name in mention.names}
                )
                for name, values in matches.items()
            }
            for name, values in matches.items():
                if not values:
                    return None
                grouped = [mention for mention in mentions if mention.uid in values]
                if len(values) == 1:
                    continue
                anchor = grouped[0]
                if any(
                    not any(
                        identity_bridge(anchor, candidate) is not None
                        for candidate in grouped
                        if candidate.uid == uid
                    )
                    for uid in values
                    if uid != anchor.uid
                ):
                    return None
        uids = tuple(dict.fromkeys(uid for name in names for uid in sorted(matches[name])))
        if len(uids) > self.settings.query_graph_seed_limit:
            return None
        params = GraphParams(
            **retrieval_params.model_dump(),
            seed_entity_uids=uids,
            projections=tuple(item.ref() for item in projections),
            max_hops=self.settings.query_graph_max_hops,
            matched_entity_names=names,
        )
        # 本次预检查已读过投影与种子；只复用本调用局部值，原文回读仍刷新授权。
        result = await self._recall_from_visible_seeds(
            identity, scope, params, retrieval_params.candidate_k, projections, uids
        )
        if not result.hits:
            return None
        if result.path_count == 0:
            params = params.model_copy(update={"max_hops": 1})
        return GraphReadiness(query=query, matched_entities=entities, params=params)

    async def recall(
        self,
        identity: AuthIdentity,
        scope: AccessScope,
        params: GraphParams,
        limit: int,
    ) -> GraphRecallResult:
        async with self._operation():
            return await self._recall(identity, scope, params, limit)

    async def _recall(self, identity, scope, params, limit):
        _check_scope(identity, scope)
        _check_limit(limit)
        if not self.enabled:
            raise ValueError("GRAPH_DISABLED")
        if params.max_hops > self.settings.query_graph_max_hops:
            raise ValueError("GRAPH_HOPS_UNSUPPORTED")
        uids = params.seed_entity_uids
        if len(set(uids)) != len(uids) or len(uids) > self.settings.query_graph_seed_limit:
            raise ValueError("GRAPH_SEED_BUDGET_INVALID")
        if any(not _UID.fullmatch(uid) for uid in uids):
            raise ValueError("GRAPH_SEED_INVALID")
        requested = {item.model_dump_json() for item in params.projections}
        current = await self.repository.projections(identity, scope)
        projections = tuple(item for item in current if item.ref().model_dump_json() in requested)
        if not projections:
            return GraphRecallResult(())
        seed_rows = await self.reader.seeds(
            projections, uids=uids, limit=self.settings.query_graph_seed_limit + 1
        )
        visible = {
            row.get("entity_uid")
            for row in seed_rows
            if row.get("entity_type") in get_args(EntityType)
        }
        if visible != set(uids):
            return GraphRecallResult(())
        canonical_names = {row.get("normalized_name") for row in seed_rows}
        if params.matched_entity_names and not set(params.matched_entity_names) <= canonical_names:
            mentions = await self._mentions(
                identity, scope, projections, uids=uids, names=params.matched_entity_names
            )
            if any(
                not any(name in mention.names and mention.uid in uids for mention in mentions)
                for name in set(params.matched_entity_names) - canonical_names
            ):
                return GraphRecallResult(())
        return await self._recall_from_visible_seeds(
            identity, scope, params, limit, projections, uids
        )

    async def _recall_from_visible_seeds(self, identity, scope, params, limit, projections, uids):
        """共享关系与原文校验；正式 recall 在调用前重新读取授权投影和种子。"""
        if params.max_hops not in (1, 2):
            raise ValueError("GRAPH_HOPS_UNSUPPORTED")
        rows = await self.reader.relations(
            projections, uids, self.settings.query_graph_relation_limit
        )
        if len(rows) > self.settings.query_graph_relation_limit:
            raise ValueError("GRAPH_RELATION_BUDGET_EXCEEDED")
        rows = tuple(row for row in rows if _valid_relation(row, projections, uids))
        if not rows:
            return GraphRecallResult((), seed_count=len(uids))
        chunks = await self.repository.chunks(
            identity, scope, projections, tuple(row["evidence"] for row in rows)
        )
        records = {_chunk_key(chunk): chunk for chunk in chunks}
        relations = {}
        assertions = set()
        for row in rows:
            evidence = row["evidence"]
            key = _evidence_key(evidence)
            chunk = records.get(key)
            if not _quote_matches(chunk, evidence):
                continue
            relation = _relation_model(row)
            relations.setdefault(key, {})[row["assertion_id"]] = relation
            assertions.add(row["assertion_id"])
        ranked = sorted(relations, key=lambda key: (-len(relations[key]), key))[:limit]
        hits = tuple(
            RecallHit(records[key], float(len(relations[key])), tuple(relations[key].values()))
            for key in ranked
        )
        result = GraphRecallResult(hits, seed_count=len(uids), assertion_count=len(assertions))
        if params.max_hops == 2 and hasattr(self.reader, "mentions"):
            supported = tuple(row for row in rows if row["assertion_id"] in assertions)
            return await self._paths(identity, scope, params, projections, supported, result, limit)
        return result

    async def _paths(self, identity, scope, params, projections, first_rows, base, limit):
        first_rows = tuple(
            row
            for row in first_rows
            if row["predicate"] in {"REQUIRES_MATERIAL", "HAS_STEP"}
            and row["subject_entity_uid"] in params.seed_entity_uids
        )
        if not first_rows:
            return base
        requested = {item.model_dump_json() for item in params.projections}
        fresh = await self.repository.projections(identity, scope)
        projections = tuple(item for item in fresh if item.ref().model_dump_json() in requested)
        if not projections:
            return base
        mid_uids = tuple(dict.fromkeys(row["object_entity_uid"] for row in first_rows))
        names = tuple(
            dict.fromkeys(
                normalize_name(row["object_name"]) for row in first_rows if row.get("object_name")
            )
        )
        mentions = await self._mentions(identity, scope, projections, names=names, uids=mid_uids)
        expanded = set(mid_uids)
        for left in mentions:
            if left.uid not in mid_uids:
                continue
            expanded.update(right.uid for right in mentions if identity_bridge(left, right))
        expanded = tuple(sorted(expanded))
        if len(expanded) > self.settings.query_graph_mention_limit:
            return base
        second_rows = await self.reader.relations(
            projections, expanded, self.settings.query_graph_relation_limit
        )
        if len(second_rows) > self.settings.query_graph_relation_limit:
            raise ValueError("GRAPH_RELATION_BUDGET_EXCEEDED")
        second_rows = tuple(
            row for row in second_rows if _valid_relation(row, projections, expanded)
        )
        # 任一 hop 或桥接来源在扩展期间失效，完整路径都必须消失。
        all_evidence = tuple(row["evidence"] for row in (*first_rows, *second_rows)) + tuple(
            {
                "tenant_id": int(item.source.tenant_id),
                "kb_id": int(item.source.knowledge_base_id),
                "document_id": int(item.source.document_id),
                "revision_id": int(item.source.revision_id),
                "generation_id": int(item.source.generation_id),
                "chunk_id": int(item.source.chunk_id),
            }
            for item in mentions
        )
        chunks = await self.repository.chunks(identity, scope, projections, all_evidence)
        records = {_chunk_key(chunk): chunk for chunk in chunks}

        def source_valid(source):
            key = tuple(
                int(getattr(source, name))
                for name in (
                    "tenant_id",
                    "knowledge_base_id",
                    "document_id",
                    "revision_id",
                    "generation_id",
                    "chunk_id",
                )
            )
            chunk = records.get(key)
            return _quote_matches(
                chunk,
                {
                    "content_sha256": source.content_sha256,
                    "quote": source.source_quote,
                    "quote_start": source.quote_start,
                    "quote_end": source.quote_end,
                },
            )

        mentions = tuple(item for item in mentions if source_valid(item.source))
        paths, path_rows, required = [], {}, set()
        for first in first_rows:
            first_evidence = first["evidence"]
            first_chunk = records.get(_evidence_key(first_evidence))
            if not _quote_matches(first_chunk, first_evidence) or not _strict_path_relation(
                first, first_chunk
            ):
                continue
            for second in second_rows:
                outgoing = first["predicate"] == "REQUIRES_MATERIAL"
                if outgoing and second["predicate"] != "HAS_FIELD":
                    continue
                if not outgoing and second["predicate"] not in {
                    "RESPONSIBLE_FOR",
                    "REVIEWS",
                    "CONFIRMS",
                }:
                    continue
                right_uid = (
                    second["subject_entity_uid"] if outgoing else second["object_entity_uid"]
                )
                second_evidence = second["evidence"]
                if first["assertion_id"] == second["assertion_id"]:
                    continue
                second_chunk = records.get(_evidence_key(second_evidence))
                if not _quote_matches(second_chunk, second_evidence) or not _strict_path_relation(
                    second, second_chunk
                ):
                    continue
                if first_evidence["kb_id"] != second_evidence["kb_id"]:
                    continue
                same_document = all(
                    first_evidence[key] == second_evidence[key]
                    for key in ("namespace", "document_id", "generation_id")
                )
                bridges = ()
                if (
                    first["object_entity_uid"] == right_uid
                    and same_document
                    and first["object_type"] in {"Material", "Process", "Actor"}
                ):
                    context_names = tuple(
                        context for mention in mentions for context in mention.contexts
                    )
                    left = _inline_mention(first, "object", first_chunk, projections, context_names)
                    right = _inline_mention(
                        second,
                        "subject" if outgoing else "object",
                        second_chunk,
                        projections,
                        context_names,
                    )
                    if identity_conflict(left, right):
                        continue
                if not (first["object_entity_uid"] == right_uid and same_document):
                    options = [
                        identity_bridge(left, right)
                        for left in mentions
                        for right in mentions
                        if left.uid == first["object_entity_uid"]
                        and right.uid == right_uid
                        and int(left.source.document_id) == first_evidence["document_id"]
                        and int(right.source.document_id) == second_evidence["document_id"]
                    ]
                    bridge = next((item for item in options if item is not None), None)
                    if bridge is None:
                        continue
                    bridges = (bridge,)
                path_first = {**first, "seed_entity_uid": first["subject_entity_uid"]}
                path_second = {**second, "seed_entity_uid": right_uid}
                hops = tuple(
                    GraphHopEvidence(
                        relation=_relation_model(row),
                        source=_source(row["evidence"], projections),
                        traversal=direction,
                        subject_name=row.get("subject_name", row["subject_entity_uid"]),
                        object_name=row.get("object_name", row["object_entity_uid"]),
                    )
                    for row, direction in (
                        (path_first, "outgoing"),
                        (path_second, "outgoing" if outgoing else "incoming"),
                    )
                )
                sources = tuple(hop.source for hop in hops) + tuple(
                    source for bridge in bridges for source in bridge.sources
                )
                ids = tuple(dict.fromkeys(source.chunk_id for source in sources))
                if len(required | set(ids)) > limit:
                    continue
                path = GraphPathEvidence(
                    path_id=stable_key(
                        first["assertion_id"],
                        second["assertion_id"],
                        [bridge.model_dump() for bridge in bridges],
                    ),
                    knowledge_base_id=str(first_evidence["kb_id"]),
                    hops=hops,
                    bridges=bridges,
                    required_chunk_ids=ids,
                )
                if any(item.path_id == path.path_id for item in paths):
                    continue
                paths.append(path)
                required.update(ids)
                for row in (path_first, path_second):
                    path_rows.setdefault(row["evidence"]["chunk_id"], {})[row["assertion_id"]] = (
                        _relation_model(row)
                    )
                if len(paths) >= self.settings.query_graph_path_limit:
                    break
            if len(paths) >= self.settings.query_graph_path_limit:
                break
        if not paths:
            return base
        records_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        previous = {hit.chunk.chunk_id: hit for hit in base.hits}
        ordered = list(
            dict.fromkeys(int(source) for path in paths for source in path.required_chunk_ids)
        )
        ordered.extend(chunk_id for chunk_id in previous if chunk_id not in ordered)
        hits = []
        for chunk_id in ordered[:limit]:
            old = previous.get(chunk_id)
            relations = {item.assertion_id: item for item in old.graph_relations} if old else {}
            relations.update(path_rows.get(chunk_id, {}))
            own_paths = tuple(path for path in paths if str(chunk_id) in path.required_chunk_ids)
            hits.append(
                RecallHit(
                    records_by_id.get(chunk_id, old.chunk if old else None),
                    float(len(relations) + len(own_paths)),
                    tuple(relations.values()),
                    own_paths,
                )
            )
        assertion_count = len({item.assertion_id for hit in hits for item in hit.graph_relations})
        return GraphRecallResult(tuple(hits), base.seed_count, assertion_count, len(paths))
