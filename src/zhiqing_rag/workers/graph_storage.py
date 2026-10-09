"""Neo4j 可重放投影：实体、断言和逐条原文证据均使用稳定键。"""

import json

from neo4j import AsyncGraphDatabase

from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers.graph_extraction import Extraction, normalize_name, stable_key


class GraphStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        )

    async def close(self):
        await self.driver.close()

    async def prepare(self):
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            for label in ("Entity", "Evidence", "Assertion", "GraphFence"):
                result = await session.run(
                    f"CREATE CONSTRAINT graph_{label.lower()}_key IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.key IS UNIQUE"
                )
                await result.consume()
            result = await session.run(
                "CREATE INDEX graph_evidence_chunk IF NOT EXISTS FOR (e:Evidence) "
                "ON (e.namespace, e.tenant_id, e.generation_id, e.chunk_id)"
            )
            await result.consume()

    def chunk_records(self, scope: dict, chunk: dict, extraction: Extraction):
        namespace = self.settings.graph_namespace
        entities = []
        local_keys = {}
        for item in extraction.entities:
            canonical = normalize_name(item.canonical_name)
            if not canonical:
                raise ValueError("规范实体名称为空")
            # 流程步骤、状态、字段和规则有上下文依赖，限制在文档内避免误合并。
            context = (
                (scope["document_id"], normalize_name(chunk["section_title"] or ""))
                if item.type in {"Step", "State", "Field", "Rule"}
                else None
            )
            key = stable_key(
                namespace, scope["tenant_id"], scope["kb_id"], item.type, canonical, context
            )
            local_keys[item.id] = key
            entities.append(
                {
                    "key": key,
                    "type": item.type,
                    "name": item.canonical_name,
                    "normalized_name": canonical,
                    "surface_name": item.name,
                    "evidence": self._evidence(scope, chunk, item.evidence_quote),
                }
            )
        relations = []
        for item in extraction.relations:
            evidence = self._evidence(scope, chunk, item.evidence_quote)
            source_key = local_keys[item.source_id]
            target_key = local_keys[item.target_id]
            relations.append(
                {
                    "key": stable_key(
                        evidence["key"], source_key, item.type, target_key, item.condition_text
                    ),
                    "source_key": source_key,
                    "target_key": target_key,
                    "type": item.type,
                    "condition_text": item.condition_text,
                    "evidence": evidence,
                }
            )
        return entities, relations

    async def write_chunk(self, scope: dict, chunk: dict, extraction: Extraction):
        entities, relations = self.chunk_records(scope, chunk, extraction)
        namespace = self.settings.graph_namespace
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            await session.execute_write(
                self._write_transaction, entities, relations, scope, namespace, chunk["id"]
            )
        return {entity["key"] for entity in entities}, {relation["key"] for relation in relations}

    async def snapshot(self, scope: dict) -> dict:
        """读取一个文档版本的实际节点与边，供校验和带证据抽查。"""
        parameters = {
            "namespace": self.settings.graph_namespace,
            "tenant_id": scope["tenant_id"],
            "document_id": scope["document_id"],
            "generation_id": scope["generation_id"],
        }
        evidence_scope = (
            "e.namespace=$namespace AND e.tenant_id=$tenant_id "
            "AND e.document_id=$document_id AND e.generation_id=$generation_id "
        )
        async with self.driver.session(
            database=self.settings.neo4j_database, default_access_mode="READ"
        ) as session:

            async def collect(query):
                result = await session.run(query, **parameters)
                return [row async for row in result]

            entities = await collect(
                "MATCH (n:Entity)-[:MENTIONED_IN]->(e:Evidence) WHERE "
                + evidence_scope
                + "RETURN DISTINCT properties(n) AS record ORDER BY record.key"
            )
            evidence = await collect(
                "MATCH (e:Evidence) WHERE "
                + evidence_scope
                + "RETURN properties(e) AS record ORDER BY record.key"
            )
            assertions = await collect(
                "MATCH (a:Assertion) WHERE a.namespace=$namespace AND a.tenant_id=$tenant_id "
                "AND a.document_id=$document_id AND a.generation_id=$generation_id "
                "OPTIONAL MATCH (a)-[:SUBJECT]->(s:Entity) "
                "OPTIONAL MATCH (a)-[:OBJECT]->(o:Entity) "
                "OPTIONAL MATCH (a)-[:SUPPORTED_BY]->(e:Evidence) "
                "RETURN properties(a) AS record, collect(DISTINCT s.key) AS subjects, "
                "collect(DISTINCT o.key) AS objects, collect(DISTINCT e.key) AS evidence "
                "ORDER BY record.key"
            )
        return {
            "entities": [row["record"] for row in entities],
            "evidence": [row["record"] for row in evidence],
            "assertions": [
                {
                    **row["record"],
                    "subjects": row["subjects"],
                    "objects": row["objects"],
                    "evidence": row["evidence"],
                }
                for row in assertions
            ],
        }

    async def delete_generation(
        self, tenant_id: int, document_id: int, generation_id: int, namespace: str
    ) -> None:
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            await session.execute_write(
                self._delete_transaction, namespace, tenant_id, document_id, generation_id
            )

    async def delete_document(
        self, tenant_id: int, document_id: int, namespaces: list[str]
    ) -> None:
        async with self.driver.session(database=self.settings.neo4j_database) as session:
            for namespace in sorted(set(namespaces)):
                await session.execute_write(
                    self._delete_transaction, namespace, tenant_id, document_id, None
                )

    def _evidence(self, scope: dict, chunk: dict, quote: str) -> dict:
        offset = chunk["content"].find(quote)
        return {
            "key": stable_key(
                self.settings.graph_namespace,
                scope["tenant_id"],
                scope["generation_id"],
                chunk["id"],
                offset,
                quote,
            ),
            "namespace": self.settings.graph_namespace,
            "tenant_id": scope["tenant_id"],
            "kb_id": scope["kb_id"],
            "document_id": scope["document_id"],
            "revision_id": scope["revision_id"],
            "generation_id": scope["generation_id"],
            "chunk_id": chunk["id"],
            "chunk_index": chunk["chunk_index"],
            "content_sha256": chunk["content_sha256"],
            "quote": quote,
            "quote_start": offset,
            "quote_end": offset + len(quote),
            "page_number": chunk["page_number"],
            "section_title": chunk["section_title"],
            "chunk_start_offset": chunk["start_offset"],
            "chunk_end_offset": chunk["end_offset"],
            "source_refs_json": json.dumps(
                chunk["metadata_"].get("source_refs", []), ensure_ascii=False
            ),
        }

    @staticmethod
    async def _write_transaction(
        tx, entities: list, relations: list, scope: dict, namespace: str, chunk_id: int
    ):
        fence_key = stable_key(namespace, scope["tenant_id"], scope["document_id"], "fence")
        result = await tx.run(
            "MERGE (f:GraphFence {key:$key}) "
            "ON CREATE SET f.namespace=$namespace, f.tenant_id=$tenant_id, "
            "f.document_id=$document_id, f.retired_generations=[], f.document_deleted=false "
            "SET f.touch=coalesce(f.touch,0)+1 "
            "RETURN f.retired_generations AS retired, f.document_deleted AS deleted",
            key=fence_key,
            namespace=namespace,
            tenant_id=scope["tenant_id"],
            document_id=scope["document_id"],
        )
        fence = await result.single()
        if fence["deleted"] or scope["generation_id"] in fence["retired"]:
            raise ValueError("GRAPH_GENERATION_RETIRED")
        # 重试同一分块时先清掉旧断言，避免 LLM 输出差异遗留旧事实。
        result = await tx.run(
            "MATCH (a:Assertion)-[:SUPPORTED_BY]->(e:Evidence) "
            "WHERE e.namespace=$namespace AND e.tenant_id=$tenant_id "
            "AND e.generation_id=$generation_id AND e.chunk_id=$chunk_id "
            "DETACH DELETE a",
            namespace=namespace,
            tenant_id=scope["tenant_id"],
            generation_id=scope["generation_id"],
            chunk_id=chunk_id,
        )
        await result.consume()
        result = await tx.run(
            "MATCH (e:Evidence) WHERE e.namespace=$namespace AND e.tenant_id=$tenant_id "
            "AND e.generation_id=$generation_id AND e.chunk_id=$chunk_id DETACH DELETE e",
            namespace=namespace,
            tenant_id=scope["tenant_id"],
            generation_id=scope["generation_id"],
            chunk_id=chunk_id,
        )
        await result.consume()
        for entity in entities:
            result = await tx.run(
                "MERGE (n:Entity {key:$key}) "
                "ON CREATE SET n.namespace=$namespace, n.tenant_id=$tenant_id, n.kb_id=$kb_id, "
                "n.type=$type, n.name=$name, n.normalized_name=$normalized_name "
                "MERGE (e:Evidence {key:$evidence_key}) "
                "ON CREATE SET e += $evidence "
                "MERGE (n)-[:MENTIONED_IN]->(e)",
                **{k: v for k, v in entity.items() if k != "evidence"},
                evidence=entity["evidence"],
                evidence_key=entity["evidence"]["key"],
                namespace=namespace,
                tenant_id=scope["tenant_id"],
                kb_id=scope["kb_id"],
            )
            await result.consume()
        for relation in relations:
            result = await tx.run(
                "MATCH (s:Entity {key:$source_key}), (o:Entity {key:$target_key}) "
                "MERGE (a:Assertion {key:$key}) "
                "ON CREATE SET a.namespace=$namespace, a.tenant_id=$tenant_id, a.kb_id=$kb_id, "
                "a.document_id=$document_id, a.generation_id=$generation_id, "
                "a.predicate=$type, a.condition_text=$condition_text "
                "MERGE (e:Evidence {key:$evidence_key}) "
                "ON CREATE SET e += $evidence "
                "MERGE (a)-[:SUBJECT]->(s) MERGE (a)-[:OBJECT]->(o) "
                "MERGE (a)-[:SUPPORTED_BY]->(e)",
                **relation,
                evidence_key=relation["evidence"]["key"],
                namespace=namespace,
                tenant_id=scope["tenant_id"],
                kb_id=scope["kb_id"],
                document_id=scope["document_id"],
                generation_id=scope["generation_id"],
            )
            await result.consume()

    @staticmethod
    async def _delete_transaction(
        tx, namespace: str, tenant_id: int, document_id: int, generation_id: int | None
    ):
        fence_key = stable_key(namespace, tenant_id, document_id, "fence")
        if generation_id is None:
            fence_update = "SET f.document_deleted=true, f.touch=coalesce(f.touch,0)+1"
        else:
            fence_update = (
                "SET f.retired_generations=CASE WHEN $generation_id IN f.retired_generations "
                "THEN f.retired_generations ELSE f.retired_generations + $generation_id END, "
                "f.touch=coalesce(f.touch,0)+1"
            )
        result = await tx.run(
            "MERGE (f:GraphFence {key:$key}) "
            "ON CREATE SET f.namespace=$namespace, f.tenant_id=$tenant_id, "
            "f.document_id=$document_id, f.retired_generations=[], f.document_deleted=false "
            + fence_update,
            key=fence_key,
            namespace=namespace,
            tenant_id=tenant_id,
            document_id=document_id,
            generation_id=generation_id,
        )
        await result.consume()
        scope = "AND a.generation_id=$generation_id" if generation_id is not None else ""
        result = await tx.run(
            "MATCH (a:Assertion) WHERE a.namespace=$namespace AND a.tenant_id=$tenant_id "
            "AND a.document_id=$document_id " + scope + " DETACH DELETE a",
            namespace=namespace,
            tenant_id=tenant_id,
            document_id=document_id,
            generation_id=generation_id,
        )
        await result.consume()
        evidence_scope = "AND e.generation_id=$generation_id" if generation_id is not None else ""
        result = await tx.run(
            "MATCH (e:Evidence) WHERE e.namespace=$namespace AND e.tenant_id=$tenant_id "
            "AND e.document_id=$document_id " + evidence_scope + " DETACH DELETE e",
            namespace=namespace,
            tenant_id=tenant_id,
            document_id=document_id,
            generation_id=generation_id,
        )
        await result.consume()
        result = await tx.run(
            "MATCH (n:Entity) WHERE n.namespace=$namespace AND n.tenant_id=$tenant_id "
            "AND NOT EXISTS { MATCH (n)--() } DELETE n",
            namespace=namespace,
            tenant_id=tenant_id,
        )
        await result.consume()
