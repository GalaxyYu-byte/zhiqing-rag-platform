"""Read-only capture of PostgreSQL graph state and Neo4j projection for frozen cases."""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from neo4j import AsyncGraphDatabase

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database.connection import database_connection

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "eval" / "graph_build" / "v1"
GOLD = ROOT / "eval" / "graph_build" / "gold-v1"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def capture_postgres(settings: Settings, documents: list[dict]) -> dict:
    schema = settings.db_schema
    ids = [item["document_id"] for item in documents]
    with database_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute(
                f"""SELECT d.id AS document_id, d.tenant_id, d.kb_id, d.status,
                           d.desired_revision_id, d.active_generation_id,
                           g.revision_id AS generation_revision_id, g.status AS generation_status,
                           g.chunk_count AS generation_chunk_count,
                           p.id AS projection_id, p.projection_version,
                           p.status AS projection_status,
                           p.graph_namespace, p.extractor_version, p.model_name,
                           p.entity_count, p.relation_count
                    FROM {schema}.document d
                    LEFT JOIN {schema}.index_generation g ON g.id=d.active_generation_id
                    LEFT JOIN {schema}.graph_projection p
                      ON p.tenant_id=d.tenant_id AND p.generation_id=d.active_generation_id
                     AND p.projection_version=1
                    WHERE d.id = ANY(%s) ORDER BY d.id""",
                (ids,),
            )
            state = cursor.fetchall()
            cursor.execute(
                f"""SELECT id, tenant_id, kb_id, document_id, revision_id, generation_id,
                           chunk_index, content_sha256,
                           metadata -> '_graph_extraction_cache' AS extraction_cache
                    FROM {schema}.document_chunk WHERE document_id = ANY(%s)
                    ORDER BY document_id, generation_id, chunk_index""",
                (ids,),
            )
            chunks = cursor.fetchall()
    return {"documents": state, "chunks": chunks}


async def records(session, query: str, **parameters) -> list[dict]:
    result = await session.run(query, **parameters)
    return [dict(row) async for row in result]


async def capture_neo4j(settings: Settings, documents: list[dict]) -> dict:
    generations = sorted({item["generation_id"] for item in documents})
    document_ids = sorted({item["document_id"] for item in documents})
    tenants = sorted({item["tenant_id"] for item in documents})
    params = {
        "ns": settings.graph_namespace,
        "gens": generations,
        "docs": document_ids,
        "tenants": tenants,
    }
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
        connection_timeout=5,
    )
    try:
        async with driver.session(database=settings.neo4j_database) as session:
            evidence = await records(
                session,
                "MATCH (e:Evidence) WHERE e.namespace=$ns AND e.tenant_id IN $tenants "
                "AND e.generation_id IN $gens RETURN properties(e) AS value "
                "ORDER BY e.generation_id, e.key",
                **params,
            )
            mentions = await records(
                session,
                "MATCH (n:Entity)-[:MENTIONED_IN]->(e:Evidence) "
                "WHERE e.namespace=$ns AND e.tenant_id IN $tenants "
                "AND e.generation_id IN $gens "
                "RETURN properties(n) AS entity, e.key AS evidence_key "
                "ORDER BY e.generation_id, e.chunk_id, n.key",
                **params,
            )
            assertions = await records(
                session,
                "MATCH (a:Assertion) WHERE a.namespace=$ns AND a.tenant_id IN $tenants "
                "AND a.generation_id IN $gens "
                "OPTIONAL MATCH (a)-[:SUBJECT]->(s:Entity) "
                "OPTIONAL MATCH (a)-[:OBJECT]->(o:Entity) "
                "OPTIONAL MATCH (a)-[:SUPPORTED_BY]->(e:Evidence) "
                "RETURN properties(a) AS assertion, collect(DISTINCT properties(s)) AS subjects, "
                "collect(DISTINCT properties(o)) AS objects, "
                "collect(DISTINCT properties(e)) AS evidences",
                **params,
            )
            orphan_entities = await records(
                session,
                "MATCH (n:Entity) WHERE n.namespace=$ns AND n.tenant_id IN $tenants "
                "AND NOT EXISTS { MATCH (n)--() } RETURN n.key AS key ORDER BY n.key",
                **params,
            )
            duplicate_entity_keys = await records(
                session,
                "MATCH (n:Entity) WHERE n.namespace=$ns AND n.tenant_id IN $tenants "
                "WITH n.key AS key, count(*) AS n WHERE n > 1 RETURN key, n ORDER BY key",
                **params,
            )
            other_generation_evidence = await records(
                session,
                "MATCH (e:Evidence) WHERE e.namespace=$ns AND e.tenant_id IN $tenants "
                "AND e.document_id IN $docs AND NOT (e.generation_id IN $gens) "
                "RETURN e.key AS key, e.document_id AS document_id, "
                "e.generation_id AS generation_id ORDER BY e.document_id, e.generation_id, e.key",
                **params,
            )
    finally:
        await driver.close()
    return {
        "evidence": [item["value"] for item in evidence],
        "mentions": mentions,
        "assertions": assertions,
        "orphan_entity_keys": [item["key"] for item in orphan_entities],
        "duplicate_entity_keys": duplicate_entity_keys,
        "other_generation_evidence": other_generation_evidence,
    }


def capture(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"快照已存在: {output}")
    settings = Settings()
    documents = read_lines(FROZEN / "documents.jsonl")
    pg = capture_postgres(settings, documents)
    neo4j = asyncio.run(capture_neo4j(settings, documents))
    snapshot = {
        "schema_version": "graph-eval-capture-v1",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "frozen_manifest_sha256": hash_file(FROZEN / "manifest.json"),
        "gold_manifest_sha256": hash_file(GOLD / "manifest.json"),
        "graph_namespace": settings.graph_namespace,
        "neo4j_database": settings.neo4j_database,
        "postgres": pg,
        "neo4j": neo4j,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"只读快照：{len(pg['documents'])} 份文档、"
        f"{len(neo4j['assertions'])} 个 Assertion、{len(neo4j['evidence'])} 个 Evidence"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.output)


if __name__ == "__main__":
    main()
