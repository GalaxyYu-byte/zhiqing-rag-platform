// Graph RAG v1 约束。全部语句均可重复执行。
CREATE CONSTRAINT graph_kb_id_unique IF NOT EXISTS
FOR (n:KnowledgeBase)
REQUIRE n.kb_id IS UNIQUE;

CREATE CONSTRAINT graph_document_id_unique IF NOT EXISTS
FOR (n:Document)
REQUIRE n.doc_id IS UNIQUE;

CREATE CONSTRAINT graph_document_version_unique IF NOT EXISTS
FOR (n:DocumentVersion)
REQUIRE (n.doc_id, n.version) IS UNIQUE;

CREATE CONSTRAINT graph_chunk_uid_unique IF NOT EXISTS
FOR (n:Chunk)
REQUIRE n.uid IS UNIQUE;

CREATE CONSTRAINT graph_entity_uid_unique IF NOT EXISTS
FOR (n:Entity)
REQUIRE n.uid IS UNIQUE;

CREATE CONSTRAINT graph_claim_uid_unique IF NOT EXISTS
FOR (n:Claim)
REQUIRE n.uid IS UNIQUE;

CREATE CONSTRAINT graph_mention_uid_unique IF NOT EXISTS
FOR ()-[r:MENTIONS]-()
REQUIRE r.uid IS UNIQUE;

CREATE CONSTRAINT graph_evidence_uid_unique IF NOT EXISTS
FOR ()-[r:SUPPORTED_BY]-()
REQUIRE r.uid IS UNIQUE;

CREATE INDEX graph_entity_lookup IF NOT EXISTS
FOR (n:Entity)
ON (n.kb_id, n.entity_type, n.identity_key);

CREATE INDEX graph_claim_predicate IF NOT EXISTS
FOR (n:Claim)
ON (n.kb_id, n.predicate);
