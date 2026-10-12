import { queryFixture } from "./query-fixture.mjs";

export function graphPathFixture({ bridge = false, astral = false } = {}) {
  const result = queryFixture();
  const first = result.retrieval.evidence[0];
  first.content = `${astral ? "😀" : ""} 关闭前，处理步骤应形成关闭摘要。\n<script>原文示例</script> `;
  const second = { ...first, citation_id: "E2", rank: 2, chunk_id: "22", document_id: "4",
    revision_id: "6", generation_id: "7", content: "关闭摘要应包含负责人确认结论。", graph_relations: [] };
  if (!bridge) Object.assign(second, { document_id: first.document_id,
    revision_id: first.revision_id, generation_id: first.generation_id });
  const proof = { ...first, citation_id: "E3", rank: 3, chunk_id: "23", document_id: "5",
    revision_id: "8", generation_id: "9", content: "本流程中的关闭记录与关闭摘要为同一材料。", graph_relations: [] };
  const projection = (item, id) => ({ tenant_id: "10", document_id: item.document_id,
    generation_id: item.generation_id, projection_id: id, namespace: "private_projection" });
  const projections = [projection(first, "11"), projection(second, "12")];
  if (bridge) projections.push(projection(proof, "13"));
  const source = (item, projectionId) => ({ ...projection(item, projectionId), knowledge_base_id: item.knowledge_base_id,
    revision_id: item.revision_id, chunk_id: item.chunk_id, content_sha256: "a".repeat(64),
    source_quote: item.content, quote_start: 0, quote_end: [...item.content].length });
  const relation = (item, id, from, to, predicate, condition = "") => ({ assertion_id: id, seed_entity_uid: from,
    subject_entity_uid: from, object_entity_uid: to, predicate, condition_text: condition,
    source_quote: item.content, quote_start: 0, quote_end: [...item.content].length });
  const firstRelation = relation(first, "assertion-1", "step-1", "material-1", "PRODUCES_MATERIAL", "关闭前");
  const secondRelation = relation(second, "assertion-2", bridge ? "material-alias" : "material-1", "field-1", "HAS_FIELD");
  first.graph_relations = [firstRelation]; second.graph_relations = [secondRelation];
  result.retrieval.evidence = bridge ? [first, second, proof] : [first, second];
  result.plan.route = result.plan.requested_route = "hybrid_graph";
  Object.assign(result.plan.params, { seed_entity_uids: ["step-1"], max_hops: 2,
    matched_entity_names: ["处理步骤"], projections });
  Object.assign(result.retrieval.trace, { graph_status: "succeeded", graph_hops: 2, graph_count: result.retrieval.evidence.length,
    graph_seed_count: 1, graph_assertion_count: 2, graph_ms: 80, graph_path_count: 1,
    graph_path_candidate_count: 3, returned_count: result.retrieval.evidence.length });
  result.retrieval.graph_paths = [{ path_id: "path-1", knowledge_base_id: first.knowledge_base_id,
    hops: [
      { relation: firstRelation, source: source(first, "11"), traversal: "outgoing", subject_name: "处理步骤", object_name: "关闭摘要" },
      { relation: secondRelation, source: source(second, "12"), traversal: "outgoing", subject_name: "关闭摘要", object_name: "负责人确认结论" },
    ],
    bridges: bridge ? [{ source_entity_uid: "material-1", target_entity_uid: "material-alias", entity_type: "Material",
      canonical_name: "关闭摘要", basis: "explicit_alias", sources: [source(first, "11"), source(proof, "13")] }] : [],
    required_chunk_ids: result.retrieval.evidence.map((item) => item.chunk_id),
  }];
  return result;
}
