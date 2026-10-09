export function queryFixture(overrides = {}) {
  const value = {
    analysis: { status: "ok", original_query: "工单关闭前需要谁确认？", analysis: { intent: "procedure" }, message: null, degradation_reason: null, latency_ms: 6057 },
    plan: { requested_route: "hybrid", route: "hybrid", reason_code: "BASIC_RETRIEVAL", planning_status: "planned", warnings: [], message: null, params: { candidate_k: 30, top_k: 5, rerank: true } },
    scope: { knowledge_base_ids: ["9007199254740993"], authorized_document_count: 2, published_document_count: 1 },
    retrieval: {
      status: "ok", message: null, degradation_codes: [],
      evidence: [{ citation_id: "E1", rank: 1, chunk_id: "21", knowledge_base_id: "9007199254740993", document_id: "3", revision_id: "4", generation_id: "5", document_title: "工单管理办法", document_name: "工单流程.pdf", chunk_index: 0, content: "  工单关闭前需由负责人确认。\n<script>原文示例</script>  ", token_count: 30, page_number: 2, section_title: "关闭流程", scores: { dense_rank: 1, dense_score: 0.9, bm25_rank: 1, bm25_score: 2, fusion_score: 0.03, rerank_score: 0.8 } }],
      trace: { dense_status: "succeeded", bm25_status: "succeeded", rerank_status: "succeeded", dense_count: 2, bm25_count: 2, fused_count: 2, authorized_count: 2, returned_count: 1, embedding_ms: 100, dense_ms: 20, bm25_ms: 30, fusion_ms: 1, authorization_ms: 10, rerank_ms: 150, total_ms: 300 },
    },
  };
  for (const [key, fields] of Object.entries(overrides)) Object.assign(value[key], fields);
  return value;
}
