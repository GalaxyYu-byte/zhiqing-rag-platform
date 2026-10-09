import { authenticatedFetch } from "./auth";
import { responseCode, responseMessage } from "./auth-client";

export type QueryRoute = "hybrid" | "hybrid_graph" | "structured" | "clarify" | "none";
export type RetrievalStatus = "ok" | "degraded" | "empty" | "skipped" | "unsupported";
export interface QueryEvidence {
  citation_id: string;
  rank: number;
  chunk_id: string;
  knowledge_base_id: string;
  document_id: string;
  revision_id: string;
  generation_id: string;
  document_title: string;
  document_name: string;
  chunk_index: number;
  content: string;
  token_count: number;
  page_number: number | null;
  section_title: string | null;
  scores: {
    dense_rank: number | null; dense_score: number | null;
    bm25_rank: number | null; bm25_score: number | null;
    fusion_score: number; rerank_score: number | null;
  };
}
export interface RetrievalTrace {
  dense_status: "succeeded" | "failed" | "skipped";
  bm25_status: "succeeded" | "failed" | "skipped";
  rerank_status: "succeeded" | "failed" | "disabled" | "skipped";
  dense_count: number; bm25_count: number; fused_count: number;
  authorized_count: number; returned_count: number;
  embedding_ms: number; dense_ms: number; bm25_ms: number;
  fusion_ms: number; authorization_ms: number; rerank_ms: number; total_ms: number;
}
export interface QuerySearchResponse {
  analysis: {
    status: "ok" | "degraded";
    original_query: string;
    analysis: { intent: string } | null;
    message: string | null;
    degradation_reason: string | null;
    latency_ms: number;
  };
  plan: {
    requested_route: QueryRoute;
    route: QueryRoute;
    reason_code: string;
    planning_status: "planned" | "unsupported" | "empty";
    warnings: string[];
    message: string | null;
    params: { question?: string; candidate_k?: number; top_k?: number; rerank?: boolean };
  };
  scope: {
    knowledge_base_ids: string[];
    authorized_document_count: number;
    published_document_count: number;
  };
  retrieval: {
    status: RetrievalStatus;
    evidence: QueryEvidence[];
    degradation_codes: string[];
    message: string | null;
    trace: RetrievalTrace;
  };
}

export class QuerySearchError extends Error {
  constructor(message: string, readonly status: number, readonly code: string) {
    super(message);
    this.name = "QuerySearchError";
  }
}

const routes = ["hybrid", "hybrid_graph", "structured", "clarify", "none"];
const statuses = ["ok", "degraded", "empty", "skipped", "unsupported"];
const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const isCount = (value: unknown) => typeof value === "number" && Number.isFinite(value) && value >= 0;
const isStrings = (value: unknown) => Array.isArray(value) && value.every((item) => typeof item === "string");

function validResponse(data: unknown): data is QuerySearchResponse {
  if (!isObject(data)) return false;
  const { analysis, plan, scope, retrieval } = data;
  if (!isObject(analysis) || !isObject(plan) || !isObject(scope) || !isObject(retrieval)) return false;
  const trace = retrieval.trace;
  return ["ok", "degraded"].includes(String(analysis.status)) && isCount(analysis.latency_ms)
    && routes.includes(String(plan.route)) && routes.includes(String(plan.requested_route))
    && isObject(plan.params) && isStrings(plan.warnings)
    && isStrings(scope.knowledge_base_ids) && isCount(scope.authorized_document_count)
    && isCount(scope.published_document_count) && statuses.includes(String(retrieval.status))
    && isStrings(retrieval.degradation_codes) && isObject(trace)
    && ["succeeded", "failed", "skipped"].includes(String(trace.dense_status))
    && ["succeeded", "failed", "skipped"].includes(String(trace.bm25_status))
    && ["succeeded", "failed", "disabled", "skipped"].includes(String(trace.rerank_status))
    && ["dense_count", "bm25_count", "fused_count", "authorized_count", "returned_count",
      "embedding_ms", "dense_ms", "bm25_ms", "fusion_ms", "authorization_ms", "rerank_ms", "total_ms"]
      .every((key) => isCount(trace[key]))
    && Array.isArray(retrieval.evidence) && retrieval.evidence.every((item) =>
      isObject(item) && ["citation_id", "chunk_id", "document_id", "knowledge_base_id",
        "document_title", "document_name", "content"].every((key) => typeof item[key] === "string")
      && isCount(item.rank) && isCount(item.chunk_index)
      && (item.page_number === null || isCount(item.page_number))
      && (item.section_title === null || typeof item.section_title === "string") && isObject(item.scores));
}

export async function searchQuery(
  query: string, knowledgeBaseId: string, signal: AbortSignal,
): Promise<QuerySearchResponse> {
  const text = query.trim();
  if (!text || [...text].length > 2000) throw new Error("请输入 1 到 2000 字的问题");
  // 计划与检索参数由服务端决定；客户端只提交问题和用户选择的范围。
  const body = { query: text, ...(knowledgeBaseId ? { knowledge_base_ids: [knowledgeBaseId] } : {}) };
  const requestSignal = AbortSignal.any([signal, AbortSignal.timeout(120_000)]);
  requestSignal.throwIfAborted();
  const response = await authenticatedFetch(import.meta.env.VITE_QUERY_SEARCH_URL?.trim() || "/api/query/search", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body), signal: requestSignal,
  });
  const data: unknown = await response.json().catch(() => null);
  requestSignal.throwIfAborted();
  if (!response.ok) throw new QuerySearchError(
    responseMessage(data, response.status === 422 ? "问题或检索范围无效，请检查后重试" : `检索失败（${response.status}），请重试`),
    response.status, responseCode(data),
  );
  if (!validResponse(data)) throw new Error("检索响应不完整，请重试");
  return data;
}
