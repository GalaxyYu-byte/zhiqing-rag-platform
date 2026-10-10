import { authenticatedFetch, authClient } from "./auth";
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

async function openQueryResponse(
  url: string, query: string, knowledgeBaseId: string, signal: AbortSignal, stream = false,
): Promise<{ response: Response; requestSignal: AbortSignal }> {
  const text = query.trim();
  if (!text || [...text].length > 2000) throw new Error("请输入 1 到 2000 字的问题");
  // 计划与检索参数由服务端决定；客户端只提交问题和用户选择的范围。
  const body = { query: text, ...(knowledgeBaseId ? { knowledge_base_ids: [knowledgeBaseId] } : {}) };
  const requestSignal = AbortSignal.any([signal, AbortSignal.timeout(120_000)]);
  requestSignal.throwIfAborted();
  const response = await authenticatedFetch(url, {
    method: "POST", headers: { "Content-Type": "application/json", ...(stream ? { Accept: "text/event-stream" } : {}) },
    body: JSON.stringify(body), signal: requestSignal,
  });
  requestSignal.throwIfAborted();
  if (!response.ok) {
    const data: unknown = await response.json().catch(() => null);
    requestSignal.throwIfAborted();
    throw new QuerySearchError(
      responseMessage(data, response.status === 422 ? "问题或检索范围无效，请检查后重试" : `检索失败（${response.status}），请重试`),
      response.status, responseCode(data),
    );
  }
  return { response, requestSignal };
}

async function requestQuery(
  url: string, query: string, knowledgeBaseId: string, signal: AbortSignal,
): Promise<unknown> {
  const { response, requestSignal } = await openQueryResponse(url, query, knowledgeBaseId, signal);
  const data: unknown = await response.json().catch(() => null);
  requestSignal.throwIfAborted();
  return data;
}

export async function searchQuery(
  query: string, knowledgeBaseId: string, signal: AbortSignal,
): Promise<QuerySearchResponse> {
  const data = await requestQuery(import.meta.env.VITE_QUERY_SEARCH_URL?.trim() || "/api/query/search",
    query, knowledgeBaseId, signal);
  if (!validResponse(data)) throw new Error("检索响应不完整，请重试");
  return data;
}

export interface QueryAnswerResult {
  status: "answered" | "insufficient" | "rejected" | "unavailable" | "skipped" | "unsupported";
  statements: { text: string; citations: { citation_id: string; quote: string }[] }[];
  message: string | null;
  codes: string[];
  model: string | null;
  review_status: "passed" | "failed" | "skipped";
  trace: {
    generation_ms: number; review_ms: number; authorization_ms: number; total_ms: number;
    context_tokens: number; input_tokens: number; output_tokens: number;
  };
}
export interface QueryAnswerResponse extends QuerySearchResponse { answer: QueryAnswerResult }

/** Shared by persisted chat history and SSE: never display unreviewed or unverifiable text. */
export function validQueryAnswerResponse(data: unknown): data is QueryAnswerResponse {
  return validResponse(data) && validAnswer(data);
}

function validAnswer(data: QuerySearchResponse & { answer?: unknown }): data is QueryAnswerResponse {
  const answer = data.answer;
  if (!isObject(answer) || !isStrings(answer.codes) || !isObject(answer.trace)
    || !["answered", "insufficient", "rejected", "unavailable", "skipped", "unsupported"].includes(String(answer.status))
    || !["passed", "failed", "skipped"].includes(String(answer.review_status))
    || !(answer.message === null || typeof answer.message === "string")
    || !(answer.model === null || typeof answer.model === "string")
    || !["generation_ms", "review_ms", "authorization_ms", "total_ms", "context_tokens", "input_tokens", "output_tokens"]
      .every((key) => isCount((answer.trace as Record<string, unknown>)[key]))
    || !Array.isArray(answer.statements) || answer.statements.length > 8) return false;
  if (answer.status !== "answered") return answer.statements.length === 0;
  if (answer.review_status !== "passed" || !answer.statements.length) return false;
  const sources = new Map(data.retrieval.evidence.map((item) => [item.citation_id, item.content]));
  return answer.statements.every((item) => isObject(item) && typeof item.text === "string" && !!item.text.trim()
    && Array.isArray(item.citations) && item.citations.length > 0 && item.citations.length <= 5
    && item.citations.every((citation) => isObject(citation) && typeof citation.citation_id === "string"
      && typeof citation.quote === "string" && citation.quote.length >= 2
      && sources.get(citation.citation_id)?.includes(citation.quote)));
}

export async function answerQuery(
  query: string, knowledgeBaseId: string, signal: AbortSignal,
): Promise<QueryAnswerResponse> {
  const data = await requestQuery(import.meta.env.VITE_QUERY_ANSWER_URL?.trim() || "/api/query/answer",
    query, knowledgeBaseId, signal);
  if (!validResponse(data) || !validAnswer(data)) throw new Error("回答或引用响应不完整，请重试");
  return data;
}

export type QueryAnswerStage = "analysis" | "retrieval" | "generation" | "review";
export type QueryAnswerStreamEvent =
  | { type: "progress"; stage: QueryAnswerStage }
  | { type: "answer"; response: QueryAnswerResponse };

export async function streamAnswerQuery(
  query: string, knowledgeBaseId: string, signal: AbortSignal,
  onEvent: (event: QueryAnswerStreamEvent) => void,
): Promise<QueryAnswerResponse> {
  const { response, requestSignal } = await openQueryResponse(
    import.meta.env.VITE_QUERY_ANSWER_STREAM_URL?.trim() || "/api/query/answer/stream",
    query, knowledgeBaseId, signal, true,
  );
  if (response.headers.get("Content-Type")?.split(";")[0]?.trim().toLowerCase() !== "text/event-stream"
    || !response.body) {
    await response.body?.cancel().catch(() => {});
    throw new Error("回答流响应无效，请重试");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const maxFrameSize = 1_000_000;
  let buffer = "", eventName = "", dataLines: string[] = [], frameSize = 0;
  let metadata = "", statements = "", statementCount = 0, lastStage = -1;
  const stages: QueryAnswerStage[] = ["analysis", "retrieval", "generation", "review"];
  const invalid = () => new Error("回答流或引用不完整，请重试");
  const cancelReader = () => { void reader.cancel().catch(() => {}); };
  requestSignal.addEventListener("abort", cancelReader, { once: true });

  function dispatch(): QueryAnswerResponse | null {
    if (!dataLines.length) return null; // 心跳注释不触发业务回调。
    let data: unknown;
    try { data = JSON.parse(dataLines.join("\n")); } catch { throw invalid(); }
    requestSignal.throwIfAborted();
    if (eventName === "error") {
      if (!isObject(data) || !Number.isInteger(data.status) || Number(data.status) < 400 || Number(data.status) > 599
        || typeof data.code !== "string" || typeof data.message !== "string") throw invalid();
      if (data.status === 401) authClient.clearSession("登录已失效，请重新登录");
      throw new QuerySearchError(data.message, Number(data.status), data.code);
    }
    if (eventName === "progress") {
      if (!isObject(data)) throw invalid();
      const index = stages.indexOf(data.stage as QueryAnswerStage);
      if (index <= lastStage || index < 0 || statementCount) throw invalid();
      lastStage = index;
      onEvent({ type: "progress", stage: stages[index]! });
      return null;
    }
    if (eventName !== "answer" && eventName !== "done") throw invalid();
    // 每个段落在显示前都校验复核状态、来源 ID 和逐字引文。
    if (!validResponse(data) || !validAnswer(data)) throw invalid();
    const key = JSON.stringify({ ...data, answer: { ...data.answer, statements: [] } });
    if (metadata && metadata !== key) throw invalid();
    if (eventName === "done") {
      if (data.answer.status === "answered" && (!statementCount
        || JSON.stringify(data.answer.statements) !== statements)) throw invalid();
      if (data.answer.status !== "answered" && statementCount) throw invalid();
      return data;
    }
    if (data.answer.status !== "answered" || data.answer.statements.length !== statementCount + 1
      || (statementCount && JSON.stringify(data.answer.statements.slice(0, -1)) !== statements)) throw invalid();
    metadata = key;
    statements = JSON.stringify(data.answer.statements);
    statementCount++;
    onEvent({ type: "answer", response: data });
    return null;
  }

  try {
    while (true) {
      requestSignal.throwIfAborted();
      const chunk = await reader.read();
      requestSignal.throwIfAborted();
      buffer += decoder.decode(chunk.value, { stream: !chunk.done });
      // 支持 LF / CRLF / CR、跨网络分块的换行、多行 data 和跨字节的中文。
      let separator: RegExpExecArray | null;
      while ((separator = /\r\n|\r|\n/.exec(buffer))) {
        if (separator[0] === "\r" && separator.index === buffer.length - 1 && !chunk.done) break;
        const line = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator[0].length);
        frameSize += line.length;
        if (frameSize > maxFrameSize) throw invalid();
        if (!line) {
          const final = dispatch();
          eventName = ""; dataLines = []; frameSize = 0;
          requestSignal.throwIfAborted();
          if (final) return final;
        } else if (!line.startsWith(":")) {
          const colon = line.indexOf(":");
          const field = colon < 0 ? line : line.slice(0, colon);
          const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") eventName = value;
          else if (field === "data") dataLines.push(value);
        }
      }
      if (frameSize + buffer.length > maxFrameSize) throw invalid();
      // done 是唯一成功终点；EOF、缺终止事件或截断帧均不能当作完整回答。
      if (chunk.done) throw new Error("回答连接中断，请重试");
    }
  } finally {
    requestSignal.removeEventListener("abort", cancelReader);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
