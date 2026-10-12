import { authenticatedFetch, authClient } from "./auth";
import { responseCode, responseMessage } from "./auth-client";

export type QueryRoute = "hybrid" | "hybrid_graph" | "structured" | "clarify" | "none";
export type RetrievalStatus = "ok" | "degraded" | "empty" | "skipped" | "unsupported";
export interface MetadataFilter {
  field: "department" | "document_name" | "document_code" | "file_type" | "business_status" | "uploaded_at" | "event_time";
  operator: "eq" | "neq" | "contains" | "gt" | "gte" | "lt" | "lte" | "range";
  value: string;
}
export interface ResolvedFilter { field: string; operator: string; values: string[] }
export interface QueryRequestOptions { filters?: MetadataFilter[]; offset?: number; limit?: number }
export interface QueryRequest extends QueryRequestOptions { query: string; knowledge_base_ids?: string[] | null; document_ids?: string[] | null }
export interface StructuredDocument {
  knowledge_base_id: string; document_id: string; revision_id: string; generation_id: string;
  title: string; document_name: string; file_type: string; business_status: string;
  department_id: string | null; event_date: string | null; uploaded_at: string;
}
export interface StructuredResult {
  operation: "count_documents" | "list_documents"; total: number; documents: StructuredDocument[];
  offset: number; limit: number; applied_filters: ResolvedFilter[]; executed_at: string;
  scope: "authorized_published_documents";
}
export interface GraphRelationEvidence {
  assertion_id: string; seed_entity_uid: string;
  subject_entity_uid: string; object_entity_uid: string;
  predicate: string; condition_text: string; source_quote: string;
  quote_start: number; quote_end: number;
}
export interface GraphProjectionRef {
  tenant_id: string; document_id: string; generation_id: string;
  projection_id: string; namespace: string;
}
export interface GraphSourceEvidence extends GraphProjectionRef {
  knowledge_base_id: string; revision_id: string; chunk_id: string;
  content_sha256: string; source_quote: string; quote_start: number; quote_end: number;
}
export interface GraphHopEvidence {
  relation: GraphRelationEvidence; source: GraphSourceEvidence;
  traversal: "outgoing" | "incoming"; subject_name: string; object_name: string;
}
export interface GraphBridgeEvidence {
  identity_id?: string | null; identity_version?: number;
  source_entity_uid: string; target_entity_uid: string;
  entity_type: "Material" | "Process" | "Actor"; canonical_name: string;
  basis: "explicit_document" | "shared_context" | "explicit_alias";
  sources: GraphSourceEvidence[];
}
export interface GraphPathEvidence {
  path_id: string; knowledge_base_id: string; hops: GraphHopEvidence[];
  bridges: GraphBridgeEvidence[]; required_chunk_ids: string[];
}
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
  graph_relations?: GraphRelationEvidence[];
  scores: {
    dense_rank: number | null; dense_score: number | null;
    bm25_rank: number | null; bm25_score: number | null;
    graph_rank?: number | null; graph_score?: number | null;
    fusion_score: number; rerank_score: number | null;
  };
}
export interface RetrievalTrace {
  dense_status: "succeeded" | "failed" | "skipped";
  bm25_status: "succeeded" | "failed" | "skipped";
  graph_status?: "succeeded" | "failed" | "skipped" | "empty";
  graph_count?: number; graph_seed_count?: number; graph_assertion_count?: number;
  graph_hops?: number; graph_ms?: number;
  graph_path_count?: number; graph_path_candidate_count?: number;
  rerank_status: "succeeded" | "failed" | "disabled" | "skipped";
  dense_count: number; bm25_count: number; fused_count: number;
  authorized_count: number; returned_count: number;
  embedding_ms: number; dense_ms: number; bm25_ms: number;
  fusion_ms: number; authorization_ms: number; rerank_ms: number; total_ms: number;
}
export interface QuerySearchResponse {
  structured?: StructuredResult | null;
  request?: QueryRequest | null;
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
    resolved_filters?: ResolvedFilter[];
    message: string | null;
    params: { question?: string; candidate_k?: number; top_k?: number; rerank?: boolean;
      max_hops?: number; seed_entity_uids?: string[]; matched_entity_names?: string[]; projections?: GraphProjectionRef[] };
  };
  scope: {
    knowledge_base_ids: string[];
    authorized_document_count: number;
    published_document_count: number;
  };
  retrieval: {
    status: RetrievalStatus;
    evidence: QueryEvidence[];
    graph_paths?: GraphPathEvidence[];
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
const isStrings = (value: unknown): value is string[] => Array.isArray(value) && value.every((item) => typeof item === "string");
const isIntegerCount = (value: unknown) => isCount(value) && Number.isSafeInteger(value);
const isId = (value: unknown): value is string => typeof value === "string"
  && /^[1-9][0-9]{0,18}$/.test(value) && BigInt(value) <= 9223372036854775807n;
const isText = (value: unknown, max = 200) => typeof value === "string" && value.length > 0 && value.length <= max;
const graphPredicates = new Set(["HAS_STEP", "NEXT_STEP", "RESPONSIBLE_FOR", "REVIEWS", "CONFIRMS",
  "APPLIES_TO", "REQUIRES_MATERIAL", "PRODUCES_MATERIAL", "OPERATES_ON", "HAS_FIELD", "HAS_STATE",
  "GOVERNED_BY", "DEPENDS_ON", "ESCALATES_TO", "TRANSITIONS_TO"]);

function validQueryRequest(value: unknown): value is QueryRequest {
  if (!isObject(value) || typeof value.query !== "string" || !value.query.trim() || [...value.query].length > 2000
    || Object.keys(value).some((key) => !["query", "knowledge_base_ids", "document_ids", "filters", "offset", "limit"].includes(key))) return false;
  for (const [field, maximum] of [["knowledge_base_ids", 50], ["document_ids", 200]] as const) {
    const ids = value[field];
    if (ids !== undefined && ids !== null && (!Array.isArray(ids) || !ids.length || ids.length > maximum
      || !ids.every(isId) || new Set(ids).size !== ids.length)) return false;
  }
  if (value.offset !== undefined && (!isIntegerCount(value.offset) || Number(value.offset) > 100000)) return false;
  if (value.limit !== undefined && (!isIntegerCount(value.limit) || Number(value.limit) < 1 || Number(value.limit) > 100)) return false;
  if (value.filters !== undefined && (!Array.isArray(value.filters) || value.filters.length > 20
    || !value.filters.every((filter) => isObject(filter) && Object.keys(filter).length === 3
      && ["department", "document_name", "document_code", "file_type", "business_status", "uploaded_at", "event_time"].includes(String(filter.field))
      && ["eq", "neq", "contains", "gt", "gte", "lt", "lte", "range"].includes(String(filter.operator))
      && typeof filter.value === "string" && !!filter.value.trim() && filter.value.length <= 200))) return false;
  return true;
}

function validStructured(result: QuerySearchResponse): boolean {
  const value: unknown = result.structured;
  if (value === undefined || value === null) return true;
  if (!isObject(value) || result.plan.route !== "structured" || result.plan.planning_status !== "planned"
    || !["count_documents", "list_documents"].includes(String(value.operation))
    || !isIntegerCount(value.total) || !isIntegerCount(value.offset) || Number(value.offset) > 100000
    || !isIntegerCount(value.limit) || Number(value.limit) < 1 || Number(value.limit) > 100
    || value.scope !== "authorized_published_documents" || typeof value.executed_at !== "string"
    || !Number.isFinite(Date.parse(value.executed_at)) || !Array.isArray(value.applied_filters)
    || value.applied_filters.length > 40 || !value.applied_filters.every((filter) => isObject(filter)
      && ["department_id", "document_name", "document_code", "file_type", "business_status", "uploaded_at", "event_date"].includes(String(filter.field))
      && ["eq", "neq", "contains", "gt", "gte", "lt", "lte", "range"].includes(String(filter.operator))
      && Array.isArray(filter.values) && filter.values.length >= 1 && filter.values.length <= 10
      && filter.values.every((item) => typeof item === "string"
        && (filter.field !== "department_id" || isId(item))))
    || !Array.isArray(value.documents) || value.documents.length > Number(value.limit)
    || !value.documents.every((doc) => isObject(doc)
      && ["knowledge_base_id", "document_id", "revision_id", "generation_id"].every((key) => isId(doc[key]))
      && result.scope.knowledge_base_ids.includes(String(doc.knowledge_base_id))
      && ["title", "document_name", "file_type", "business_status"].every((key) => typeof doc[key] === "string")
      && (doc.department_id === null || isId(doc.department_id))
      && (doc.event_date === null || (typeof doc.event_date === "string" && /^\d{4}-\d{2}-\d{2}$/.test(doc.event_date)))
      && typeof doc.uploaded_at === "string" && Number.isFinite(Date.parse(doc.uploaded_at)))) return false;
  if (value.operation === "count_documents") return value.documents.length === 0;
  return value.documents.length === Math.min(Number(value.limit), Math.max(0, Number(value.total) - Number(value.offset)))
    && new Set(value.documents.map((doc) => doc.document_id)).size === value.documents.length;
}

export function structuredAnswerText(value: StructuredResult): string {
  const prefix = "当前可读的已发布文档中，符合条件的共有";
  if (value.operation === "count_documents") return `${prefix} ${value.total} 份。`;
  if (!value.documents.length) return `${prefix} ${value.total} 份，本页没有文档。`;
  return `${prefix} ${value.total} 份，当前展示第 ${value.offset + 1}—${value.offset + value.documents.length} 份。`;
}

function exactQuote(content: string, quote: unknown, start: unknown, end: unknown): boolean {
  if (typeof quote !== "string" || !quote.length || !isIntegerCount(start) || !isIntegerCount(end)) return false;
  // Python offsets count Unicode code points; JS string.slice counts UTF-16 code units.
  const points = [...content], first = Number(start), last = Number(end);
  return first < last && last <= points.length && last - first === [...quote].length
    && points.slice(first, last).join("") === quote;
}

function validRelation(value: unknown, evidence: QueryEvidence): value is GraphRelationEvidence {
  if (!isObject(value)) return false;
  return ["assertion_id", "seed_entity_uid", "subject_entity_uid", "object_entity_uid"]
    .every((key) => isText(value[key], 150))
    && typeof value.predicate === "string" && graphPredicates.has(value.predicate)
    && typeof value.condition_text === "string" && value.condition_text.length <= 2000
    && (value.condition_text === "" || (typeof value.source_quote === "string" && value.source_quote.includes(value.condition_text)))
    && exactQuote(evidence.content, value.source_quote, value.quote_start, value.quote_end)
    && (value.seed_entity_uid === value.subject_entity_uid || value.seed_entity_uid === value.object_entity_uid);
}

function sameRelation(left: GraphRelationEvidence, right: GraphRelationEvidence) {
  return ["assertion_id", "seed_entity_uid", "subject_entity_uid", "object_entity_uid", "predicate",
    "condition_text", "source_quote", "quote_start", "quote_end"]
    .every((key) => left[key as keyof GraphRelationEvidence] === right[key as keyof GraphRelationEvidence]);
}

function validGraphPaths(result: QuerySearchResponse): boolean {
  const paths: unknown = result.retrieval.graph_paths;
  const trace = result.retrieval.trace;
  if (paths === undefined) return trace.graph_path_count === undefined || trace.graph_path_count === 0;
  if (!Array.isArray(paths) || paths.length > 100
    || (trace.graph_path_count !== undefined && trace.graph_path_count !== paths.length)) return false;
  if (!paths.length) return true;
  const params = result.plan.params;
  if (result.plan.route !== "hybrid_graph" || trace.graph_status !== "succeeded" || trace.graph_hops !== 2
    || params.max_hops !== 2 || !isStrings(params.seed_entity_uids) || !params.seed_entity_uids?.length
    || params.seed_entity_uids.length > 20 || new Set(params.seed_entity_uids).size !== params.seed_entity_uids.length
    || !Array.isArray(params.projections) || !params.projections.length) return false;
  const seeds = new Set(params.seed_entity_uids);
  const evidence = new Map(result.retrieval.evidence.map((item) => [item.chunk_id, item]));
  const hashes = new Map<string, string>();
  const pathIds = new Set<string>();

  function sourceValid(value: unknown, knowledgeBaseId: string): value is GraphSourceEvidence {
    if (!isObject(value)
      || !["tenant_id", "knowledge_base_id", "document_id", "revision_id", "generation_id", "chunk_id", "projection_id"]
        .every((key) => isId(value[key]))
      || !isText(value.namespace, 100) || typeof value.content_sha256 !== "string"
      || !/^[a-f0-9]{64}$/.test(value.content_sha256)) return false;
    const item = evidence.get(String(value.chunk_id));
    if (!item || value.knowledge_base_id !== knowledgeBaseId
      || !result.scope.knowledge_base_ids.includes(knowledgeBaseId)
      || ["knowledge_base_id", "document_id", "revision_id", "generation_id", "chunk_id"]
        .some((key) => value[key] !== item[key as keyof QueryEvidence])
      || !params.projections!.some((projection) => isObject(projection)
        && ["tenant_id", "document_id", "generation_id", "projection_id", "namespace"]
          .every((key) => value[key] === projection[key]))
      || !exactQuote(item.content, value.source_quote, value.quote_start, value.quote_end)) return false;
    const previous = hashes.get(item.chunk_id);
    if (previous !== undefined && previous !== value.content_sha256) return false;
    hashes.set(item.chunk_id, value.content_sha256);
    return true;
  }

  return paths.every((path) => {
    if (!isObject(path) || !isText(path.path_id, 200) || pathIds.has(String(path.path_id))
      || !isId(path.knowledge_base_id) || !Array.isArray(path.hops) || path.hops.length !== 2
      || !Array.isArray(path.bridges) || path.bridges.length > 1
      || !isStrings(path.required_chunk_ids) || !path.required_chunk_ids.length
      || path.required_chunk_ids.length > 6 || !path.required_chunk_ids.every(isId)
      || new Set(path.required_chunk_ids).size !== path.required_chunk_ids.length) return false;
    pathIds.add(String(path.path_id));
    const knowledgeBaseId = path.knowledge_base_id;
    const sources: GraphSourceEvidence[] = [];
    const hops: GraphHopEvidence[] = [];
    for (const hop of path.hops) {
      if (!isObject(hop) || !["outgoing", "incoming"].includes(String(hop.traversal))
        || typeof hop.subject_name !== "string" || typeof hop.object_name !== "string"
        || hop.subject_name.length > 2000 || hop.object_name.length > 2000
        || !sourceValid(hop.source, knowledgeBaseId)) return false;
      const item = evidence.get(hop.source.chunk_id)!;
      if (!validRelation(hop.relation, item)
        || hop.relation.source_quote !== hop.source.source_quote
        || hop.relation.quote_start !== hop.source.quote_start || hop.relation.quote_end !== hop.source.quote_end
        || !item.graph_relations?.some((relation) => sameRelation(relation, hop.relation as GraphRelationEvidence))) return false;
      const checked = hop as unknown as GraphHopEvidence;
      const start = checked.traversal === "outgoing" ? checked.relation.subject_entity_uid : checked.relation.object_entity_uid;
      if (checked.relation.seed_entity_uid !== start) return false;
      hops.push(checked); sources.push(hop.source);
    }
    const start = (hop: GraphHopEvidence) => hop.traversal === "outgoing" ? hop.relation.subject_entity_uid : hop.relation.object_entity_uid;
    const end = (hop: GraphHopEvidence) => hop.traversal === "outgoing" ? hop.relation.object_entity_uid : hop.relation.subject_entity_uid;
    const first = hops[0]!, second = hops[1]!;
    if (!seeds.has(start(first)) || start(first) === end(first) || start(second) === end(second)
      || start(first) === end(second) || first.relation.assertion_id === second.relation.assertion_id) return false;
    if (!path.bridges.length && (end(first) !== start(second)
      || ["document_id", "namespace", "generation_id"]
        .some((key) => first.source[key as keyof GraphSourceEvidence] !== second.source[key as keyof GraphSourceEvidence]))) return false;
    for (const bridge of path.bridges) {
      if (!isObject(bridge) || bridge.source_entity_uid !== end(first) || bridge.target_entity_uid !== start(second)
        || (bridge.identity_id !== undefined && bridge.identity_id !== null
          && (typeof bridge.identity_id !== "string"
            || (bridge.identity_id !== "" && !/^[a-f0-9]{64}$/.test(bridge.identity_id))))
        || (bridge.identity_version !== undefined && bridge.identity_version !== 1)
        || !["Material", "Process", "Actor"].includes(String(bridge.entity_type))
        || !["explicit_document", "shared_context", "explicit_alias"].includes(String(bridge.basis))
        || !isText(bridge.canonical_name) || !Array.isArray(bridge.sources)
        || bridge.sources.length < 2 || bridge.sources.length > 4
        || !bridge.sources.every((source) => sourceValid(source, knowledgeBaseId))) return false;
      sources.push(...bridge.sources as GraphSourceEvidence[]);
    }
    const required = new Set(sources.map((source) => source.chunk_id));
    return required.size === path.required_chunk_ids.length
      && path.required_chunk_ids.every((id) => required.has(id));
  });
}

/** A second display guard also protects restored history supplied by UI state. */
export function validatedGraphPaths(result: QuerySearchResponse): GraphPathEvidence[] {
  return validResponse(result) ? result.retrieval.graph_paths ?? [] : [];
}

function validResponse(data: unknown): data is QuerySearchResponse {
  if (!isObject(data)) return false;
  const { analysis, plan, scope, retrieval } = data;
  if (!isObject(analysis) || !isObject(plan) || !isObject(scope) || !isObject(retrieval)) return false;
  const trace = retrieval.trace;
  const valid = ["ok", "degraded"].includes(String(analysis.status)) && isCount(analysis.latency_ms)
    && routes.includes(String(plan.route)) && routes.includes(String(plan.requested_route))
    && isObject(plan.params) && isStrings(plan.warnings)
    && (plan.params.matched_entity_names === undefined || (isStrings(plan.params.matched_entity_names)
      && plan.params.matched_entity_names.length <= 20 && plan.params.matched_entity_names.every((name) => !!name.trim() && [...name].length <= 200)))
    && isStrings(scope.knowledge_base_ids) && isCount(scope.authorized_document_count)
    && isCount(scope.published_document_count) && statuses.includes(String(retrieval.status))
    && isStrings(retrieval.degradation_codes) && isObject(trace)
    && ["succeeded", "failed", "skipped"].includes(String(trace.dense_status))
    && ["succeeded", "failed", "skipped"].includes(String(trace.bm25_status))
    && (trace.graph_status === undefined || ["succeeded", "failed", "skipped", "empty"].includes(String(trace.graph_status)))
    && ["graph_count", "graph_seed_count", "graph_assertion_count", "graph_hops", "graph_ms", "graph_path_count", "graph_path_candidate_count"]
      .every((key) => trace[key] === undefined || isIntegerCount(trace[key]))
    && (trace.graph_hops === undefined || Number(trace.graph_hops) <= 2)
    && (trace.graph_path_candidate_count === undefined || trace.graph_path_count === undefined
      || Number(trace.graph_path_candidate_count) >= Number(trace.graph_path_count))
    && ["succeeded", "failed", "disabled", "skipped"].includes(String(trace.rerank_status))
    && ["dense_count", "bm25_count", "fused_count", "authorized_count", "returned_count",
      "embedding_ms", "dense_ms", "bm25_ms", "fusion_ms", "authorization_ms", "rerank_ms", "total_ms"]
      .every((key) => isCount(trace[key]))
    && Array.isArray(retrieval.evidence) && retrieval.evidence.every((item) =>
      isObject(item) && ["citation_id", "chunk_id", "document_id", "knowledge_base_id",
        "document_title", "document_name", "content"].every((key) => typeof item[key] === "string")
      && isCount(item.rank) && isCount(item.chunk_index)
      && (item.page_number === null || isCount(item.page_number))
      && (item.section_title === null || typeof item.section_title === "string") && isObject(item.scores)
      && (item.graph_relations === undefined || (Array.isArray(item.graph_relations)
        && item.graph_relations.every((relation) => validRelation(relation, item as unknown as QueryEvidence)))));
  if (!valid) return false;
  const result = data as unknown as QuerySearchResponse;
  if (result.request !== undefined && result.request !== null && (!validQueryRequest(result.request)
    || result.request.knowledge_base_ids?.some((id) => !result.scope.knowledge_base_ids.includes(id)))) return false;
  return new Set(result.retrieval.evidence.map((item) => item.chunk_id)).size === result.retrieval.evidence.length
    && new Set(result.retrieval.evidence.map((item) => item.citation_id)).size === result.retrieval.evidence.length
    && result.retrieval.evidence.every((item) => /^E[1-9][0-9]*$/.test(item.citation_id))
    && validGraphPaths(result) && validStructured(result);
}

async function openQueryResponse(
  url: string, query: string, knowledgeBaseId: string, signal: AbortSignal, stream = false,
  options: QueryRequestOptions = {},
): Promise<{ response: Response; requestSignal: AbortSignal }> {
  const text = query.trim();
  if (!text || [...text].length > 2000) throw new Error("请输入 1 到 2000 字的问题");
  // 计划与检索参数由服务端决定；客户端只提交问题和用户选择的范围。
  const body = { query: text, ...(knowledgeBaseId ? { knowledge_base_ids: [knowledgeBaseId] } : {}), ...options };
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
  url: string, query: string, knowledgeBaseId: string, signal: AbortSignal, options: QueryRequestOptions = {},
): Promise<unknown> {
  const { response, requestSignal } = await openQueryResponse(url, query, knowledgeBaseId, signal, false, options);
  const data: unknown = await response.json().catch(() => null);
  requestSignal.throwIfAborted();
  return data;
}

export async function searchQuery(
  query: string, knowledgeBaseId: string, signal: AbortSignal, options: QueryRequestOptions = {},
): Promise<QuerySearchResponse> {
  const data = await requestQuery(import.meta.env.VITE_QUERY_SEARCH_URL?.trim() || "/api/query/search",
    query, knowledgeBaseId, signal, options);
  if (!validResponse(data)) throw new Error("检索响应不完整，请重试");
  return data;
}

export async function searchQueryPage(request: QueryRequest, offset: number, signal: AbortSignal): Promise<QuerySearchResponse> {
  if (!validQueryRequest(request)) throw new Error("清单请求不完整，请重新查询");
  if (!Number.isSafeInteger(offset) || offset < 0 || offset > 100000) throw new Error("清单分页位置无效");
  const { query, ...options } = request;
  const data = await requestQuery(import.meta.env.VITE_QUERY_SEARCH_URL?.trim() || "/api/query/search",
    query, "", signal, { ...options, offset });
  if (!validResponse(data)) throw new Error("清单响应不完整，请重试");
  return data;
}

export interface QueryAnswerResult {
  status: "answered" | "structured" | "insufficient" | "rejected" | "unavailable" | "skipped" | "unsupported";
  statements: { text: string; citations: { citation_id: string; quote: string }[] }[];
  message: string | null;
  codes: string[];
  model: string | null;
  review_status: "passed" | "failed" | "skipped";
  trace: {
    generation_ms: number; review_ms: number; authorization_ms: number; total_ms: number;
    context_tokens: number; input_tokens: number; output_tokens: number;
    repair_count?: number;
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
    || !["answered", "structured", "insufficient", "rejected", "unavailable", "skipped", "unsupported"].includes(String(answer.status))
    || !["passed", "failed", "skipped"].includes(String(answer.review_status))
    || !(answer.message === null || typeof answer.message === "string")
    || !(answer.model === null || typeof answer.model === "string")
    || !["generation_ms", "review_ms", "authorization_ms", "total_ms", "context_tokens", "input_tokens", "output_tokens"]
      .every((key) => isCount((answer.trace as Record<string, unknown>)[key]))
    || (answer.trace.repair_count !== undefined
      && (!isIntegerCount(answer.trace.repair_count) || Number(answer.trace.repair_count) > 1))
    || !Array.isArray(answer.statements) || answer.statements.length > 8) return false;
  if (answer.status === "structured") return answer.statements.length === 0 && answer.review_status === "passed"
    && !!data.structured && answer.message === structuredAnswerText(data.structured);
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
  let metadata = "", statements = "", statementCount = 0, lastStage = -1, structuredReceived = false;
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
      if (index <= lastStage || index < 0 || statementCount || structuredReceived) throw invalid();
      lastStage = index;
      onEvent({ type: "progress", stage: stages[index]! });
      return null;
    }
    if (eventName !== "answer" && eventName !== "done") throw invalid();
    // 每个段落在显示前都校验复核状态、来源 ID 和逐字引文。
    if (!validResponse(data) || !validAnswer(data)) throw invalid();
    const key = JSON.stringify({ ...data, answer: { ...data.answer, statements: [] } });
    if (metadata && metadata !== key) throw invalid();
    if (data.answer.status === "structured") {
      if (statementCount || (eventName === "answer" && structuredReceived)) throw invalid();
      if (eventName === "done") return data;
      metadata = key; structuredReceived = true;
      onEvent({ type: "answer", response: data });
      return null;
    }
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
