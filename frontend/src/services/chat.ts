import { authenticatedFetch, authClient } from "./auth";
import { responseCode, responseMessage } from "./auth-client";
import { QuerySearchError, validQueryAnswerResponse, type QueryAnswerResponse, type QueryAnswerStage, type QueryRequestOptions } from "./query";

export interface ChatSession {
  id: string; title: string; status: "ACTIVE" | "ARCHIVED" | "DELETED";
  last_active_at: string; created_at: string;
}
export type ChatRunStatus = "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "CANCELLED" | "ABANDONED";
export interface ChatRun {
  id: string; session_id: string; client_request_id: string; question: string;
  standalone_question: string | null; status: ChatRunStatus; created_at: string;
  finished_at: string | null; error_code: string | null;
  response: QueryAnswerResponse | null; visibility: "visible" | "unavailable" | "pending";
}
export interface ChatHistory { runs: ChatRun[]; has_more: boolean; next_before_run_id: string | null }
export interface ChatRequest extends QueryRequestOptions { query: string; client_request_id: string; knowledge_base_ids?: string[]; document_ids?: string[] }
export type ChatStreamEvent = { type: "run"; run: ChatRun }
  | { type: "progress"; stage: QueryAnswerStage }
  | { type: "answer"; run: ChatRun; response: QueryAnswerResponse };

const endpoint = "/api/chat/sessions";
export const CHAT_SESSIONS_PAGE_SIZE = 50;
const configuredTimeout = Number(import.meta.env.VITE_CHAT_STREAM_TIMEOUT_MS);
const streamTimeout = Number.isFinite(configuredTimeout) && configuredTimeout >= 180_000 && configuredTimeout <= 600_000
  ? configuredTimeout : 180_000;
const object = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const id = (value: unknown): value is string => typeof value === "string" && /^[1-9]\d*$/.test(value);
const nullableString = (value: unknown) => value === null || typeof value === "string";
const timestamp = (value: unknown) => typeof value === "string" && Number.isFinite(Date.parse(value));
const statuses: ChatRunStatus[] = ["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED", "ABANDONED"];
export const isPendingRun = (run: ChatRun) => run.status === "PENDING" || run.status === "RUNNING";

function validSession(data: unknown): data is ChatSession {
  return object(data) && id(data.id) && typeof data.title === "string"
    && ["ACTIVE", "ARCHIVED", "DELETED"].includes(String(data.status))
    && timestamp(data.last_active_at) && timestamp(data.created_at);
}
function validRun(data: unknown, envelope = false): data is ChatRun {
  if (!object(data) || !id(data.id) || !id(data.session_id) || typeof data.client_request_id !== "string"
    || !data.client_request_id.trim() || typeof data.question !== "string" || !data.question.trim()
    || !nullableString(data.standalone_question) || !statuses.includes(data.status as ChatRunStatus)
    || !timestamp(data.created_at) || !(data.finished_at === null || timestamp(data.finished_at))
    || !nullableString(data.error_code) || !["visible", "unavailable", "pending"].includes(String(data.visibility))) return false;
  const pending = data.status === "PENDING" || data.status === "RUNNING";
  if (pending !== (data.finished_at === null) || pending !== (data.visibility === "pending")) return false;
  if (data.visibility !== "visible") return data.response === null;
  if (data.status !== "SUCCEEDED") return data.response === null;
  return envelope ? data.response === null : validQueryAnswerResponse(data.response);
}
function sessionUrl(sessionId: string) {
  if (!id(sessionId)) throw new Error("会话标识无效");
  return `${endpoint}/${sessionId}`;
}
async function open(url: string, signal: AbortSignal, init?: RequestInit, stream = false) {
  const requestSignal = AbortSignal.any([signal, AbortSignal.timeout(stream ? streamTimeout : 20_000)]);
  requestSignal.throwIfAborted();
  const response = await authenticatedFetch(url, { ...init, signal: requestSignal });
  requestSignal.throwIfAborted();
  if (!response.ok) {
    const data: unknown = await response.json().catch(() => null);
    requestSignal.throwIfAborted();
    throw new QuerySearchError(responseMessage(data, `会话请求失败（${response.status}），请重试`), response.status, responseCode(data));
  }
  return { response, requestSignal };
}
async function json(url: string, signal: AbortSignal, init?: RequestInit): Promise<unknown> {
  const { response, requestSignal } = await open(url, signal, init);
  const data: unknown = await response.json().catch(() => null);
  requestSignal.throwIfAborted();
  return data;
}
export async function listChatSessions(signal: AbortSignal, offset = 0): Promise<ChatSession[]> {
  if (!Number.isSafeInteger(offset) || offset < 0) throw new Error("会话分页位置无效");
  const data = await json(`${endpoint}?offset=${offset}`, signal);
  if (!object(data) || !Array.isArray(data.sessions) || data.sessions.length > CHAT_SESSIONS_PAGE_SIZE
    || !data.sessions.every(validSession) || new Set(data.sessions.map((session) => session.id)).size !== data.sessions.length)
    throw new Error("会话列表响应不完整，请重试");
  return data.sessions;
}
export async function createChatSession(signal: AbortSignal, title?: string): Promise<ChatSession> {
  const data = await json(endpoint, signal, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(title ? { title } : {}) });
  if (!validSession(data)) throw new Error("新会话响应不完整，请重试");
  return data;
}
export async function listChatRuns(sessionId: string, signal: AbortSignal, beforeRunId?: string): Promise<ChatHistory> {
  if (beforeRunId !== undefined && !id(beforeRunId)) throw new Error("历史游标无效");
  const data = await json(`${sessionUrl(sessionId)}/runs?limit=20${beforeRunId ? `&before_run_id=${beforeRunId}` : ""}`, signal);
  if (!object(data) || !Array.isArray(data.runs) || !data.runs.every((run) => validRun(run) && run.session_id === sessionId)
    || typeof data.has_more !== "boolean" || !(data.next_before_run_id === null || id(data.next_before_run_id))
    || data.has_more !== (data.next_before_run_id !== null)) throw new Error("会话历史响应不完整，请重试");
  const runs = data.runs as ChatRun[];
  if (runs.some((run, index) => index > 0 && BigInt(run.id) <= BigInt(runs[index - 1]!.id))) throw new Error("会话历史顺序无效，请重试");
  return data as unknown as ChatHistory;
}
export async function getChatRun(sessionId: string, runId: string, signal: AbortSignal): Promise<ChatRun> {
  if (!id(runId)) throw new Error("轮次标识无效");
  const data = await json(`${sessionUrl(sessionId)}/runs/${runId}`, signal);
  if (!validRun(data) || data.session_id !== sessionId || data.id !== runId) throw new Error("会话轮次响应不完整，请重试");
  return data;
}
/** Reuse the returned object unchanged on network retries; drafts are separate UI state. */
export function createChatRequest(query: string, knowledgeBaseId = "", requestId: string = crypto.randomUUID(), options: QueryRequestOptions = {}): ChatRequest {
  const text = query.trim();
  if (!text || [...text].length > 2000) throw new Error("请输入 1 到 2000 字的问题");
  if (!requestId.trim() || requestId.length > 128) throw new Error("请求标识无效");
  if (knowledgeBaseId && !id(knowledgeBaseId)) throw new Error("知识库标识无效");
  return { query: text, client_request_id: requestId, ...(knowledgeBaseId ? { knowledge_base_ids: [knowledgeBaseId] } : {}), ...options };
}
function requestInit(request: ChatRequest, stream = false): RequestInit {
  if (!request.query.trim() || [...request.query].length > 2000 || !request.client_request_id.trim() || request.client_request_id.length > 128)
    throw new Error("问题或请求标识无效");
  return { method: "POST", headers: { "Content-Type": "application/json", ...(stream ? { Accept: "text/event-stream" } : {}) }, body: JSON.stringify(request) };
}
export async function answerChat(sessionId: string, request: ChatRequest, signal: AbortSignal): Promise<ChatRun> {
  const data = await json(`${sessionUrl(sessionId)}/answer`, signal, requestInit(request));
  if (!validRun(data) || data.session_id !== sessionId || data.client_request_id !== request.client_request_id)
    throw new Error("会话回答响应不完整，请重试");
  return data;
}
export async function streamChatAnswer(sessionId: string, request: ChatRequest, signal: AbortSignal,
  onEvent: (event: ChatStreamEvent) => void): Promise<ChatRun> {
  const { response, requestSignal } = await open(`${sessionUrl(sessionId)}/answer/stream`, signal, requestInit(request, true), true);
  if (response.headers.get("Content-Type")?.split(";")[0]?.trim().toLowerCase() !== "text/event-stream" || !response.body) {
    await response.body?.cancel().catch(() => {});
    throw new Error("会话回答流响应无效，请恢复本轮");
  }
  const reader = response.body.getReader(), decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "", eventName = "", lines: string[] = [], size = 0;
  let runId = "", metadata = "", statements = "", count = 0, stageIndex = -1, receivedRun = false, structuredReceived = false;
  const stages: QueryAnswerStage[] = ["analysis", "retrieval", "generation", "review"];
  const invalid = () => new Error("会话回答流或引用不完整，请恢复本轮");
  const abortReader = () => { void reader.cancel().catch(() => {}); };
  requestSignal.addEventListener("abort", abortReader, { once: true });
  function checkRun(data: unknown): asserts data is ChatRun {
    if (!validRun(data, true) || data.session_id !== sessionId || data.client_request_id !== request.client_request_id
      || data.question !== request.query || (runId && data.id !== runId)) throw invalid();
    runId = data.id;
  }
  function dispatch(): ChatRun | null {
    if (!lines.length) return null;
    let data: unknown;
    try { data = JSON.parse(lines.join("\n")); } catch { throw invalid(); }
    requestSignal.throwIfAborted();
    if (eventName === "error") {
      if (!object(data) || !Number.isInteger(data.status) || Number(data.status) < 400 || Number(data.status) > 599
        || typeof data.code !== "string" || typeof data.message !== "string") throw invalid();
      if (data.status === 401) authClient.clearSession("登录已失效，请重新登录");
      throw new QuerySearchError(data.message, Number(data.status), data.code);
    }
    if (eventName === "run") {
      checkRun(data);
      if (receivedRun || count) throw invalid();
      receivedRun = true;
      onEvent({ type: "run", run: data });
      return null;
    }
    if (eventName === "progress") {
      if (!object(data)) throw invalid();
      const next = stages.indexOf(data.stage as QueryAnswerStage);
      if (next <= stageIndex || next < 0 || count || structuredReceived) throw invalid();
      stageIndex = next;
      onEvent({ type: "progress", stage: stages[next]! });
      return null;
    }
    if (!object(data) || (eventName !== "answer" && eventName !== "done")) throw invalid();
    checkRun(data.run);
    if (isPendingRun(data.run)) throw invalid();
    if (data.response === null) {
      if (eventName !== "done" || count || (data.run.visibility === "visible" && data.run.status === "SUCCEEDED")) throw invalid();
      return data.run;
    }
    if (!validQueryAnswerResponse(data.response) || data.run.visibility !== "visible") throw invalid();
    const answer = data.response.answer;
    const key = JSON.stringify({ ...data.response, answer: { ...answer, statements: [] } });
    if (metadata && key !== metadata) throw invalid();
    const fullRun = { ...data.run, response: data.response };
    if (eventName === "done") {
      if (count && JSON.stringify(answer.statements) !== statements) throw invalid();
      return fullRun;
    }
    if (answer.status === "structured") {
      if (count || structuredReceived) throw invalid();
      metadata = key; structuredReceived = true;
      onEvent({ type: "answer", run: fullRun, response: data.response });
      return null;
    }
    if (answer.status !== "answered" || answer.statements.length !== count + 1
      || (count && JSON.stringify(answer.statements.slice(0, -1)) !== statements)) throw invalid();
    metadata = key; statements = JSON.stringify(answer.statements); count++;
    onEvent({ type: "answer", run: fullRun, response: data.response });
    return null;
  }
  try {
    while (true) {
      requestSignal.throwIfAborted();
      const chunk = await reader.read();
      requestSignal.throwIfAborted();
      buffer += decoder.decode(chunk.value, { stream: !chunk.done });
      let separator: RegExpExecArray | null;
      while ((separator = /\r\n|\r|\n/.exec(buffer))) {
        if (separator[0] === "\r" && separator.index === buffer.length - 1 && !chunk.done) break;
        const line = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator[0].length);
        size += line.length;
        if (size > 1_000_000) throw invalid();
        if (!line) {
          const final = dispatch(); eventName = ""; lines = []; size = 0;
          requestSignal.throwIfAborted();
          if (final) return final;
        } else if (!line.startsWith(":")) {
          const colon = line.indexOf(":"), field = colon < 0 ? line : line.slice(0, colon);
          const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
          if (field === "event") eventName = value;
          else if (field === "data") lines.push(value);
        }
      }
      if (size + buffer.length > 1_000_000) throw invalid();
      if (chunk.done) throw new Error("回答连接中断，正在恢复服务器已保存的本轮结果");
    }
  } finally {
    requestSignal.removeEventListener("abort", abortReader);
    await reader.cancel().catch(() => {}); reader.releaseLock();
  }
}

export function chatSessionStorageKey(tenantId: string, userId: string) {
  return `zhiqing-chat-session:${encodeURIComponent(tenantId)}:${encodeURIComponent(userId)}`;
}
