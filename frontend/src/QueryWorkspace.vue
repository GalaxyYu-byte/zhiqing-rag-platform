<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { FileText, LoaderCircle, Search, RefreshCw, X, ShieldCheck, Plus, MessageSquare } from "lucide-vue-next";
import { isDemoMode, type KnowledgeBaseOption } from "./services/upload";
import { searchQuery, searchQueryPage, QuerySearchError, validatedGraphPaths, type MetadataFilter, type QuerySearchResponse, type QueryAnswerResponse, type QueryAnswerResult, type QueryRoute, type QueryAnswerStage } from "./services/query";
import { analysisFailureLabel, queryStatus, slowestQueryStage, graphPathViews } from "./services/query-diagnostics";
import { authState } from "./services/auth";
import { listChatSessions, createChatSession, listChatRuns, getChatRun, createChatRequest, streamChatAnswer, CHAT_SESSIONS_PAGE_SIZE,
  isPendingRun, chatSessionStorageKey, type ChatSession, type ChatRun, type ChatRequest } from "./services/chat";

const props = defineProps<{ bases: KnowledgeBaseOption[]; basesLoading: boolean; basesError: string }>();
const emit = defineEmits<{ reloadBases: [] }>();
const query = ref("");
const selectedBase = ref("");
const filterDepartment = ref("");
const filterName = ref("");
const filterCode = ref("");
const filterFileType = ref("");
const filterStatus = ref("");
const filterUploadFrom = ref("");
const filterUploadTo = ref("");
const filterEventFrom = ref("");
const filterEventTo = ref("");
const businessStatusLabels: Record<string, string> = { UNKNOWN: "未知", DRAFT: "草稿", EFFECTIVE: "生效", ARCHIVED: "归档", REPEALED: "废止" };
const filterLabels: Record<string, string> = { department_id: "部门", document_name: "文件名", document_code: "文档编号", file_type: "文件类型", business_status: "业务状态", uploaded_at: "发布版本上传时间", event_date: "业务日期" };
const operatorLabels: Record<string, string> = { eq: "等于", neq: "不等于", contains: "包含", gt: "晚于", gte: "不早于", lt: "早于", lte: "不晚于", range: "范围" };
function selectedFilters(): MetadataFilter[] {
  const result: MetadataFilter[] = [];
  const add = (field: MetadataFilter["field"], operator: MetadataFilter["operator"], value: string) => { if (value.trim()) result.push({ field, operator, value: value.trim() }); };
  add("department", "eq", filterDepartment.value); add("document_name", "contains", filterName.value);
  add("document_code", "contains", filterCode.value); add("file_type", "eq", filterFileType.value);
  add("business_status", "eq", filterStatus.value); add("uploaded_at", "gte", filterUploadFrom.value);
  add("uploaded_at", "lte", filterUploadTo.value); add("event_time", "gte", filterEventFrom.value); add("event_time", "lte", filterEventTo.value);
  return result;
}
const busy = ref(false);
const error = ref("");
const notice = ref("");
const searchResult = ref<QuerySearchResponse | null>(null);
const sessions = ref<ChatSession[]>([]);
const selectedSession = ref("");
const messages = ref<ChatRun[]>([]);
const sessionsLoading = ref(false);
const sessionsPageLoading = ref(false);
const sessionsHasMore = ref(false);
const sessionsOffset = ref(0);
const historyLoading = ref(false);
const olderLoading = ref(false);
const hasMore = ref(false);
const historyCursor = ref<string | null>(null);
const retryRequest = ref<{ sessionId: string; body: ChatRequest; runId?: string } | null>(null);
const elapsedByRun = ref<Record<string, number>>({});
const pending = computed(() => messages.value.some(isPendingRun));
const runLabels: Record<ChatRun["status"], string> = {
  PENDING: "等待执行", RUNNING: "正在生成回答", SUCCEEDED: "已保存", FAILED: "本轮执行失败",
  CANCELLED: "本轮已取消", ABANDONED: "本轮执行中断，请重新提问",
};
const mode = ref<"search" | "answer">("answer");
const progressMessage = ref("");
const progressLabels: Record<QueryAnswerStage, string> = {
  analysis: "正在分析问题…", retrieval: "正在检索资料…",
  generation: "正在根据资料生成回答…", review: "正在核对回答的原文依据…",
};
const answerLabels: Record<QueryAnswerResult["status"], string> = {
  answered: "回答", structured: "文档统计", insufficient: "资料不足", rejected: "本次回答未发布",
  unavailable: "回答暂不可用", skipped: "请补充问题", unsupported: "当前问题暂不支持",
};
const submittedScope = ref("");
const requestElapsedMs = ref(0);
let controller: AbortController | undefined;
let requestVersion = 0;
let pollController: AbortController | undefined;
let earlierController: AbortController | undefined;
let sessionsPageController: AbortController | undefined;
let pollTimer: ReturnType<typeof setTimeout> | undefined;
const canSearch = computed(() => !isDemoMode && !props.basesLoading && !props.basesError
  && props.bases.length > 0 && !!query.value.trim() && !busy.value && !pending.value && !historyLoading.value && !sessionsLoading.value);
const routeLabels: Record<QueryRoute, string> = {
  hybrid: "混合检索", hybrid_graph: "图增强混合检索", structured: "文档目录", clarify: "需要澄清", none: "无需检索",
};
const stageLabels: Record<string, string> = { succeeded: "成功", failed: "失败", skipped: "未执行", disabled: "未启用", empty: "无可用证据" };
const degradationLabels: Record<string, string> = {
  EMBEDDING_UNAVAILABLE: "问题向量化失败", DENSE_UNAVAILABLE: "向量检索不可用",
  BM25_UNAVAILABLE: "关键词检索不可用", RERANKER_UNAVAILABLE: "重排不可用，保留融合排序",
  SCOPE_CHANGED: "检索期间文档或权限发生变化",
  GRAPH_UNAVAILABLE: "图关系召回暂不可用，已使用基础混合检索",
  GRAPH_NO_EVIDENCE: "当前图谱未返回可用关系证据，已使用基础混合检索",
  GRAPH_PROVENANCE_CHANGED: "图谱在检索期间发生变化，已移除失效关系出处",
  GRAPH_PROVENANCE_UNAVAILABLE: "无法再次确认图关系出处，已保留当前授权原文证据",
};
function resultMessage(value: QuerySearchResponse) {
  if (value.plan.route === "clarify") return value.plan.params.question || value.plan.message || "请补充问题后重新检索。";
  if (value.plan.route === "none") return value.retrieval.message || "当前问题无需检索文档。";
  return value.retrieval.message || value.plan.message || (value.retrieval.status === "empty"
    ? (value.scope.published_document_count === 0 ? "当前范围内没有已发布的文档，请先上传文档并等待入库完成。" : "请尝试更具体的问题，或调整知识库范围。")
    : "");
}
const time = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(2)} 秒` : `${ms} 毫秒`;
const date = (value: string) => new Date(value).toLocaleString("zh-CN");
const panels = computed(() => [
  ...messages.value.map((run) => ({ id: run.id, run, result: run.response as QuerySearchResponse | QueryAnswerResponse | null,
    answer: run.response?.answer || null, scope: run.response ? `${run.response.scope.knowledge_base_ids.length} 个知识库` : "",
    elapsed: elapsedByRun.value[run.id] ?? null,
    paths: run.response ? graphPathViews(validatedGraphPaths(run.response), run.response.retrieval.evidence) : [] })),
  ...(searchResult.value ? [{ id: `search-${requestVersion}`, run: null, result: searchResult.value,
    answer: null, scope: submittedScope.value, elapsed: requestElapsedMs.value,
    paths: graphPathViews(validatedGraphPaths(searchResult.value), searchResult.value.retrieval.evidence) }] : []),
]);
function stopPolling() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = undefined;
  pollController?.abort(); pollController = undefined;
}
function upsertRun(run: ChatRun) {
  const existing = messages.value.findIndex((item) => item.id === run.id);
  if (existing >= 0) messages.value[existing] = run;
  else messages.value.push(run);
  messages.value.sort((left, right) => BigInt(left.id) < BigInt(right.id) ? -1 : 1);
}
function active(version: number, sessionId: string, signal: AbortSignal) {
  return version === requestVersion && selectedSession.value === sessionId && !signal.aborted;
}
function storageKey() {
  const identity = authState.value.identity;
  return identity ? chatSessionStorageKey(identity.tenant.id, identity.user.id) : null;
}
function rememberSession() {
  const key = storageKey();
  try { if (key) selectedSession.value ? localStorage.setItem(key, selectedSession.value) : localStorage.removeItem(key); } catch { /* Storage can be unavailable. */ }
}
function storedSession() {
  try { const key = storageKey(); return key ? localStorage.getItem(key) || "" : ""; } catch { return ""; }
}
function causeMessage(cause: unknown, fallback: string) {
  return cause instanceof Error && cause.name === "TimeoutError" ? "请求超时，请恢复本轮或刷新历史。"
    : cause instanceof Error ? cause.message : fallback;
}
function restoredRunNotice(run: ChatRun) {
  if (run.status === "SUCCEEDED") return run.response ? "已恢复服务器保存的回答。" : "本轮来源已不可用，原回答已隐藏。";
  return runLabels[run.status];
}
function appendSessions(items: ChatSession[]) {
  const next = new Map(sessions.value.map((session) => [session.id, session]));
  for (const item of items) next.set(item.id, item);
  sessions.value = [...next.values()];
}

function invalidate() {
  requestVersion++;
  controller?.abort();
  controller = undefined;
  earlierController?.abort(); earlierController = undefined;
  sessionsPageController?.abort(); sessionsPageController = undefined;
  busy.value = false;
  sessionsLoading.value = false;
  sessionsPageLoading.value = false;
  stopPolling();
  historyLoading.value = false; olderLoading.value = false;
  progressMessage.value = "";
  error.value = "";
}
async function cancel() {
  const sessionId = selectedSession.value;
  invalidate();
  notice.value = mode.value === "answer" ? "已停止本次连接，正在确认服务器的轮次状态。" : "已取消本次检索。";
  if (mode.value === "answer" && sessionId) await refreshHistory();
}
function reloadBases() {
  notice.value = "";
  emit("reloadBases");
}
// Draft typing and scope selection never discard saved messages or a retry's original body.
watch(selectedBase, () => { searchResult.value = null; }, { flush: "sync" });
watch(() => props.bases, (bases) => {
  if (selectedBase.value && !bases.some((base) => base.id === selectedBase.value)) {
    selectedBase.value = "";
    notice.value = "原知识库已不可用，请重新确认检索范围。";
  }
  // A refreshed readable scope can mean revoked sources. Remove old assistant text
  // immediately, then ask the server to authorize every restored response again.
  if (!isDemoMode && selectedSession.value) {
    invalidate(); messages.value = []; searchResult.value = null; hasMore.value = false;
    notice.value = "知识库范围已更新，正在按当前权限恢复历史。";
    void refreshHistory();
  }
}, { flush: "sync" });
onBeforeUnmount(invalidate);

function pollRun(run: ChatRun, version = requestVersion) {
  stopPolling();
  const sessionId = run.session_id;
  const request = new AbortController(); pollController = request;
  let attempts = 0;
  async function poll() {
    if (!active(version, sessionId, request.signal)) return;
    try {
      const current = await getChatRun(sessionId, run.id, request.signal);
      if (!active(version, sessionId, request.signal)) return;
      upsertRun(current);
      if (!isPendingRun(current)) {
        stopPolling();
        if (retryRequest.value?.body.client_request_id === current.client_request_id) retryRequest.value = null;
        notice.value = restoredRunNotice(current);
        return;
      }
      if (++attempts >= 40) {
        stopPolling(); notice.value = "本轮仍在处理，自动恢复已暂停。请稍后点击刷新历史确认结果。"; return;
      }
      pollTimer = setTimeout(poll, 3000);
    } catch (cause) {
      if (!active(version, sessionId, request.signal)) return;
      stopPolling(); error.value = causeMessage(cause, "无法恢复本轮，请刷新历史。");
      if (cause instanceof QuerySearchError && (cause.status === 403 || cause.status === 404)) messages.value = [];
    }
  }
  pollTimer = setTimeout(poll, 1500);
}
async function refreshHistory() {
  const sessionId = selectedSession.value;
  if (!sessionId || busy.value) return;
  stopPolling(); controller?.abort();
  const version = ++requestVersion, request = new AbortController(); controller = request;
  historyLoading.value = true; error.value = "";
  try {
    const history = await listChatRuns(sessionId, request.signal);
    if (!active(version, sessionId, request.signal)) return;
    messages.value = history.runs; hasMore.value = history.has_more; historyCursor.value = history.next_before_run_id;
    const current = history.runs.find(isPendingRun);
    const retried = history.runs.find((run) => run.client_request_id === retryRequest.value?.body.client_request_id);
    if (retried && !isPendingRun(retried)) { retryRequest.value = null; notice.value = restoredRunNotice(retried); }
    if (current) pollRun(current, version);
  } catch (cause) {
    if (!active(version, sessionId, request.signal)) return;
    messages.value = []; hasMore.value = false;
    error.value = causeMessage(cause, "会话历史加载失败，请重试。");
  } finally {
    if (version === requestVersion) { historyLoading.value = false; controller = undefined; }
  }
}
async function selectSession(sessionId: string) {
  invalidate(); selectedSession.value = sessionId; rememberSession();
  messages.value = []; searchResult.value = null; retryRequest.value = null; query.value = ""; notice.value = "";
  hasMore.value = false; historyCursor.value = null;
  await refreshHistory();
}
async function loadSessions() {
  if (isDemoMode || !authState.value.identity) return;
  invalidate(); sessionsLoading.value = true;
  const version = requestVersion, request = new AbortController(); controller = request;
  let loaded = false;
  const cached = selectedSession.value || storedSession();
  try {
    sessions.value = []; sessionsOffset.value = 0; sessionsHasMore.value = false;
    // Preserve a cached conversation even when it lies beyond the newest page.
    do {
      const items = await listChatSessions(request.signal, sessionsOffset.value);
      if (version !== requestVersion || request.signal.aborted) return;
      appendSessions(items);
      sessionsOffset.value += items.length;
      sessionsHasMore.value = items.length === CHAT_SESSIONS_PAGE_SIZE;
    } while (cached && !sessions.value.some((session) => session.id === cached) && sessionsHasMore.value);
    const sessionId = sessions.value.find((session) => session.id === cached)?.id || sessions.value[0]?.id || "";
    if (sessionId !== selectedSession.value) { messages.value = []; searchResult.value = null; retryRequest.value = null; query.value = ""; }
    selectedSession.value = sessionId; rememberSession();
    if (cached && sessionId !== cached) notice.value = "原会话已不可用，已恢复最近的可读会话。";
    loaded = true;
  } catch (cause) {
    if (version !== requestVersion || request.signal.aborted) return;
    error.value = causeMessage(cause, "会话列表加载失败，请重试。");
  } finally {
    if (version === requestVersion) { sessionsLoading.value = false; controller = undefined; }
  }
  if (loaded && version === requestVersion && selectedSession.value) await refreshHistory();
}
async function loadMoreSessions() {
  if (!sessionsHasMore.value || sessionsLoading.value || sessionsPageLoading.value || busy.value) return;
  const version = requestVersion, request = new AbortController(); sessionsPageController = request;
  sessionsPageLoading.value = true; error.value = "";
  try {
    const items = await listChatSessions(request.signal, sessionsOffset.value);
    if (version !== requestVersion || request.signal.aborted) return;
    appendSessions(items); sessionsOffset.value += items.length;
    sessionsHasMore.value = items.length === CHAT_SESSIONS_PAGE_SIZE;
  } catch (cause) {
    if (version === requestVersion && !request.signal.aborted) error.value = causeMessage(cause, "更早会话加载失败，请重试。");
  } finally {
    if (version === requestVersion) { sessionsPageLoading.value = false; sessionsPageController = undefined; }
  }
}
async function newSession() {
  if (isDemoMode || sessionsLoading.value) return;
  invalidate(); sessionsLoading.value = true;
  const version = requestVersion, request = new AbortController(); controller = request;
  try {
    const session = await createChatSession(request.signal);
    if (version !== requestVersion || request.signal.aborted) return;
    sessions.value.unshift(session);
    sessionsOffset.value++;
    sessionsLoading.value = false;
    await selectSession(session.id);
  } catch (cause) {
    if (version === requestVersion && !request.signal.aborted) error.value = causeMessage(cause, "新会话创建失败，请重试。");
  } finally {
    if (version === requestVersion) { sessionsLoading.value = false; controller = undefined; }
  }
}
async function loadEarlier() {
  if (!hasMore.value || !historyCursor.value || olderLoading.value || busy.value || historyLoading.value) return;
  const sessionId = selectedSession.value, version = requestVersion, request = new AbortController();
  earlierController = request;
  olderLoading.value = true;
  try {
    const history = await listChatRuns(sessionId, request.signal, historyCursor.value);
    if (!active(version, sessionId, request.signal)) return;
    for (const run of history.runs) upsertRun(run);
    hasMore.value = history.has_more; historyCursor.value = history.next_before_run_id;
  } catch (cause) {
    if (active(version, sessionId, request.signal)) error.value = causeMessage(cause, "更早历史加载失败，请重试。");
  } finally { if (version === requestVersion) { olderLoading.value = false; earlierController = undefined; } }
}
watch(() => authState.value.identity ? `${authState.value.identity.tenant.id}:${authState.value.identity.user.id}` : "", () => {
  invalidate(); sessions.value = []; messages.value = []; selectedSession.value = "";
  sessionsHasMore.value = false; sessionsOffset.value = 0;
  retryRequest.value = null; searchResult.value = null;
  void loadSessions();
}, { immediate: true });

async function recoverRetry(sendIfMissing = false) {
  const retry = retryRequest.value;
  if (!retry || retry.sessionId !== selectedSession.value || busy.value) return;
  if (retry.runId) {
    const version = requestVersion, request = new AbortController(); controller = request;
    historyLoading.value = true; error.value = "";
    try {
      const run = await getChatRun(retry.sessionId, retry.runId, request.signal);
      if (!active(version, retry.sessionId, request.signal)) return;
      upsertRun(run);
      if (isPendingRun(run)) pollRun(run, version);
      else { retryRequest.value = null; notice.value = restoredRunNotice(run); }
    } catch (cause) {
      if (!active(version, retry.sessionId, request.signal)) return;
      error.value = causeMessage(cause, "本轮恢复失败，请刷新历史。");
      if (cause instanceof QuerySearchError && (cause.status === 403 || cause.status === 404)) { messages.value = []; retryRequest.value = null; }
    } finally { if (version === requestVersion) { historyLoading.value = false; controller = undefined; } }
    return;
  }
  await refreshHistory();
  if (retryRequest.value !== retry || error.value || pending.value || retry.sessionId !== selectedSession.value) return;
  const existing = messages.value.find((run) => run.client_request_id === retry.body.client_request_id);
  if (existing) { retryRequest.value = null; return; }
  if (sendIfMissing) await sendAnswer(retry.body);
  else notice.value = "未查询到已保存的本轮。可点击恢复或重试，沿用原问题与请求标识。";
}
async function sendAnswer(body: ChatRequest) {
  invalidate(); notice.value = ""; busy.value = true; mode.value = "answer";
  const version = requestVersion, request = new AbortController(); controller = request;
  const started = performance.now();
  let sessionId = selectedSession.value;
  try {
    if (!sessionId) {
      const session = await createChatSession(request.signal, body.query.slice(0, 80));
      if (version !== requestVersion || request.signal.aborted) return;
      sessionId = session.id; selectedSession.value = sessionId; sessions.value.unshift(session); rememberSession();
      sessionsOffset.value++;
    }
    retryRequest.value = { sessionId, body };
    const run = await streamChatAnswer(sessionId, body, request.signal, (event) => {
      if (!active(version, sessionId, request.signal)) return;
      if (event.type === "progress") progressMessage.value = progressLabels[event.stage];
      else {
        if (retryRequest.value) retryRequest.value.runId = event.run.id;
        upsertRun(event.run);
        if (event.type === "answer") progressMessage.value = "正在展示已保存的回答…";
      }
    });
    if (!active(version, sessionId, request.signal)) return;
    upsertRun(run); elapsedByRun.value[run.id] = Math.max(0, Math.round(performance.now() - started));
    retryRequest.value = null;
    const session = sessions.value.find((item) => item.id === sessionId);
    if (session) { session.last_active_at = run.created_at; if (session.title === "新会话") session.title = body.query.slice(0, 80); }
    sessions.value.sort((left, right) => Date.parse(right.last_active_at) - Date.parse(left.last_active_at));
    if (run.status !== "SUCCEEDED") notice.value = restoredRunNotice(run);
    if (run.status === "SUCCEEDED" && query.value.trim() === body.query) query.value = "";
  } catch (cause) {
    if (!active(version, sessionId, request.signal) && (version !== requestVersion || request.signal.aborted)) return;
    error.value = causeMessage(cause, "回答失败，正在确认已保存的轮次。");
    if (cause instanceof QuerySearchError && cause.status === 403) emit("reloadBases");
  } finally {
    if (version === requestVersion) { busy.value = false; controller = undefined; }
  }
  if (version === requestVersion && retryRequest.value && sessionId) await recoverRetry();
}
async function submit(requestMode: "search" | "answer" = "answer") {
  if (!canSearch.value) return;
  if (requestMode === "answer") {
    try { await sendAnswer(createChatRequest(query.value, selectedBase.value, crypto.randomUUID(), { filters: selectedFilters() })); }
    catch (cause) { error.value = causeMessage(cause, "问题无效，请重试。"); }
    return;
  }
  invalidate();
  searchResult.value = null;
  notice.value = "";
  const version = requestVersion;
  const request = new AbortController();
  controller = request;
  busy.value = true;
  mode.value = requestMode;
  const started = performance.now();
  submittedScope.value = props.bases.find((base) => base.id === selectedBase.value)?.name || "全部可读知识库";
  try {
    const response = await searchQuery(query.value, selectedBase.value, request.signal, { filters: selectedFilters() });
    if (version === requestVersion && !request.signal.aborted) {
      requestElapsedMs.value = Math.max(0, Math.round(performance.now() - started));
      searchResult.value = response;
    }
  } catch (cause) {
    if (version !== requestVersion || request.signal.aborted) return;
    searchResult.value = null;
    error.value = cause instanceof Error && cause.name === "TimeoutError" ? "请求等待超过 2 分钟，请稍后重试。"
      : cause instanceof Error ? cause.message : "检索失败，请检查网络后重试。";
    if (cause instanceof QuerySearchError && cause.status === 403) emit("reloadBases");
  } finally {
    if (version === requestVersion) { busy.value = false; controller = undefined; }
  }
}
async function loadDocumentPage(result: QuerySearchResponse, offset: number) {
  if (busy.value || pending.value || historyLoading.value || sessionsLoading.value || !result.request) return;
  invalidate(); busy.value = true; mode.value = "search";
  const version = requestVersion, request = new AbortController(); controller = request;
  const started = performance.now();
  try {
    const page = await searchQueryPage(result.request, offset, request.signal);
    if (version !== requestVersion || request.signal.aborted) return;
    searchResult.value = page;
    submittedScope.value = `${page.scope.knowledge_base_ids.length} 个知识库`;
    requestElapsedMs.value = Math.round(performance.now() - started);
  } catch (cause) {
    if (version === requestVersion && !request.signal.aborted) error.value = causeMessage(cause, "清单加载失败，请重试。");
  } finally { if (version === requestVersion) { busy.value = false; controller = undefined; } }
}
</script>

<template>
  <div class="query-workspace">
    <div class="page-heading">
      <div><h1>智能问答 <span class="query-tag">多轮会话 · 附原文引用</span></h1><p>在同一会话继续追问，回答与原文依据会自动保存。刷新页面可恢复历史。</p></div>
    </div>
    <section v-if="!isDemoMode" class="card chat-session-card" aria-label="会话列表">
      <div class="query-result-heading"><h2>我的会话</h2><div class="query-actions">
        <button class="button secondary" type="button" :disabled="sessionsLoading || busy" @click="loadSessions"><RefreshCw :size="15" />刷新会话</button>
        <button class="button secondary" type="button" :disabled="sessionsLoading" @click="newSession"><Plus :size="15" />新会话</button>
      </div></div>
      <p v-if="sessionsLoading" class="query-hint" role="status">正在加载会话…</p>
      <div v-else-if="sessions.length" class="chat-session-list">
        <button v-for="session in sessions" :key="session.id" type="button" class="chat-session-option"
          :class="{ selected: session.id === selectedSession }" :aria-pressed="session.id === selectedSession" @click="selectSession(session.id)">
          <MessageSquare :size="15" /><span>{{ session.title }}</span>
        </button>
      </div>
      <button v-if="sessionsHasMore && !sessionsLoading" class="button secondary chat-more-sessions" type="button"
        :disabled="sessionsPageLoading || busy" @click="loadMoreSessions"><LoaderCircle v-if="sessionsPageLoading" :size="15" class="spin" />{{ sessionsPageLoading ? '正在加载更早会话…' : '加载更早会话' }}</button>
      <p v-else class="query-hint">输入第一个问题即可开始新会话。</p>
      <div v-if="selectedSession" class="chat-history-actions">
        <span class="query-hint">{{ messages.length }} 轮已加载 · 历史按当前资料权限恢复</span>
        <button type="button" class="button secondary" :disabled="busy || historyLoading" @click="refreshHistory"><RefreshCw :size="15" />刷新历史</button>
      </div>
      <p v-if="historyLoading" class="query-hint" role="status">正在恢复会话历史…</p>
    </section>
    <section class="card query-form-card" aria-labelledby="query-form-title">
      <h2 id="query-form-title">你的问题</h2>
      <p v-if="isDemoMode" class="query-message">当前为离线演示。请使用 API 模式并登录后测试真实检索。</p>
      <form @submit.prevent="submit()">
        <label for="query-scope">知识库范围</label>
        <div class="query-scope-row">
          <select id="query-scope" v-model="selectedBase" :disabled="isDemoMode || basesLoading || busy || !!basesError">
            <option value="">全部可读知识库</option>
            <option v-for="base in bases" :key="base.id" :value="base.id">{{ base.name }}</option>
          </select>
          <button type="button" class="button secondary" :disabled="isDemoMode || basesLoading || busy" @click="reloadBases"><RefreshCw :size="15" />刷新知识库</button>
        </div>
        <p v-if="basesLoading" class="query-hint" role="status">正在加载可读知识库…</p>
        <p v-else-if="basesError" class="query-error" role="alert">{{ basesError }}</p>
        <p v-else-if="!isDemoMode && !bases.length" class="query-hint">当前没有可读知识库，请联系管理员授权。</p>
        <details class="query-details">
          <summary>筛选文档</summary>
          <div class="metadata-filter-grid">
            <label>归属部门<input v-model="filterDepartment" :disabled="busy" maxlength="200" placeholder="准确部门名称或代码" /></label>
            <label>文件名包含<input v-model="filterName" :disabled="busy" maxlength="200" /></label>
            <label>文档编号包含<input v-model="filterCode" :disabled="busy" maxlength="100" /></label>
            <label>文件类型<select v-model="filterFileType" :disabled="busy"><option value="">全部类型</option><option v-for="type in ['PDF', 'DOCX', 'XLSX', 'XLS', 'TXT', 'Markdown']" :key="type" :value="type">{{ type }}</option></select></label>
            <label>业务状态<select v-model="filterStatus" :disabled="busy"><option value="">全部状态</option><option v-for="(label, value) in businessStatusLabels" :key="value" :value="value">{{ label }}</option></select></label>
            <label>发布版本上传起始日期<input v-model="filterUploadFrom" :disabled="busy" type="date" /></label>
            <label>发布版本上传结束日期<input v-model="filterUploadTo" :disabled="busy" type="date" /></label>
            <label>业务起始日期<input v-model="filterEventFrom" :disabled="busy" type="date" /></label>
            <label>业务结束日期<input v-model="filterEventTo" :disabled="busy" type="date" /></label>
          </div>
          <p class="query-hint">筛选与问题中的条件共同生效，日期包含当天。可提问“文档有多少份？”或“列出所有文档”。</p>
        </details>
        <label for="query-text">你的问题</label>
        <textarea id="query-text" v-model="query" maxlength="2000" rows="4" placeholder="例如：工单关闭前需要谁确认？" @keydown.ctrl.enter.prevent="submit()" @keydown.meta.enter.prevent="submit()"></textarea>
        <div class="query-form-footer">
          <span class="query-hint">{{ query.length }}/2000 · Ctrl / ⌘ + Enter 提问</span>
          <div class="query-actions">
            <button v-if="busy" type="button" class="button secondary" @click="cancel"><X :size="15" />取消</button>
            <button type="button" class="button secondary" :disabled="!canSearch" @click="submit('search')">仅检索证据</button>
            <button type="submit" class="button primary" :disabled="!canSearch"><LoaderCircle v-if="busy" :size="17" class="spin" /><Search v-else :size="17" />{{ busy ? (mode === 'answer' ? '正在生成回答…' : '正在检索…') : '生成回答' }}</button>
          </div>
        </div>
      </form>
      <p v-if="busy" class="query-hint" role="status" aria-live="polite">{{ mode === 'answer' ? (progressMessage || '正在连接问答服务…') : '正在分析问题并检索文档，请稍候。' }}</p>
      <p v-if="error" class="query-error" role="alert">{{ error }}</p>
      <p v-if="notice" class="query-message" role="status">{{ notice }}</p>
      <p v-if="pending && !busy" class="query-message" role="status">本轮仍在执行，正在查询服务器状态，完成后可继续追问。</p>
      <button v-if="retryRequest && !busy && !pending" type="button" class="button secondary" :disabled="historyLoading" @click="recoverRetry(true)">恢复或重试本轮</button>
    </section>

    <div v-if="hasMore" class="chat-earlier"><button type="button" class="button secondary" :disabled="olderLoading || busy || historyLoading" @click="loadEarlier"><LoaderCircle v-if="olderLoading" :size="15" class="spin" />{{ olderLoading ? '正在加载…' : '加载更早的历史' }}</button></div>
    <section v-for="panel in panels" :key="panel.id" class="query-results" :aria-labelledby="`query-results-title-${panel.id}`">
      <article v-if="panel.run" class="card chat-question">
        <div class="query-result-heading"><h2 :id="`query-results-title-${panel.id}`">{{ panel.run.question }}</h2><span class="query-tag">{{ runLabels[panel.run.status] }}</span></div>
        <p class="query-hint">{{ date(panel.run.created_at) }}</p>
        <p v-if="!panel.result && (panel.run.visibility === 'unavailable' || (!isPendingRun(panel.run) && panel.run.status !== 'SUCCEEDED'))" class="query-message">{{ panel.run.status === 'SUCCEEDED' ? '本轮回答的来源已不可用或当前账号无权查看，原回答与引用已隐藏。' : runLabels[panel.run.status] }}<span v-if="panel.run.error_code">（{{ panel.run.error_code }}）</span></p>
        <p v-if="panel.run.visibility === 'pending'" class="query-hint">本轮尚未保存最终回答，正在恢复执行状态。</p>
        <details v-if="panel.run.standalone_question && panel.run.standalone_question !== panel.run.question" class="query-details"><summary>查看本轮独立问题</summary><p>{{ panel.run.standalone_question }}</p></details>
      </article>
      <template v-if="panel.result">
      <article v-if="panel.result.structured" class="card structured-result" aria-label="文档目录结果">
        <h2>符合条件的文档 <strong>{{ panel.result.structured.total }}</strong> 份</h2>
        <p class="query-hint">当前账号可读 · 已发布且活动索引就绪 · 查询于 {{ date(panel.result.structured.executed_at) }}</p>
        <p v-for="(filter, index) in panel.result.structured.applied_filters" :key="index" class="query-hint">{{ filterLabels[filter.field] || filter.field }} · {{ operatorLabels[filter.operator] || filter.operator }} · {{ filter.values.join(' / ') }}</p>
        <div v-if="panel.result.structured.operation === 'list_documents'" class="structured-table-wrap">
          <table class="structured-table"><thead><tr><th>文档</th><th>文件名</th><th>业务状态</th><th>业务日期</th><th>发布版本上传时间</th></tr></thead>
            <tbody><tr v-for="doc in panel.result.structured.documents" :key="doc.document_id"><td>{{ doc.title }}</td><td>{{ doc.document_name }}</td><td>{{ businessStatusLabels[doc.business_status] || doc.business_status }}</td><td>{{ doc.event_date || '未填写' }}</td><td>{{ date(doc.uploaded_at) }}</td></tr></tbody>
          </table>
          <p v-if="!panel.result.structured.documents.length" class="query-message">本页没有符合条件的文档。</p>
          <div class="query-actions"><button type="button" class="button secondary" :disabled="busy || pending || historyLoading || sessionsLoading || !panel.result.request || panel.result.structured.offset === 0" @click="loadDocumentPage(panel.result, Math.max(0, panel.result.structured.offset - panel.result.structured.limit))">上一页</button>
            <span class="query-hint">{{ panel.result.structured.total ? panel.result.structured.offset + 1 : 0 }}—{{ panel.result.structured.offset + panel.result.structured.documents.length }} / {{ panel.result.structured.total }}</span>
            <button type="button" class="button secondary" :disabled="busy || pending || historyLoading || sessionsLoading || !panel.result.request || panel.result.structured.offset + panel.result.structured.limit >= panel.result.structured.total" @click="loadDocumentPage(panel.result, panel.result.structured.offset + panel.result.structured.limit)">下一页</button></div>
        </div>
      </article>
      <article v-if="panel.answer" class="card query-answer" :aria-labelledby="`query-answer-title-${panel.id}`">
        <h2 :id="`query-answer-title-${panel.id}`">{{ answerLabels[panel.answer.status] }}</h2>
        <p v-if="panel.answer.message" class="query-message">{{ panel.answer.message }}</p>
        <div v-for="(statement, index) in panel.answer.statements" :key="index" class="answer-statement">
          <p>{{ statement.text }} <a v-for="citation in statement.citations" :key="citation.citation_id + citation.quote" :href="`#query-evidence-${panel.id}-${citation.citation_id}`" class="answer-reference">[{{ citation.citation_id }}]</a></p>
          <details class="answer-quotes"><summary>查看这段回答的原文依据</summary><blockquote v-for="citation in statement.citations" :key="citation.citation_id + citation.quote"><span class="query-citation">{{ citation.citation_id }}</span> {{ citation.quote }}</blockquote></details>
        </div>
        <p v-if="panel.answer.status === 'skipped' && resultMessage(panel.result)" class="query-message">{{ resultMessage(panel.result) }}</p>
        <p class="query-hint">回答生成 {{ time(panel.answer.trace.generation_ms) }} · 证据复核 {{ time(panel.answer.trace.review_ms) }}</p>
      </article>
      <div class="card query-summary">
        <div class="query-result-heading"><h2 :id="panel.run ? undefined : `query-results-title-${panel.id}`">检索结果</h2><span class="query-tag" :class="{ warning: queryStatus(panel.result).warning }">{{ queryStatus(panel.result).label }}</span></div>
        <p class="query-hint">{{ panel.scope }} · {{ routeLabels[panel.result.plan.route] }} · {{ panel.result.retrieval.evidence.length }} 条原文证据</p>
        <div class="query-timings"><span v-if="panel.elapsed !== null">本次等待 <strong>{{ time(panel.elapsed) }}</strong></span><span>问题分析 <strong>{{ time(panel.result.analysis.latency_ms) }}</strong></span><span>检索执行 <strong>{{ time(panel.result.retrieval.trace.total_ms) }}</strong></span></div>
        <p v-if="slowestQueryStage(panel.result)" class="query-hint">耗时最多的已记录阶段：{{ slowestQueryStage(panel.result)?.label }}（{{ time(slowestQueryStage(panel.result)!.ms) }}）。</p>
        <p v-if="resultMessage(panel.result)" class="query-message">{{ resultMessage(panel.result) }}</p>
        <div v-if="panel.result.analysis.status === 'degraded'" class="query-message warning"><strong>问题分析已降级：{{ analysisFailureLabel(panel.result.analysis.degradation_reason) }}</strong><p v-if="panel.result.analysis.message">{{ panel.result.analysis.message }}</p></div>
        <p v-for="code in panel.result.retrieval.degradation_codes" :key="code" class="query-message warning">{{ degradationLabels[code] || code }}</p>
        <p v-if="panel.result.plan.requested_route !== panel.result.plan.route" class="query-message warning">计划使用{{ routeLabels[panel.result.plan.requested_route] }}，本次实际使用{{ routeLabels[panel.result.plan.route] }}。</p>
        <details class="query-details">
          <summary>查看检索详情</summary>
          <dl class="query-trace">
            <div><dt>问题分析状态</dt><dd>{{ panel.result.analysis.status === 'ok' ? '成功' : '已降级' }} · {{ time(panel.result.analysis.latency_ms) }}</dd></div>
            <div v-if="panel.result.analysis.status === 'degraded'"><dt>分析降级原因码</dt><dd>{{ panel.result.analysis.degradation_reason || '未提供' }}</dd></div>
            <div v-if="panel.result.plan.params.candidate_k !== undefined"><dt>计划候选 / 最终上限</dt><dd>{{ panel.result.plan.params.candidate_k }} / {{ panel.result.plan.params.top_k }} · {{ panel.result.plan.params.rerank ? '启用重排' : '关闭重排' }}</dd></div>
            <div><dt>可读 / 已发布文档</dt><dd>{{ panel.result.scope.authorized_document_count }} / {{ panel.result.scope.published_document_count }}</dd></div>
            <div><dt>向量检索</dt><dd>{{ stageLabels[panel.result.retrieval.trace.dense_status] }} · {{ panel.result.retrieval.trace.dense_count }} 条 · {{ time(panel.result.retrieval.trace.dense_ms) }}</dd></div>
            <div><dt>关键词检索</dt><dd>{{ stageLabels[panel.result.retrieval.trace.bm25_status] }} · {{ panel.result.retrieval.trace.bm25_count }} 条 · {{ time(panel.result.retrieval.trace.bm25_ms) }}</dd></div>
            <div v-if="panel.result.retrieval.trace.graph_status && panel.result.retrieval.trace.graph_status !== 'skipped'"><dt>图关系召回</dt><dd>{{ stageLabels[panel.result.retrieval.trace.graph_status] }} · {{ panel.result.retrieval.trace.graph_count }} 条原文 · {{ panel.result.retrieval.trace.graph_hops }} 跳 · {{ time(panel.result.retrieval.trace.graph_ms || 0) }}</dd></div>
            <div v-if="panel.result.retrieval.trace.graph_path_count !== undefined"><dt>完整关联路径</dt><dd>{{ panel.result.retrieval.trace.graph_path_count }} 条</dd></div>
            <div><dt>融合 / 授权后候选</dt><dd>{{ panel.result.retrieval.trace.fused_count }} / {{ panel.result.retrieval.trace.authorized_count }}</dd></div>
            <div><dt>重排</dt><dd>{{ stageLabels[panel.result.retrieval.trace.rerank_status] }} · {{ time(panel.result.retrieval.trace.rerank_ms) }}</dd></div>
            <div><dt>问题向量化</dt><dd>{{ time(panel.result.retrieval.trace.embedding_ms) }}</dd></div>
            <div><dt>融合 / 授权校验耗时</dt><dd>{{ time(panel.result.retrieval.trace.fusion_ms) }} / {{ time(panel.result.retrieval.trace.authorization_ms) }}</dd></div>
            <div><dt>路由原因</dt><dd>{{ panel.result.plan.reason_code }}</dd></div>
          </dl>
          <section v-if="panel.paths.length" class="query-graph-paths" aria-label="原文支持的关联路径">
            <h3>关联路径</h3>
            <p class="query-hint">每条路径展示两条关系及其原文出处。适用条件分别保留，可点击引用核对。</p>
            <article v-for="(path, pathIndex) in panel.paths" :key="path.id" class="query-graph-path">
              <h4>路径 {{ pathIndex + 1 }}</h4>
              <ol>
                <li v-for="(step, stepIndex) in path.steps" :key="stepIndex">
                  <p class="query-path-fact"><strong>{{ step.subject }}</strong><span class="query-path-predicate">{{ step.predicate }}</span><strong>{{ step.object }}</strong></p>
                  <p v-if="step.condition" class="query-path-condition">适用条件：{{ step.condition }}</p>
                  <blockquote>{{ step.quote }}</blockquote>
                  <a :href="`#query-evidence-${panel.id}-${step.citation}`" class="answer-reference" :title="step.document">[{{ step.citation }}] 查看原文</a>
                  <aside v-if="stepIndex === 0 && path.bridges.length" class="query-path-bridge">
                    <div v-for="(bridge, bridgeIndex) in path.bridges" :key="bridgeIndex">
                      <p>对象对应：{{ bridge.name }} · {{ bridge.basis }}</p>
                      <div v-for="(source, sourceIndex) in bridge.sources" :key="sourceIndex">
                        <blockquote>{{ source.quote }}</blockquote>
                        <a :href="`#query-evidence-${panel.id}-${source.citation}`" class="answer-reference" :title="source.document">[{{ source.citation }}] 核对对应依据</a>
                      </div>
                    </div>
                  </aside>
                </li>
              </ol>
            </article>
          </section>
          <p v-if="panel.result.analysis.analysis" class="query-hint">问题意图：{{ panel.result.analysis.analysis.intent }}</p>
          <p v-for="warning in panel.result.plan.warnings" :key="warning" class="query-hint">{{ warning }}</p>
          <p class="query-hint">本次等待包括网络、身份与范围校验、分析、检索及本次执行的回答生成和复核。召回分支并行执行，各阶段耗时不能直接相加；最耗时阶段用于定位等待位置。</p>
        </details>
      </div>
      <article v-for="item in panel.result.retrieval.evidence" :id="`query-evidence-${panel.id}-${item.citation_id}`" :key="item.chunk_id" class="card query-evidence" tabindex="-1">
        <div class="query-evidence-heading"><span class="query-citation">{{ item.citation_id }}</span><FileText :size="18" /><h3>{{ item.document_title || item.document_name }}</h3><span class="query-hint">第 {{ item.rank }} 条</span></div>
        <p class="query-source">{{ item.document_name }}<span v-if="item.page_number !== null"> · 第 {{ item.page_number }} 页</span><span v-if="item.section_title"> · {{ item.section_title }}</span> · 分块 {{ item.chunk_index + 1 }}</p>
        <div class="query-original">{{ item.content }}</div>
      </article>
      </template>
    </section>
    <div v-if="!panels.length && !busy && !error && !historyLoading" class="query-empty"><Search :size="30" /><p>会话中的问题与回答会显示在这里</p><span>建议先用已入库文档中的具体问题进行测试。</span></div>
    <footer class="page-footer"><span>知擎 · 智能问答</span><span><ShieldCheck :size="13" />按当前账号权限检索已发布文档</span></footer>
  </div>
</template>

<style scoped>
.query-workspace { max-width: 1120px; margin: 0 auto; }
.chat-session-card { padding: 20px 24px; margin-bottom: 20px; }
.chat-session-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; max-height: 180px; overflow-y: auto; }
.chat-session-option { display: flex; align-items: center; gap: 8px; max-width: 260px; padding: 10px 12px; border: 1px solid #dce3df; border-radius: 8px; color: #53675c; background: #fff; font: inherit; font-size: 13px; cursor: pointer; }
.chat-session-option span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chat-session-option.selected { border-color: #167c70; background: #e8f5ef; color: #167c70; }
.chat-session-option:focus-visible { outline: 2px solid #167c70; outline-offset: 2px; }
.chat-more-sessions { margin-top: 12px; }
.chat-history-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 16px; flex-wrap: wrap; }
.chat-earlier { display: flex; justify-content: center; margin-top: 24px; }
.chat-question { padding: 20px 24px; background: #f2f7f4; }
.chat-question h2 { margin: 0; font-size: 16px; line-height: 1.7; white-space: pre-wrap; overflow-wrap: anywhere; }
.chat-question .query-hint { margin-bottom: 0; }
.query-form-card, .query-summary, .query-evidence { padding: 24px; }
.query-answer { padding: 24px; }
.answer-statement { line-height: 1.8; margin-bottom: 18px; overflow-wrap: anywhere; }
.answer-statement p { white-space: pre-wrap; }
.answer-reference { color: #167c70; font-weight: 600; text-decoration: none; padding: 2px 4px; }
.answer-reference:focus-visible { outline: 2px solid #167c70; }
.answer-quotes { font-size: 13px; color: #52645c; }
.answer-quotes summary { cursor: pointer; }
.answer-quotes blockquote { margin: 12px 0; border-left: 3px solid #c9ded4; padding: 8px 14px; background: #f7faf8; white-space: pre-wrap; }
.query-evidence:target { outline: 2px solid #167c70; scroll-margin-top: 20px; }
h2 { font-size: 17px; margin: 0 0 18px; }
.query-tag { display: inline-flex; align-items: center; vertical-align: middle; font-size: 12px; font-weight: 500; padding: 5px 9px; border-radius: 6px; background: #e8f5ef; color: #167c70; }
h1 .query-tag { margin-left: 8px; }
.query-form-card label { display: block; font-size: 14px; font-weight: 600; margin: 18px 0 9px; }
.query-scope-row, .query-actions, .query-form-footer, .query-result-heading, .query-evidence-heading { display: flex; align-items: center; gap: 12px; }
.query-scope-row select, textarea { border: 1px solid #dce3df; border-radius: 8px; padding: 12px; background: #fff; color: #243c34; font: inherit; }
.query-scope-row select { flex: 1; min-width: 0; }
textarea { display: block; width: 100%; resize: vertical; min-height: 120px; box-sizing: border-box; line-height: 1.7; }
textarea:focus, select:focus { outline: 2px solid #167c70; outline-offset: 2px; }
.query-form-footer { justify-content: space-between; margin-top: 14px; flex-wrap: wrap; }
.query-hint, .query-source { color: #738079; font-size: 13px; line-height: 1.7; }
.query-error { color: #b33737; font-size: 14px; line-height: 1.7; overflow-wrap: anywhere; }
.query-message { margin: 10px 0; padding: 12px 14px; background: #f3f6f4; border-radius: 8px; font-size: 14px; line-height: 1.7; overflow-wrap: anywhere; }
.query-message p { margin-top: 6px; }
.warning { color: #925b17; background: #fff4df; }
.query-results { display: grid; gap: 16px; margin-top: 24px; }
.query-result-heading { justify-content: space-between; flex-wrap: wrap; }
.query-result-heading h2 { margin: 0; }
.query-timings { display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: #738079; margin: 18px 0; }
.query-timings strong { color: #243c34; margin-left: 8px; }
.query-details { border-top: 1px solid #e8ede9; padding-top: 16px; margin-top: 18px; font-size: 13px; }
.query-details summary { cursor: pointer; color: #167c70; }
.query-trace { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; }
.query-trace dt { color: #738079; margin-bottom: 5px; }
.query-trace dd { margin: 0; overflow-wrap: anywhere; }
.query-graph-paths { margin: 20px 0; }
.query-graph-paths h3 { font-size: 14px; margin: 0 0 8px; }
.query-graph-path { border: 1px solid #dce3df; border-radius: 8px; padding: 16px; margin-top: 12px; }
.query-graph-path h4 { margin: 0 0 12px; font-size: 13px; color: #53675c; }
.query-graph-path ol { margin: 0; padding-left: 22px; }
.query-graph-path li + li { margin-top: 18px; }
.query-path-fact { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px; line-height: 1.7; overflow-wrap: anywhere; }
.query-path-predicate { color: #167c70; }
.query-path-condition { background: #fff4df; color: #925b17; border-radius: 6px; padding: 8px 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
.query-graph-path blockquote { margin: 10px 0; padding: 8px 12px; background: #f7faf8; border-left: 3px solid #c9ded4; white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.8; }
.query-path-bridge { border: 1px dashed #c9ded4; border-radius: 6px; padding: 10px 12px; margin-top: 14px; }
.query-path-bridge p { margin-top: 0; }
.query-citation { font-size: 12px; color: #167c70; background: #e8f5ef; border-radius: 5px; padding: 5px 8px; }
.query-evidence-heading { color: #167c70; flex-wrap: wrap; }
.query-evidence-heading h3 { color: #243c34; margin: 0; font-size: 15px; flex: 1; min-width: 100px; overflow-wrap: anywhere; }
.query-source { overflow-wrap: anywhere; }
.query-original { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.85; font-size: 14px; border-top: 1px solid #e8ede9; padding-top: 16px; }
.metadata-filter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 14px; }
.metadata-filter-grid label { display: grid; gap: 6px; font-size: 13px; }
.metadata-filter-grid input, .metadata-filter-grid select { width: 100%; box-sizing: border-box; border: 1px solid #d9e3dd; border-radius: 8px; padding: 9px 10px; background: white; }
.structured-result h2 strong { color: #167c70; font-size: 28px; }
.structured-table-wrap { overflow-x: auto; }
.structured-table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 13px; }
.structured-table th, .structured-table td { text-align: left; padding: 10px; border-bottom: 1px solid #e5ece7; }
.structured-table th { white-space: nowrap; color: #52645c; }
.query-empty { text-align: center; color: #8b9790; padding: 60px 16px; }
.query-empty p { color: #53675c; font-size: 15px; }
.query-empty span { font-size: 13px; }
@media (max-width: 640px) {
  .query-form-card, .query-summary, .query-evidence { padding: 18px; }
  .query-scope-row { flex-wrap: wrap; }
  .query-scope-row select { flex-basis: 100%; }
  .query-trace { grid-template-columns: 1fr; }
  .query-form-footer .query-actions { width: 100%; justify-content: flex-end; }
  .query-actions { flex-wrap: wrap; }
}
</style>
