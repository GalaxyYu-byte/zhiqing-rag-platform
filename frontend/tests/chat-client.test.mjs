import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import { compileScript, parse } from "@vue/compiler-sfc";
import { createRenderer, nextTick } from "vue";
import { queryFixture } from "./helpers/query-fixture.mjs";

const moduleUrl = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText).toString("base64")}`;
const authUrl = moduleUrl(await readFile(new URL("../src/services/auth-client.ts", import.meta.url), "utf8"));
const { AuthClient } = await import(authUrl);
const dependencyUrl = moduleUrl(`export let client; export function setClient(value) { client = value; }
  export const authClient = { clearSession: (notice) => client.clearSession(notice) };
  export const authenticatedFetch = (url, options) => client.fetch(url, options);`);
const { setClient } = await import(dependencyUrl);
const prepare = (source) => source.replace('"./auth"', JSON.stringify(dependencyUrl))
  .replace('"./auth-client"', JSON.stringify(authUrl)).replaceAll("import.meta.env", "({})");
const queryUrl = moduleUrl(prepare(await readFile(new URL("../src/services/query.ts", import.meta.url), "utf8")));
const chatUrl = moduleUrl(prepare(await readFile(new URL("../src/services/chat.ts", import.meta.url), "utf8"))
  .replace('"./query"', JSON.stringify(queryUrl)));
const chat = await import(chatUrl);
const { structuredAnswerText } = await import(queryUrl);
const { listChatSessions, createChatSession, listChatRuns, getChatRun, createChatRequest, answerChat, streamChatAnswer, chatSessionStorageKey } = chat;
const json = (body, status = 200) => new Response(JSON.stringify(body), { status });
const login = (token = "token") => ({ access_token: token, user: { id: "1", email: "test@example.com", display_name: "测试" }, tenant: { id: "10", name: "测试租户", code: "TEST", role: "MEMBER" } });
const session = { id: "9007199254740993", title: "工单问题", status: "ACTIVE", created_at: "2026-10-10T00:00:00Z", last_active_at: "2026-10-10T00:00:00Z" };
const request = createChatRequest("  如何关闭？  ", "42", "stable-request");
function answerFixture() {
  return { ...queryFixture(), answer: {
    status: "answered", statements: [{ text: "关闭前需由负责人确认。", citations: [{ citation_id: "E1", quote: "工单关闭前需由负责人确认。" }] }],
    message: null, codes: [], model: "fake", review_status: "passed",
    trace: { generation_ms: 20, review_ms: 10, authorization_ms: 2, total_ms: 32, context_tokens: 100, input_tokens: 150, output_tokens: 40 },
  } };
}
const runFixture = (overrides = {}) => ({ id: "9007199254740994", session_id: session.id, client_request_id: request.client_request_id,
  question: request.query, standalone_question: "工单关闭前需要谁确认？", status: "SUCCEEDED", created_at: session.created_at,
  finished_at: "2026-10-10T00:00:01Z", error_code: null, visibility: "visible", response: answerFixture(), ...overrides });
const pendingFixture = () => runFixture({ status: "RUNNING", finished_at: null, visibility: "pending", response: null });
const envelope = (response = answerFixture(), overrides = {}) => ({ run: runFixture({ response: null, ...overrides }), response });
async function setup(handler) {
  const calls = [];
  const client = new AuthClient({ fetcher: async (url, init) => {
    if (url.endsWith("/select-tenant")) return json(login());
    calls.push({ url, init }); return handler(url, init);
  } });
  await client.selectTenant("selection", "10"); setClient(client);
  return { calls, client, signal: new AbortController().signal };
}
const frame = (event, data, newline = "\r\n") => `event: ${event}${newline}${JSON.stringify(data, null, 2)
  .split("\n").map((line) => `data: ${line}`).join(newline)}${newline}${newline}`;
const stream = (text, chunkSize = 1024) => {
  const bytes = new TextEncoder().encode(text); let offset = 0;
  return new Response(new ReadableStream({ pull(controller) {
    if (offset >= bytes.length) { controller.close(); return; }
    controller.enqueue(bytes.slice(offset, offset + chunkSize)); offset += chunkSize;
  } }), { headers: { "Content-Type": "text/event-stream; charset=utf-8" } });
};

test("persistent statistical answers and zero results survive stream and history validation", async () => {
  const value = answerFixture();
  value.plan.route = "structured"; value.plan.requested_route = "structured";
  value.retrieval.evidence = []; value.retrieval.status = "skipped";
  value.structured = { operation: "count_documents", total: 0, documents: [], offset: 0, limit: 20,
    applied_filters: [], scope: "authorized_published_documents", executed_at: "2026-10-11T00:00:00Z" };
  value.answer = { ...value.answer, status: "structured", statements: [], model: null,
    message: structuredAnswerText(value.structured) };
  const saved = runFixture({ response: value });
  const { signal } = await setup(() => stream(frame("run", pendingFixture())
    + frame("answer", envelope(value)) + frame("done", envelope(value)), 5));
  const events = [];
  const result = await streamChatAnswer(session.id, request, signal, (event) => events.push(event));
  assert.deepEqual(result, saved);
  assert.equal(events.filter((event) => event.type === "answer").length, 1);
  const history = await setup(() => json({ runs: [saved], has_more: false, next_before_run_id: null }));
  assert.equal((await listChatRuns(session.id, history.signal)).runs[0].response.structured.total, 0);
});

test("chat requests preserve explicit filters and page options on retries", () => {
  const filters = [{ field: "business_status", operator: "eq", value: "EFFECTIVE" }];
  assert.deepEqual(createChatRequest("文档有多少份？", "42", "same-request", { filters, offset: 0, limit: 5 }),
    { query: "文档有多少份？", knowledge_base_ids: ["42"], client_request_id: "same-request", filters, offset: 0, limit: 5 });
});

test("session list, creation and paginated history use authenticated server data and precision-safe IDs", async () => {
  const { calls, signal } = await setup((url) => url.includes("/runs")
    ? json({ runs: [runFixture()], has_more: false, next_before_run_id: null })
    : calls.length === 1 ? json({ sessions: [session] }) : json(session));
  assert.deepEqual(await listChatSessions(signal), [session]);
  assert.deepEqual(await createChatSession(signal, "工单问题"), session);
  assert.equal((await listChatRuns(session.id, signal, "9007199254740995")).runs[0].id, "9007199254740994");
  assert.equal(calls[2].url, `/api/chat/sessions/${session.id}/runs?limit=20&before_run_id=9007199254740995`);
  assert.deepEqual(JSON.parse(calls[1].init.body), { title: "工单问题" });
  assert.equal(calls[0].init.headers.get("Authorization"), "Bearer token");
});

test("session pagination sends a nonnegative offset and rejects malformed pages without issuing unsafe requests", async () => {
  const { calls, signal } = await setup(() => json({ sessions: [session] }));
  await listChatSessions(signal, 50);
  assert.equal(calls[0].url, "/api/chat/sessions?offset=50");
  for (const offset of [-1, 1.5, NaN, Number.MAX_SAFE_INTEGER + 1]) await assert.rejects(listChatSessions(signal, offset), /分页位置无效/);
  assert.equal(calls.length, 1);
  const broken = await setup(() => json({ sessions: [session, session] }));
  await assert.rejects(listChatSessions(broken.signal), /响应不完整/);
});

test("persistent answer token refresh preserves exactly the original idempotent request body", async () => {
  let answers = 0;
  const { calls, signal } = await setup((url) => url.endsWith("/refresh") ? json(login("new-token"))
    : ++answers === 1 ? json({ detail: { code: "ACCESS_TOKEN_EXPIRED" } }, 401) : json(runFixture()));
  const run = await answerChat(session.id, request, signal);
  const attempts = calls.filter(({ url }) => url.endsWith("/answer"));
  assert.equal(run.client_request_id, "stable-request");
  assert.equal(attempts.length, 2); assert.equal(attempts[0].init.body, attempts[1].init.body);
  assert.deepEqual(JSON.parse(attempts[0].init.body), { query: "如何关闭？", client_request_id: "stable-request", knowledge_base_ids: ["42"] });
});

test("new questions get fresh request IDs while explicit retries can reuse the entire original request", () => {
  const first = createChatRequest("第一问", "42"), second = createChatRequest("第二问", "42");
  assert.notEqual(first.client_request_id, second.client_request_id);
  assert.equal(createChatRequest("第一问", "42", first.client_request_id).client_request_id, first.client_request_id);
  assert.throws(() => createChatRequest(" "), /1 到 2000/);
  assert.throws(() => createChatRequest("字".repeat(2001)), /1 到 2000/);
});

test("current-session storage keys are isolated by both tenant and user", () => {
  assert.notEqual(chatSessionStorageKey("10", "1"), chatSessionStorageKey("11", "1"));
  assert.notEqual(chatSessionStorageKey("10", "1"), chatSessionStorageKey("10", "2"));
});

test("history source revocation and terminal failures hide all previous assistant text", async () => {
  for (const status of ["SUCCEEDED", "FAILED", "CANCELLED", "ABANDONED"]) {
    const hidden = runFixture({ status, visibility: "unavailable", response: null, error_code: "CHAT_SOURCE_UNAVAILABLE" });
    const { signal } = await setup(() => json(hidden));
    assert.equal((await getChatRun(session.id, hidden.id, signal)).response, null);
  }
  const { signal } = await setup(() => json(runFixture({ visibility: "unavailable" })));
  await assert.rejects(getChatRun(session.id, "9007199254740994", signal), /响应不完整/);
});

test("server terminal failure payloads remain visible with no response in history and replay streams", async () => {
  for (const status of ["FAILED", "CANCELLED", "ABANDONED"]) {
    const terminal = runFixture({ status, visibility: "visible", response: null, error_code: "CHAT_EXECUTION_FAILED", standalone_question: null });
    const { signal } = await setup((url) => url.endsWith("/stream")
      ? stream(frame("run", terminal) + frame("done", { run: terminal, response: null }))
      : json({ runs: [terminal], has_more: false, next_before_run_id: null }));
    const history = await listChatRuns(session.id, signal);
    assert.equal(history.runs[0].visibility, "visible"); assert.equal(history.runs[0].response, null);
    const events = [], replay = await streamChatAnswer(session.id, request, signal, (event) => events.push(event));
    assert.equal(replay.status, status); assert.equal(replay.response, null);
    assert.deepEqual(events.map((event) => event.type), ["run"]);
  }
});

test("history rejects cross-session rows, unsafe numeric IDs, unordered runs and unchecked citations", async () => {
  const corrupted = answerFixture(); corrupted.answer.statements[0].citations[0].quote = "无需确认";
  for (const runs of [[runFixture({ session_id: "77" })], [runFixture({ id: 9007199254740994 })],
    [runFixture(), runFixture()], [runFixture({ response: corrupted })]]) {
    const { signal } = await setup(() => json({ runs, has_more: false, next_before_run_id: null }));
    await assert.rejects(listChatRuns(session.id, signal), /历史/);
  }
});

test("SSE reads early run IDs, cumulative reviewed statements and final durable result across split Chinese bytes", async () => {
  const first = answerFixture(), last = answerFixture();
  last.answer.statements.push({ text: "需要负责人确认。", citations: first.answer.statements[0].citations });
  const events = [];
  const { calls, signal } = await setup(() => stream(": ping\r\n\r\n" + frame("run", pendingFixture())
    + frame("progress", { stage: "analysis" }, "\r") + frame("progress", { stage: "review" }, "\n")
    + frame("answer", envelope(first)) + frame("answer", envelope(last)) + frame("done", envelope(last)), 1));
  const run = await streamChatAnswer(session.id, request, signal, (event) => events.push(event));
  assert.deepEqual(events.map((event) => event.type), ["run", "progress", "progress", "answer", "answer"]);
  assert.equal(run.id, "9007199254740994"); assert.equal(run.response.answer.statements.length, 2);
  assert.equal(calls[0].init.headers.get("Accept"), "text/event-stream");
  assert.deepEqual(JSON.parse(calls[0].init.body), request);
});

test("replaying an unavailable or failed terminal run emits done with null response", async () => {
  for (const status of ["SUCCEEDED", "FAILED", "CANCELLED", "ABANDONED"]) {
    const events = [];
    const { signal } = await setup(() => stream(frame("done", envelope(null, { status, visibility: "unavailable" }))));
    const run = await streamChatAnswer(session.id, request, signal, (event) => events.push(event));
    assert.equal(run.response, null); assert.deepEqual(events, []);
  }
});

test("successful server replay starts with terminal run metadata before reviewed answer and done", async () => {
  const value = envelope(), firstRun = { ...value.run, standalone_question: null };
  const { signal } = await setup(() => stream(frame("run", firstRun) + frame("answer", value) + frame("done", value)));
  const events = [], replay = await streamChatAnswer(session.id, request, signal, (event) => events.push(event));
  assert.deepEqual(events.map((event) => event.type), ["run", "answer"]);
  assert.equal(events[0].run.response, null);
  assert.equal(replay.standalone_question, value.run.standalone_question);
  assert.deepEqual(replay.response, value.response);
});

test("SSE rejects foreign runs, changed immutable metadata and forged citations before callbacks", async () => {
  for (const corruption of ["foreign_session", "foreign_request", "foreign_question", "forged_quote", "unreviewed"]) {
    const value = envelope();
    if (corruption === "foreign_session") value.run.session_id = "42";
    if (corruption === "foreign_request") value.run.client_request_id = "other-request";
    if (corruption === "foreign_question") value.run.question = "另一个问题";
    if (corruption === "forged_quote") value.response.answer.statements[0].citations[0].quote = "无需确认";
    if (corruption === "unreviewed") value.response.answer.review_status = "failed";
    const events = [], { signal } = await setup(() => stream(frame("answer", value) + frame("done", value)));
    await assert.rejects(streamChatAnswer(session.id, request, signal, (event) => events.push(event)), /回答流或引用不完整/);
    assert.deepEqual(events, []);
  }
});

test("missing terminal event can be recovered by fetching the already persisted run without re-posting", async () => {
  const { calls, signal } = await setup((url) => url.endsWith("/stream")
    ? stream(frame("run", pendingFixture()) + frame("answer", envelope())) : json(runFixture()));
  let runId;
  await assert.rejects(streamChatAnswer(session.id, request, signal, (event) => { if (event.type === "run") runId = event.run.id; }), /连接中断/);
  const persisted = await getChatRun(session.id, runId, signal);
  assert.equal(persisted.status, "SUCCEEDED");
  assert.equal(calls.filter(({ init }) => init.method === "POST").length, 1);
});

test("SSE after-header 401 clears login and never refreshes or repeats generation", async () => {
  const { signal, calls, client } = await setup(() => stream(frame("error", { status: 401, code: "SESSION_REVOKED", message: "登录失效" })));
  await assert.rejects(streamChatAnswer(session.id, request, signal, () => {}), (error) => error.status === 401);
  assert.equal(client.snapshot.identity, null); assert.equal(calls.length, 1);
});

test("in-progress conflicts and permission denials preserve scope without implicit retries", async () => {
  for (const [status, code] of [[409, "CHAT_RUN_IN_PROGRESS"], [403, "SCOPE_FORBIDDEN"]]) {
    const { calls, signal } = await setup(() => json({ detail: { code, message: "需要恢复历史" } }, status));
    await assert.rejects(streamChatAnswer(session.id, request, signal, () => {}), (error) => error.status === status && error.code === code);
    assert.equal(calls.length, 1); assert.deepEqual(JSON.parse(calls[0].init.body), request);
  }
});

test("stream cancellation releases the reader and suppresses late events", async () => {
  const controller = new AbortController(); let cancelled = false;
  const { calls } = await setup(() => new Response(new ReadableStream({ cancel() { cancelled = true; } }), { headers: { "Content-Type": "text/event-stream" } }));
  const events = [], result = streamChatAnswer(session.id, request, controller.signal, (event) => events.push(event));
  await new Promise((resolve) => setImmediate(resolve)); controller.abort();
  await assert.rejects(result, { name: "AbortError" });
  assert.equal(cancelled, true); assert.equal(calls.length, 1); assert.deepEqual(events, []);
});

test("SSE final response must preserve all previously displayed statements and evidence", async () => {
  const first = envelope(), changed = envelope();
  changed.response.answer.statements[0].text = "更改的回答";
  const changedScope = envelope(); changedScope.response.scope.knowledge_base_ids = ["99"];
  for (const last of [changed, changedScope]) {
    const { signal } = await setup(() => stream(frame("answer", first) + frame("done", last)));
    await assert.rejects(streamChatAnswer(session.id, request, signal, () => {}), /回答流或引用不完整/);
  }
});

test("SSE cannot replace the early run ID or embed a complete answer in the run envelope", async () => {
  const foreignRun = envelope(undefined, { id: "9007199254740995" });
  const prematureAnswer = envelope(undefined, { response: answerFixture() });
  for (const value of [foreignRun, prematureAnswer]) {
    const { signal } = await setup(() => stream(frame("run", pendingFixture()) + frame("answer", value)));
    const events = [];
    await assert.rejects(streamChatAnswer(session.id, request, signal, (event) => events.push(event)), /回答流或引用不完整/);
    assert.deepEqual(events.map((event) => event.type), ["run"]);
  }
});

test("truncated and non-SSE responses never become successful persisted answers", async () => {
  for (const response of [stream("event: done\ndata: {\n\n"), stream(frame("done", envelope()).slice(0, -2)), json(runFixture())]) {
    const { signal } = await setup(() => response);
    await assert.rejects(streamChatAnswer(session.id, request, signal, () => {}), /回答流|连接中断/);
  }
});

// Mount the real setup function with Vue's renderer. API doubles keep interaction tests
// deterministic while exercising request guards, refs, watchers and recovery sequencing.
const vueUrl = import.meta.resolve("vue");
const uiAuthUrl = moduleUrl(`import { shallowRef } from ${JSON.stringify(vueUrl)};
  export const authState = shallowRef({ identity: { user: { id: "1" }, tenant: { id: "10" } } });`);
const uiApiUrl = moduleUrl(`export { createChatRequest, isPendingRun, chatSessionStorageKey, CHAT_SESSIONS_PAGE_SIZE } from ${JSON.stringify(chatUrl)};
  export let handlers; export function setHandlers(value) { handlers = value; }
  export const listChatSessions = (...args) => handlers.listChatSessions(...args);
  export const createChatSession = (...args) => handlers.createChatSession(...args);
  export const listChatRuns = (...args) => handlers.listChatRuns(...args);
  export const getChatRun = (...args) => handlers.getChatRun(...args);
  export const streamChatAnswer = (...args) => handlers.streamChatAnswer(...args);`);
const { setHandlers } = await import(uiApiUrl);
const iconsUrl = moduleUrl(`export const FileText = {}, LoaderCircle = {}, Search = {}, RefreshCw = {}, X = {}, ShieldCheck = {}, Plus = {}, MessageSquare = {};`);
const diagnosticsUrl = moduleUrl("export const analysisFailureLabel = () => ''; export const queryStatus = () => ({}); export const slowestQueryStage = () => null; export const graphPathViews = () => [];");
const uploadUrl = moduleUrl("export const isDemoMode = false;");
const sfc = parse(await readFile(new URL("../src/QueryWorkspace.vue", import.meta.url), "utf8"), { filename: "QueryWorkspace.vue" });
const componentSource = compileScript(sfc.descriptor, { id: "chat-interaction-test" }).content
  .replace(/from ["']vue["']/g, `from ${JSON.stringify(vueUrl)}`)
  .replace('"lucide-vue-next"', JSON.stringify(iconsUrl))
  .replace('"./services/upload"', JSON.stringify(uploadUrl))
  .replace('"./services/query"', JSON.stringify(queryUrl))
  .replace('"./services/query-diagnostics"', JSON.stringify(diagnosticsUrl))
  .replace('"./services/auth"', JSON.stringify(uiAuthUrl))
  .replace('"./services/chat"', JSON.stringify(uiApiUrl));
const { default: ChatWorkspace } = await import(moduleUrl(componentSource));
ChatWorkspace.render = () => null;
const renderer = createRenderer({
  createElement: (type) => ({ type, children: [] }), createText: (text) => ({ text }), createComment: (text) => ({ text }),
  insert: (child, parent) => { parent.children.push(child); child.parent = parent; },
  remove: (child) => { if (child.parent) child.parent.children = child.parent.children.filter((item) => item !== child); },
  setText: (node, text) => { node.text = text; }, setElementText: (node, text) => { node.text = text; },
  patchProp: () => {}, parentNode: (node) => node.parent, nextSibling: () => null,
});
const flushUi = async () => { for (let turn = 0; turn < 6; turn++) { await nextTick(); await new Promise((resolve) => setImmediate(resolve)); } };
async function mountWorkspace(overrides = {}) {
  const calls = [];
  setHandlers({
    listChatSessions: async () => [session, { ...session, id: "22", title: "另一个会话" }],
    createChatSession: async () => ({ ...session, id: "33", title: "新会话" }),
    listChatRuns: async (sessionId) => { calls.push(["history", sessionId]); return { runs: sessionId === session.id ? [runFixture()] : [], has_more: false, next_before_run_id: null }; },
    getChatRun: async (sessionId, runId) => { calls.push(["get", sessionId, runId]); return runFixture(); },
    streamChatAnswer: async () => { throw new Error("test must configure streaming"); }, ...overrides,
  });
  const app = renderer.createApp(ChatWorkspace, { bases: [{ id: "42", name: "工单知识库" }], basesLoading: false, basesError: "" });
  app.mount({ children: [] }); await flushUi();
  return { app, vm: app._instance.setupState, calls };
}

test("typing a follow-up keeps previous answer and citations intact; switching scope keeps saved history", async () => {
  const { app, vm } = await mountWorkspace();
  try {
    assert.equal(vm.messages.length, 1);
    vm.query = "它怎么处理？"; vm.selectedBase = "42"; await flushUi();
    assert.equal(vm.messages.length, 1);
    assert.equal(vm.messages[0].response.answer.statements[0].citations[0].citation_id, "E1");
    assert.equal(vm.query, "它怎么处理？");
  } finally { app.unmount(); }
});

test("switching sessions ignores late history responses and never replaces the new conversation", async () => {
  const { app, vm } = await mountWorkspace();
  let resolveOld;
  setHandlers({ listChatRuns: async (sessionId) => sessionId === session.id
    ? new Promise((resolve) => { resolveOld = resolve; }) : { runs: [], has_more: false, next_before_run_id: null } });
  try {
    const oldLoad = vm.refreshHistory(); await flushUi();
    await vm.selectSession("22");
    resolveOld({ runs: [runFixture()], has_more: false, next_before_run_id: null }); await oldLoad; await flushUi();
    assert.equal(vm.selectedSession, "22"); assert.deepEqual(vm.messages, []);
  } finally { app.unmount(); }
});

test("a missing done event restores the known run by GET and sends only one generation request", async () => {
  let streamCalls = 0, getCalls = 0;
  const { app, vm } = await mountWorkspace({
    streamChatAnswer: async (sessionId, body, signal, onEvent) => {
      streamCalls++;
      onEvent({ type: "run", run: pendingFixture() });
      throw new Error("回答连接中断");
    },
    getChatRun: async () => { getCalls++; return runFixture(); },
  });
  try {
    vm.query = request.query;
    await vm.sendAnswer(request); await flushUi();
    assert.equal(streamCalls, 1); assert.equal(getCalls, 1);
    assert.equal(vm.messages[0].status, "SUCCEEDED"); assert.equal(vm.retryRequest, null);
    assert.equal(vm.messages[0].response.answer.statements[0].text, "关闭前需由负责人确认。");
  } finally { app.unmount(); }
});

test("a failure before receiving run ID checks history; manual retry reuses the identical old body despite draft edits", async () => {
  const sent = [];
  const { app, vm } = await mountWorkspace({ listChatRuns: async () => ({ runs: [], has_more: false, next_before_run_id: null }),
    streamChatAnswer: async (sessionId, body) => { sent.push(JSON.stringify(body)); if (sent.length === 1) throw new Error("网络中断"); return runFixture(); } });
  try {
    vm.query = request.query; await vm.sendAnswer(request);
    assert.ok(vm.retryRequest);
    vm.query = "改成另一个问题"; vm.selectedBase = "";
    await vm.recoverRetry(true);
    assert.equal(sent.length, 2); assert.equal(sent[0], sent[1]);
    assert.equal(vm.query, "改成另一个问题"); assert.equal(vm.retryRequest, null);
  } finally { app.unmount(); }
});

test("readable-base refresh removes previous assistant text before restoring authorization-checked history", async () => {
  const { app, vm } = await mountWorkspace(); let finishRestore;
  setHandlers({ listChatRuns: async () => new Promise((resolve) => { finishRestore = resolve; }) });
  try {
    assert.equal(vm.messages[0].response.answer.status, "answered");
    app._instance.props.bases = [];
    await nextTick();
    assert.deepEqual(vm.messages, []); assert.equal(vm.historyLoading, true);
    finishRestore({ runs: [runFixture({ response: null, visibility: "unavailable", error_code: "CHAT_SOURCE_UNAVAILABLE" })], has_more: false, next_before_run_id: null });
    await flushUi();
    assert.equal(vm.messages[0].response, null); assert.equal(vm.messages[0].visibility, "unavailable");
  } finally { app.unmount(); }
});

test("cached conversations beyond the first 50 sessions restore by paging without selecting an unrelated first session", async () => {
  const originalStorage = globalThis.localStorage;
  const pages = Array.from({ length: 101 }, (_, index) => ({ ...session, id: String(1001 + index), title: `历史会话 ${index}` }));
  const target = pages[100].id, key = chatSessionStorageKey("10", "1"), storage = new Map([[key, target]]);
  globalThis.localStorage = { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key) };
  const offsets = []; let completeSecondPage;
  let mounted;
  try {
    mounted = await mountWorkspace({
      listChatSessions: async (signal, offset) => {
        offsets.push(offset);
        return offset === 50 ? new Promise((resolve) => { completeSecondPage = resolve; }) : pages.slice(offset, offset + 50);
      },
      listChatRuns: async (sessionId) => ({ runs: [runFixture({ session_id: sessionId })], has_more: false, next_before_run_id: null }),
    });
    assert.deepEqual(offsets, [0, 50]);
    assert.equal(mounted.vm.selectedSession, ""); assert.equal(storage.get(key), target);
    completeSecondPage(pages.slice(50, 100)); await flushUi();
    assert.deepEqual(offsets, [0, 50, 100]);
    assert.equal(mounted.vm.selectedSession, target);
    assert.equal(mounted.vm.messages[0].session_id, target);
    assert.equal(mounted.vm.sessions.length, 101); assert.equal(mounted.vm.sessionsHasMore, false);
  } finally {
    mounted?.app.unmount();
    if (originalStorage === undefined) delete globalThis.localStorage; else globalThis.localStorage = originalStorage;
  }
});

test("loading earlier sessions appends unique rows and keeps the currently selected conversation", async () => {
  const pages = Array.from({ length: 51 }, (_, index) => ({ ...session, id: String(2001 + index), title: `会话 ${index}` }));
  const offsets = [];
  const { app, vm } = await mountWorkspace({
    listChatSessions: async (signal, offset) => { offsets.push(offset); return offset === 0 ? pages.slice(0, 50) : [pages[0], pages[50]]; },
    listChatRuns: async () => ({ runs: [], has_more: false, next_before_run_id: null }),
  });
  try {
    assert.equal(vm.sessions.length, 50); assert.equal(vm.sessionsHasMore, true);
    const selected = vm.selectedSession;
    await vm.loadMoreSessions();
    assert.deepEqual(offsets, [0, 50]); assert.equal(vm.sessions.length, 51);
    assert.equal(vm.selectedSession, selected); assert.equal(vm.sessionsHasMore, false);
  } finally { app.unmount(); }
});

test("recovering visible terminal failures reports their status instead of claiming a restored answer", async () => {
  for (const status of ["FAILED", "CANCELLED", "ABANDONED"]) {
    const terminal = runFixture({ status, response: null, visibility: "visible" });
    const { app, vm } = await mountWorkspace({ getChatRun: async () => terminal });
    try {
      vm.retryRequest = { sessionId: session.id, body: request, runId: terminal.id };
      await vm.recoverRetry();
      assert.equal(vm.retryRequest, null); assert.equal(vm.messages[0].response, null);
      assert.doesNotMatch(vm.notice, /已恢复.*回答/);
      assert.match(vm.notice, status === "FAILED" ? /失败/ : status === "CANCELLED" ? /取消/ : /中断/);
    } finally { app.unmount(); }
  }
});

test("polling a visible failed terminal run stops recovery and reports failure without an answer", async (context) => {
  const terminal = runFixture({ status: "FAILED", response: null, visibility: "visible" });
  const { app, vm } = await mountWorkspace({ getChatRun: async () => terminal });
  context.mock.timers.enable({ apis: ["setTimeout"] });
  try {
    vm.retryRequest = { sessionId: session.id, body: request, runId: terminal.id };
    vm.pollRun(pendingFixture());
    context.mock.timers.tick(1500); await flushUi();
    assert.equal(vm.messages[0].status, "FAILED"); assert.equal(vm.retryRequest, null);
    assert.match(vm.notice, /失败/); assert.doesNotMatch(vm.notice, /已恢复.*回答/);
  } finally { app.unmount(); context.mock.timers.reset(); }
});
