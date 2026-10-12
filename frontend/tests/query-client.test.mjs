import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import { queryFixture } from "./helpers/query-fixture.mjs";
import { graphPathFixture } from "./helpers/graph-path-fixture.mjs";

const moduleUrl = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText).toString("base64")}`;
const authUrl = moduleUrl(await readFile(new URL("../src/services/auth-client.ts", import.meta.url), "utf8"));
const { AuthClient } = await import(authUrl);
const dependencyUrl = moduleUrl(`export let client; export function setClient(value) { client = value; }
  export const authClient = { clearSession: (notice) => client.clearSession(notice) };
  export const authenticatedFetch = (url, options) => client.fetch(url, options);`);
const { setClient } = await import(dependencyUrl);
const source = (await readFile(new URL("../src/services/query.ts", import.meta.url), "utf8"))
  .replace('"./auth"', JSON.stringify(dependencyUrl)).replace('"./auth-client"', JSON.stringify(authUrl))
  .replaceAll("import.meta.env", "({})");
const { searchQuery, searchQueryPage, answerQuery, streamAnswerQuery, QuerySearchError, validatedGraphPaths, validQueryAnswerResponse, structuredAnswerText } = await import(moduleUrl(source));
const json = (body, status = 200) => new Response(JSON.stringify(body), { status });
const session = (token = "token") => ({ access_token: token, user: { id: "1", email: "test@example.com", display_name: "测试" }, tenant: { id: "10", name: "测试租户", code: "TEST", role: "MEMBER" } });
async function setup(handler = () => json(queryFixture())) {
  const calls = [];
  const client = new AuthClient({ fetcher: async (url, init) => {
    if (url.endsWith("/select-tenant")) return json(session());
    calls.push({ url, init });
    return handler(url, init);
  } });
  await client.selectTenant("selection", "10");
  setClient(client);
  return { calls, client, signal: new AbortController().signal };
}

function answerFixture() {
  return { ...queryFixture(), answer: {
    status: "answered", statements: [{ text: "关闭前需由负责人确认。", citations: [{ citation_id: "E1", quote: "工单关闭前需由负责人确认。" }] }],
    message: null, codes: [], model: "fake", review_status: "passed",
    trace: { generation_ms: 20, review_ms: 10, authorization_ms: 2, total_ms: 32, context_tokens: 100, input_tokens: 150, output_tokens: 40 },
  } };
}

test("answer requests submit only the authenticated question and selected scope", async () => {
  const { calls, signal } = await setup(() => json(answerFixture()));
  const result = await answerQuery("  如何关闭？  ", "42", signal);
  assert.equal(calls[0].url, "/api/query/answer");
  assert.deepEqual(JSON.parse(calls[0].init.body), { query: "如何关闭？", knowledge_base_ids: ["42"] });
  assert.equal(result.answer.statements[0].citations[0].citation_id, "E1");
});

test("answer token refresh preserves the identical scope and request body", async () => {
  let requests = 0;
  const { calls, signal } = await setup((url) => url.endsWith("/refresh") ? json(session("new-token"))
    : ++requests === 1 ? json({ detail: { code: "ACCESS_TOKEN_EXPIRED" } }, 401) : json(answerFixture()));
  await answerQuery("如何关闭？", "42", signal);
  const answers = calls.filter(({ url }) => url.endsWith("/answer"));
  assert.equal(answers.length, 2);
  assert.equal(answers[0].init.body, answers[1].init.body);
});

for (const corruption of ["unknown_reference", "rewritten_quote", "unreviewed", "insufficient_draft"]) {
  test(`answer rejects ${corruption} before displaying model text`, async () => {
    const value = answerFixture();
    if (corruption === "unknown_reference") value.answer.statements[0].citations[0].citation_id = "E9";
    if (corruption === "rewritten_quote") value.answer.statements[0].citations[0].quote = "无需确认即可关闭";
    if (corruption === "unreviewed") value.answer.review_status = "failed";
    if (corruption === "insufficient_draft") value.answer.status = "insufficient";
    const { signal } = await setup(() => json(value));
    await assert.rejects(answerQuery("如何关闭？", "", signal), /回答或引用响应不完整/);
  });
}

test("insufficient answers remain explicit and contain no guessed draft", async () => {
  const value = answerFixture();
  Object.assign(value.answer, { status: "insufficient", statements: [], review_status: "skipped", message: "资料不足" });
  const { signal } = await setup(() => json(value));
  const result = await answerQuery("无答案问题", "", signal);
  assert.equal(result.answer.status, "insufficient");
  assert.deepEqual(result.answer.statements, []);
});

test("graph-enhanced results preserve trace and original relation provenance", async () => {
  const value = queryFixture();
  value.plan.route = value.plan.requested_route = "hybrid_graph";
  Object.assign(value.retrieval.trace, { graph_status: "succeeded", graph_count: 1,
    graph_seed_count: 1, graph_assertion_count: 1, graph_hops: 1, graph_ms: 80 });
  value.retrieval.evidence[0].graph_relations = [{ assertion_id: "a", seed_entity_uid: "s",
    subject_entity_uid: "s", object_entity_uid: "o", predicate: "RESPONSIBLE_FOR",
    condition_text: "关闭前", source_quote: value.retrieval.evidence[0].content,
    quote_start: 0, quote_end: [...value.retrieval.evidence[0].content].length }];
  const { signal } = await setup(() => json(value));
  const result = await searchQuery("如何关闭？", "42", signal);
  assert.equal(result.retrieval.trace.graph_status, "succeeded");
  assert.equal(result.retrieval.evidence[0].graph_relations[0].condition_text, "关闭前");
});

test("invalid graph trace cannot be presented as a valid retrieval result", async () => {
  const value = queryFixture();
  value.retrieval.trace.graph_status = "unbounded";
  const { signal } = await setup(() => json(value));
  await assert.rejects(searchQuery("如何关闭？", "42", signal), /检索响应不完整/);
});

test("two-hop paths retain complete cross-document provenance and Unicode code-point offsets", async () => {
  for (const options of [{}, { bridge: true }, { astral: true }]) {
    const value = graphPathFixture(options);
    const { signal } = await setup(() => json(value));
    const result = await searchQuery("关闭摘要与确认结论如何关联？", "", signal);
    assert.deepEqual(validatedGraphPaths(result), value.retrieval.graph_paths);
    assert.equal(result.retrieval.trace.graph_path_count, 1);
    assert.equal(result.retrieval.graph_paths[0].hops[0].source.source_quote, value.retrieval.evidence[0].content);
  }
});

test("incoming path traversal preserves the business assertion direction", async () => {
  const value = graphPathFixture();
  const hop = value.retrieval.graph_paths[0].hops[0];
  const relation = hop.relation;
  [relation.subject_entity_uid, relation.object_entity_uid] = [relation.object_entity_uid, relation.subject_entity_uid];
  [hop.subject_name, hop.object_name] = [hop.object_name, hop.subject_name];
  hop.traversal = "incoming";
  const { signal } = await setup(() => json(value));
  assert.equal((await searchQuery("关联路径？", "", signal)).retrieval.graph_paths[0].hops[0].traversal, "incoming");
});

const pathCorruptions = {
  unknown_predicate: (value, path) => { path.hops[0].relation.predicate = "RELATED_TO"; },
  unknown_traversal: (value, path) => { path.hops[0].traversal = "unbounded"; },
  missing_final_source: (value) => { value.retrieval.evidence.pop(); },
  missing_required_chunk: (value, path) => { path.required_chunk_ids.pop(); },
  extra_required_chunk: (value, path) => { path.required_chunk_ids.push("999"); },
  duplicate_required_chunk: (value, path) => { path.required_chunk_ids.push(path.required_chunk_ids[0]); },
  wrong_document: (value, path) => { path.hops[0].source.document_id = "999"; },
  wrong_revision: (value, path) => { path.hops[0].source.revision_id = "999"; },
  wrong_generation: (value, path) => { path.hops[0].source.generation_id = "999"; },
  wrong_tenant: (value, path) => { path.hops[0].source.tenant_id = "99"; },
  wrong_projection: (value, path) => { path.hops[0].source.projection_id = "99"; },
  wrong_namespace: (value, path) => { path.hops[0].source.namespace = "another_projection"; },
  cross_knowledge_base: (value, path) => { path.knowledge_base_id = "99"; },
  offset_overflow: (value, path) => { path.hops[0].source.quote_end++; },
  changed_whitespace: (value, path) => { path.hops[0].source.source_quote = path.hops[0].source.source_quote.trim(); },
  dropped_relation: (value) => { value.retrieval.evidence[1].graph_relations = []; },
  disconnected_hops: (value, path) => { Object.assign(path.hops[1].relation, { subject_entity_uid: "unknown-middle", seed_entity_uid: "unknown-middle" }); },
  cycle: (value, path) => { path.hops[1].relation.object_entity_uid = "step-1"; },
  duplicate_assertion: (value, path) => { path.hops[1].relation.assertion_id = path.hops[0].relation.assertion_id; },
  unapproved_seed: (value) => { value.plan.params.seed_entity_uids = ["another-seed"]; },
  unsupported_hops: (value) => { value.plan.params.max_hops = 3; },
  wrong_path_count: (value) => { value.retrieval.trace.graph_path_count = 2; },
  wrong_candidate_count: (value) => { value.retrieval.trace.graph_path_candidate_count = 0; },
  failed_graph_trace: (value) => { value.retrieval.trace.graph_status = "failed"; },
};
for (const [name, corrupt] of Object.entries(pathCorruptions)) {
  test(`unsafe two-hop ${name} is rejected and hidden from display`, async () => {
    const value = graphPathFixture();
    corrupt(value, value.retrieval.graph_paths[0]);
    const { signal } = await setup(() => json(value));
    await assert.rejects(searchQuery("关联路径？", "", signal), /检索响应不完整/);
    assert.deepEqual(validatedGraphPaths(value), []);
  });
}

test("bridge mappings require allowed entity type, explicit basis and every source citation", async () => {
  for (const corruption of ["entity_type", "basis", "disconnected", "missing_source", "scope"]) {
    const value = graphPathFixture({ bridge: true });
    const bridge = value.retrieval.graph_paths[0].bridges[0];
    if (corruption === "entity_type") bridge.entity_type = "Step";
    if (corruption === "basis") bridge.basis = "same_name_only";
    if (corruption === "disconnected") bridge.target_entity_uid = "unrelated-entity";
    if (corruption === "missing_source") value.retrieval.evidence.pop();
    if (corruption === "scope") bridge.sources[1].document_id = "999";
    const { signal } = await setup(() => json(value));
    await assert.rejects(searchQuery("关联路径？", "", signal), /检索响应不完整/);
  }
});

test("shared legacy entity UID still requires grounded cross-document mapping", async () => {
  const value = graphPathFixture({ bridge: true });
  const path = value.retrieval.graph_paths[0];
  const bridge = path.bridges[0];
  bridge.target_entity_uid = bridge.source_entity_uid;
  bridge.identity_id = "b".repeat(64); bridge.identity_version = 1;
  Object.assign(path.hops[1].relation, { subject_entity_uid: bridge.target_entity_uid,
    seed_entity_uid: bridge.target_entity_uid });
  const { signal } = await setup(() => json(value));
  assert.equal((await searchQuery("关联路径？", "", signal)).retrieval.graph_paths.length, 1);
  path.bridges = [];
  const next = await setup(() => json(value));
  await assert.rejects(searchQuery("关联路径？", "", next.signal), /检索响应不完整/);
});

test("answer repair count is optional and bounded to one attempt", () => {
  const value = answerFixture();
  for (const count of [0, 1]) {
    value.answer.trace.repair_count = count;
    assert.ok(validQueryAnswerResponse(value));
  }
  for (const count of [-1, 2, 0.5, "1"]) {
    value.answer.trace.repair_count = count;
    assert.equal(validQueryAnswerResponse(value), false);
  }
});

test("matched entity names accept old absence but reject invalid text and excessive budgets", async () => {
  for (const names of [[" "], [123], ["名".repeat(201)], Array(21).fill("对象")]) {
    const value = graphPathFixture(); value.plan.params.matched_entity_names = names;
    const { signal } = await setup(() => json(value));
    await assert.rejects(searchQuery("关联路径？", "", signal), /检索响应不完整/);
  }
  const value = graphPathFixture(); delete value.plan.params.matched_entity_names;
  const { signal } = await setup(() => json(value));
  assert.equal((await searchQuery("关联路径？", "", signal)).retrieval.graph_paths.length, 1);
});

test("legacy history without paths or optional graph fields remains readable", () => {
  const value = answerFixture();
  assert.ok(validQueryAnswerResponse(value));
  assert.deepEqual(validatedGraphPaths(value), []);
});

test("retrieval authenticates and submits only query and optional string knowledge-base ID", async () => {
  const { calls, signal } = await setup();
  const result = await searchQuery("  工单关闭前需要谁确认？  ", "9007199254740993", signal);
  assert.equal(result.retrieval.evidence[0].content, queryFixture().retrieval.evidence[0].content);
  await searchQuery("工单如何关闭？", "", signal);
  assert.equal(calls[0].url, "/api/query/search");
  assert.deepEqual(JSON.parse(calls[0].init.body), { query: "工单关闭前需要谁确认？", knowledge_base_ids: ["9007199254740993"] });
  assert.deepEqual(JSON.parse(calls[1].init.body), { query: "工单如何关闭？" });
  assert.equal(calls[0].init.headers.get("Authorization"), "Bearer token");
  assert.equal(calls[0].init.headers.get("X-Requested-With"), "XMLHttpRequest");
  assert.equal(calls[0].init.credentials, "include");
});

test("expired token refresh retries the identical selected scope once", async () => {
  let searches = 0;
  const { calls, signal } = await setup((url) => url.endsWith("/refresh") ? json(session("new-token"))
    : ++searches === 1 ? json({ detail: { code: "ACCESS_TOKEN_EXPIRED" } }, 401) : json(queryFixture()));
  await searchQuery("如何关闭？", "9007199254740993", signal);
  const requests = calls.filter(({ url }) => url.endsWith("/search"));
  assert.equal(requests.length, 2);
  assert.equal(requests[0].init.body, requests[1].init.body);
  assert.equal(requests[1].init.headers.get("Authorization"), "Bearer new-token");
});

test("scope denial exposes the error without widening scope or retrying", async () => {
  const { calls, client, signal } = await setup(() => json({ detail: { code: "SCOPE_FORBIDDEN", message: "当前范围已不可读" } }, 403));
  await assert.rejects(searchQuery("关闭流程", "42", signal), (error) => error instanceof QuerySearchError
    && error.status === 403 && error.code === "SCOPE_FORBIDDEN" && error.message === "当前范围已不可读");
  assert.equal(calls.length, 1);
  assert.ok(client.snapshot.identity);
});

test("empty questions and prior cancellation never issue a request", async () => {
  const { calls, signal } = await setup();
  await assert.rejects(searchQuery("  ", "", signal), /1 到 2000/);
  await assert.rejects(searchQuery("字".repeat(2001), "", signal), /1 到 2000/);
  await assert.rejects(searchQuery("问题", "", AbortSignal.abort()), { name: "AbortError" });
  assert.equal(calls.length, 0);
});

test("cancellation while decoding a response cannot expose late evidence", async () => {
  const controller = new AbortController();
  await setup(() => ({ status: 200, ok: true, json: async () => { controller.abort(); return queryFixture(); } }));
  await assert.rejects(searchQuery("问题", "", controller.signal), { name: "AbortError" });
});

test("malformed success responses and non-JSON errors fail clearly", async () => {
  for (const response of [json({}), json(queryFixture({ retrieval: { trace: {} } })), json(queryFixture({ retrieval: { evidence: [{ content: "残缺证据" }] } }))]) {
    const { signal } = await setup(() => response);
    await assert.rejects(searchQuery("问题", "", signal), /响应不完整/);
  }
  const { signal } = await setup(() => new Response("bad gateway", { status: 502 }));
  await assert.rejects(searchQuery("问题", "", signal), (error) => error.status === 502 && error.message.includes("502"));
});

test("clarification, unsupported and empty are valid responses without invented evidence", async () => {
  for (const status of ["skipped", "unsupported", "empty"]) {
    const { signal } = await setup(() => json(queryFixture({ retrieval: { status, evidence: [] } })));
    assert.equal((await searchQuery("问题", "", signal)).retrieval.evidence.length, 0);
  }
});

const frame = (event, data, newline = "\r\n") => `event: ${event}${newline}${JSON.stringify(data, null, 2)
  .split("\n").map((line) => `data: ${line}`).join(newline)}${newline}${newline}`;
const streamResponse = (text, chunkSize = 1024) => {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  return new Response(new ReadableStream({
    pull(controller) {
      if (offset === bytes.length) { controller.close(); return; }
      controller.enqueue(bytes.slice(offset, offset += chunkSize));
      offset = Math.min(offset, bytes.length);
    },
  }), { headers: { "Content-Type": "text/event-stream; charset=utf-8" } });
};

function structuredFixture(total = 0) {
  const value = answerFixture();
  value.plan.route = "structured"; value.plan.requested_route = "structured";
  value.retrieval.evidence = []; value.retrieval.status = "skipped";
  value.structured = { operation: "count_documents", total, documents: [], offset: 0, limit: 20,
    applied_filters: [], scope: "authorized_published_documents", executed_at: "2026-10-11T00:00:00Z" };
  value.answer = { ...value.answer, status: "structured", statements: [], model: null,
    message: structuredAnswerText(value.structured) };
  return value;
}

test("zero document statistics remain publishable without chunk citations in JSON and SSE", async () => {
  const value = structuredFixture();
  assert.equal(validQueryAnswerResponse(value), true);
  const { signal } = await setup(() => streamResponse(frame("answer", value) + frame("done", value), 3));
  const events = [];
  assert.deepEqual(await streamAnswerQuery("文档有多少份？", "", signal, (event) => events.push(event)), value);
  assert.equal(events.length, 1);
  assert.equal(events[0].response.structured.total, 0);
});

test("department filters preserve bigint IDs as decimal strings through JSON and SSE", async () => {
  const value = structuredFixture();
  value.structured.applied_filters = [{ field: "department_id", operator: "eq", values: ["9007199254740993"] }];
  assert.equal(validQueryAnswerResponse(value), true);
  const { signal } = await setup(() => streamResponse(frame("answer", value) + frame("done", value), 5));
  const result = await streamAnswerQuery("文档有多少份？", "", signal, () => {});
  assert.equal(result.structured.applied_filters[0].values[0], "9007199254740993");
  for (const invalid of [9007199254740992, "0", "-1", "9223372036854775808", "1.5"]) {
    value.structured.applied_filters[0].values = [invalid];
    assert.equal(validQueryAnswerResponse(value), false);
  }
});

test("statistical answers reject missing results, changed totals and arbitrary summary text", () => {
  for (const kind of ["missing_result", "changed_total", "unreviewed", "arbitrary_text", "wrong_route", "negative_count"]) {
    const value = structuredFixture();
    if (kind === "missing_result") value.structured = null;
    if (kind === "changed_total") value.structured.total = 4;
    if (kind === "unreviewed") value.answer.review_status = "skipped";
    if (kind === "arbitrary_text") value.answer.message = "实际有一百万份文档";
    if (kind === "wrong_route") value.plan.route = "hybrid";
    if (kind === "negative_count") value.structured.total = -1;
    assert.equal(validQueryAnswerResponse(value), false, kind);
  }
});

test("catalog pagination resends the saved selection and filters instead of current drafts", async () => {
  const value = structuredFixture();
  const { signal, calls } = await setup(() => json(value));
  const filters = [{ field: "file_type", operator: "eq", value: "PDF" }];
  await searchQueryPage({ query: "列出所有文档", knowledge_base_ids: ["42"], document_ids: ["100"], filters, offset: 0, limit: 5 }, 5, signal);
  assert.deepEqual(JSON.parse(calls[0].init.body), { query: "列出所有文档", knowledge_base_ids: ["42"], document_ids: ["100"], filters, offset: 5, limit: 5 });
});

test("malformed replay requests are rejected before display or pagination", async () => {
  for (const request of [{ query: {} }, { query: "清单", limit: 0 }, { query: "清单", filters: [{ field: "sql", operator: "eq", value: "x" }] }, { query: "清单", knowledge_base_ids: ["bad"] }]) {
    const value = structuredFixture(); value.request = request;
    assert.equal(validQueryAnswerResponse(value), false);
    const { signal, calls } = await setup(() => json(value));
    await assert.rejects(searchQueryPage(request, 0, signal), /清单请求不完整/);
    assert.equal(calls.length, 0);
  }
});

test("reviewed two-hop JSON, SSE and restored-history responses use the same path checks", async () => {
  const value = graphPathFixture({ bridge: true, astral: true });
  value.answer = { ...answerFixture().answer, statements: [{ text: "两条关系的条件需分别核对。",
    citations: [{ citation_id: "E1", quote: value.retrieval.evidence[0].content }] }] };
  assert.ok(validQueryAnswerResponse(value));
  const { signal } = await setup(() => streamResponse(frame("answer", value) + frame("done", value), 7));
  const events = [];
  assert.deepEqual(await streamAnswerQuery("关联路径？", "", signal, (event) => events.push(event)), value);
  assert.equal(events.length, 1);
  value.retrieval.graph_paths[0].hops[0].source.quote_end++;
  assert.equal(validQueryAnswerResponse(value), false);
  const { signal: invalidSignal } = await setup(() => streamResponse(frame("answer", value) + frame("done", value)));
  const unpublished = [];
  await assert.rejects(streamAnswerQuery("关联路径？", "", invalidSignal, (event) => unpublished.push(event)), /回答流或引用不完整/);
  assert.deepEqual(unpublished, []);
});

test("SSE preserves authenticated POST scope and parses split UTF-8, CRLF, multiline data and heartbeats", async () => {
  const value = answerFixture();
  const events = [];
  const { calls, signal } = await setup(() => streamResponse(": ping\r\n\r\n"
    + frame("progress", { stage: "analysis" }) + frame("progress", { stage: "retrieval" }, "\r")
    + frame("progress", { stage: "generation" }, "\n") + frame("progress", { stage: "review" })
    + frame("answer", value) + frame("done", value), 1));
  const result = await streamAnswerQuery("  如何关闭？  ", "9007199254740993", signal, (event) => events.push(event));
  assert.equal(calls[0].url, "/api/query/answer/stream");
  assert.equal(calls[0].init.method, "POST");
  assert.equal(calls[0].init.headers.get("Authorization"), "Bearer token");
  assert.equal(calls[0].init.headers.get("Accept"), "text/event-stream");
  assert.deepEqual(JSON.parse(calls[0].init.body), { query: "如何关闭？", knowledge_base_ids: ["9007199254740993"] });
  assert.deepEqual(events.slice(0, 4).map((event) => event.stage), ["analysis", "retrieval", "generation", "review"]);
  assert.equal(events[4].response.answer.statements[0].text, value.answer.statements[0].text);
  assert.deepEqual(result, value);
});

test("progress and reviewed paragraphs arrive before the terminal event", async () => {
  let controller;
  const value = answerFixture();
  let finish;
  const seen = new Promise((resolve) => { finish = resolve; });
  const { signal } = await setup(() => new Response(new ReadableStream({ start(value) { controller = value; } }), {
    headers: { "Content-Type": "text/event-stream" },
  }));
  let completed = false;
  const events = [];
  const request = streamAnswerQuery("问题", "", signal, (event) => {
    events.push(event);
    if (event.type === "answer") finish();
  }).then((result) => { completed = true; return result; });
  controller.enqueue(new TextEncoder().encode(frame("progress", { stage: "analysis" }) + frame("answer", value)));
  await seen;
  assert.equal(completed, false);
  assert.deepEqual(events.map((event) => event.type), ["progress", "answer"]);
  controller.enqueue(new TextEncoder().encode(frame("done", value)));
  controller.close();
  assert.deepEqual(await request, value);
});

test("SSE publishes cumulative complete paragraphs and confirms the identical final response", async () => {
  const first = answerFixture(), last = answerFixture();
  last.answer.statements.push({ text: "需要负责人确认。", citations: first.answer.statements[0].citations });
  const counts = [];
  const { signal } = await setup(() => streamResponse(frame("answer", first) + frame("answer", last) + frame("done", last)));
  const result = await streamAnswerQuery("问题", "", signal, (event) => counts.push(event.response.answer.statements.length));
  assert.deepEqual(counts, [1, 2]);
  assert.equal(result.answer.statements.length, 2);
});

test("initial SSE token expiry refreshes the identical POST once before streaming", async () => {
  let requests = 0;
  const value = answerFixture();
  const { calls, signal } = await setup((url) => url.endsWith("/refresh") ? json(session("new-token"))
    : ++requests === 1 ? json({ detail: { code: "ACCESS_TOKEN_EXPIRED" } }, 401)
      : streamResponse(frame("answer", value) + frame("done", value)));
  await streamAnswerQuery("问题", "42", signal, () => {});
  const streams = calls.filter(({ url }) => url.endsWith("/stream"));
  assert.equal(streams.length, 2);
  assert.equal(streams[0].init.body, streams[1].init.body);
  assert.equal(streams[1].init.headers.get("Authorization"), "Bearer new-token");
});

for (const corruption of ["unknown_reference", "rewritten_quote", "unreviewed", "insufficient_draft"]) {
  test(`SSE rejects ${corruption} before publishing a paragraph`, async () => {
    const value = answerFixture();
    if (corruption === "unknown_reference") value.answer.statements[0].citations[0].citation_id = "E9";
    if (corruption === "rewritten_quote") value.answer.statements[0].citations[0].quote = "无需确认即可关闭";
    if (corruption === "unreviewed") value.answer.review_status = "failed";
    if (corruption === "insufficient_draft") value.answer.status = "insufficient";
    const events = [];
    const { signal } = await setup(() => streamResponse(frame("answer", value) + frame("done", value)));
    await assert.rejects(streamAnswerQuery("问题", "", signal, (event) => events.push(event)), /回答流或引用不完整/);
    assert.deepEqual(events, []);
  });
}

test("non-answer SSE outcomes terminate explicitly without publishing any draft", async () => {
  for (const status of ["insufficient", "rejected", "unavailable", "skipped", "unsupported"]) {
    const value = answerFixture(), events = [];
    Object.assign(value.answer, { status, statements: [], review_status: "skipped", message: "资料不足" });
    const { signal } = await setup(() => streamResponse(frame("done", value)));
    assert.equal((await streamAnswerQuery("问题", "", signal, (event) => events.push(event))).answer.status, status);
    assert.deepEqual(events, []);
  }
});

test("stream errors after headers are surfaced without refreshing or retrying generation", async () => {
  const { calls, signal, client } = await setup(() => streamResponse(frame("progress", { stage: "analysis" })
    + frame("error", { status: 401, code: "SESSION_REVOKED", message: "登录已失效" })));
  await assert.rejects(streamAnswerQuery("问题", "42", signal, () => {}), (error) => error instanceof QuerySearchError
    && error.status === 401 && error.code === "SESSION_REVOKED");
  assert.equal(calls.length, 1);
  assert.equal(client.snapshot.identity, null);
});

test("missing terminal event, truncated JSON and non-SSE responses cannot succeed", async () => {
  const value = answerFixture();
  for (const response of [streamResponse(frame("answer", value)), streamResponse("event: done\ndata: {\n\n"),
    streamResponse(frame("done", value).slice(0, -2)), json(value), streamResponse(frame("done", value))]) {
    const { signal } = await setup(() => response);
    await assert.rejects(streamAnswerQuery("问题", "", signal, () => {}), /回答流|连接中断/);
  }
});

test("changed paragraphs, evidence and final result are rejected instead of silently replacing the stream", async () => {
  const first = answerFixture();
  const changed = answerFixture();
  changed.answer.statements[0].text = "更改的回答";
  const differentScope = answerFixture();
  differentScope.scope.knowledge_base_ids = ["99"];
  for (const suffix of [frame("answer", first), frame("done", changed), frame("done", differentScope),
    frame("progress", { stage: "review" })]) {
    const { signal } = await setup(() => streamResponse(frame("answer", first) + suffix));
    await assert.rejects(streamAnswerQuery("问题", "", signal, () => {}), /回答流或引用不完整/);
  }
});

test("cancelling a stalled stream releases its reader and cannot publish late events", async () => {
  const requestController = new AbortController();
  let released = false;
  const { calls } = await setup(() => new Response(new ReadableStream({ cancel() { released = true; } }), {
    headers: { "Content-Type": "text/event-stream" },
  }));
  const events = [];
  const request = streamAnswerQuery("问题", "", requestController.signal, (event) => events.push(event));
  // 先让 fetch 和 reader 建立，再中止没有数据到达的连接。
  await new Promise((resolve) => setImmediate(resolve));
  requestController.abort();
  await assert.rejects(request, { name: "AbortError" });
  assert.equal(released, true);
  assert.equal(calls.length, 1);
  assert.deepEqual(events, []);
});
