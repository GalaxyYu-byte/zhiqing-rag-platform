import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
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
const source = (await readFile(new URL("../src/services/query.ts", import.meta.url), "utf8"))
  .replace('"./auth"', JSON.stringify(dependencyUrl)).replace('"./auth-client"', JSON.stringify(authUrl))
  .replaceAll("import.meta.env", "({})");
const { searchQuery, answerQuery, streamAnswerQuery, QuerySearchError } = await import(moduleUrl(source));
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
