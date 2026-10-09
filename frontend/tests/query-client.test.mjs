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
  export const authenticatedFetch = (url, options) => client.fetch(url, options);`);
const { setClient } = await import(dependencyUrl);
const source = (await readFile(new URL("../src/services/query.ts", import.meta.url), "utf8"))
  .replace('"./auth"', JSON.stringify(dependencyUrl)).replace('"./auth-client"', JSON.stringify(authUrl))
  .replaceAll("import.meta.env", "({})");
const { searchQuery, QuerySearchError } = await import(moduleUrl(source));
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
