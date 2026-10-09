import assert from "node:assert/strict";
import { File } from "node:buffer";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

function moduleUrl(source) {
  const compiled = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  }).outputText;
  return `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`;
}
const authSource = await readFile(new URL("../src/services/auth-client.ts", import.meta.url), "utf8");
const { AuthClient } = await import(moduleUrl(authSource));
const dependencyUrl = moduleUrl(`
export let authClient;
export function setClient(client) { authClient = client; }
export const authenticatedFetch = (url, init) => authClient.fetch(url, init);
`);
const { setClient } = await import(dependencyUrl);
const uploadSource = (await readFile(new URL("../src/services/upload.ts", import.meta.url), "utf8"))
  .replace('"./auth"', JSON.stringify(dependencyUrl)).replaceAll("import.meta.env", "({})");
const { uploadDocument, fetchKnowledgeBases } = await import(moduleUrl(uploadSource));
const response = (body, status = 200) => new Response(JSON.stringify(body), { status });
const session = (token) => ({
  access_token: token, token_type: "Bearer", expires_in: 900,
  user: { id: "1", email: "test@example.com", display_name: "用户" },
  tenant: { id: "10", code: "TEST", name: "租户", role: "ADMIN" },
});
const stored = {
  status: "STORED", processing_status: "PENDING", document_id: "11",
  revision_id: "12", generation_id: "13", task_id: "14",
};

async function setup(t, xhrReplies) {
  const original = globalThis.XMLHttpRequest;
  const requests = [];
  globalThis.XMLHttpRequest = class {
    headers = new Map();
    upload = {};
    aborted = false;
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(name, value) { this.headers.set(name, value); }
    send(body) {
      this.body = body;
      requests.push(this);
      const reply = xhrReplies.shift();
      if (reply) queueMicrotask(() => {
        if (this.aborted) return;
        this.upload.onprogress?.({ lengthComputable: true, loaded: 12, total: 12 });
        this.status = reply.status;
        this.response = reply.body;
        this.onload();
      });
    }
    abort() { this.aborted = true; this.onabort?.(); }
  };
  t.after(() => { globalThis.XMLHttpRequest = original; });
  let refreshes = 0;
  const client = new AuthClient({ fetcher: async (url) => {
    if (url.endsWith("/refresh")) refreshes++;
    return response(session(url.endsWith("/refresh") ? "fresh" : "initial"));
  } });
  await client.selectTenant("selection", "10");
  setClient(client);
  return { requests, client, refreshes: () => refreshes };
}

test("real upload transport rebuilds authenticated XHR and multipart body after one refresh", async (t) => {
  const { requests, client, refreshes } = await setup(t, [
    { status: 401, body: { detail: { code: "ACCESS_TOKEN_EXPIRED" } } },
    { status: 202, body: stored },
  ]);
  const progress = [];
  const result = await uploadDocument(new File(["test"], "test.txt"), {
    knowledgeBase: "7", documentId: "11", expectedRevisionId: "old-revision",
    signal: new AbortController().signal, onProgress: (value) => progress.push(value),
  });
  assert.deepEqual(result, stored);
  assert.equal(refreshes(), 1);
  assert.equal(requests.length, 2);
  assert.deepEqual(requests.map((request) => request.headers.get("Authorization")), ["Bearer initial", "Bearer fresh"]);
  assert.ok(requests.every((request) => request.withCredentials));
  assert.ok(requests.every((request) => request.headers.get("X-Requested-With") === "XMLHttpRequest"));
  assert.ok(requests.every((request) => request.url === "/api/documents/11/revisions"));
  assert.notEqual(requests[0].body, requests[1].body);
  assert.equal(requests[1].body.get("expected_revision_id"), "old-revision");
  assert.equal(requests[1].body.get("knowledge_base_id"), null);
  assert.equal(requests[1].body.get("confidentiality"), null);
  assert.deepEqual(progress, [0, 100, 0, 100]);
  assert.ok(client.snapshot.identity);
});

test("403 upload reports the response without retry or refresh", async (t) => {
  const { requests, client, refreshes } = await setup(t, [{ status: 403, body: { detail: "无权写入" } }]);
  await assert.rejects(uploadDocument(new File(["test"], "test.txt"), {
    knowledgeBase: "7", signal: new AbortController().signal, onProgress: () => {},
  }), { message: "无权写入" });
  assert.equal(refreshes(), 0);
  assert.equal(requests.length, 1);
  assert.ok(client.snapshot.identity);
});

test("session invalidation aborts an active XHR upload", async (t) => {
  const { requests, client } = await setup(t, []);
  const upload = uploadDocument(new File(["test"], "test.txt"), {
    knowledgeBase: "7", signal: new AbortController().signal, onProgress: () => {},
  });
  assert.equal(requests.length, 1);
  client.clearSession("已退出所有设备");
  await assert.rejects(upload, (error) => error.name === "AbortError");
  assert.equal(requests[0].aborted, true);
  assert.equal(client.snapshot.identity, null);
});

test("new upload uses department-internal classification by default and sends an explicit selection", async (t) => {
  const { requests } = await setup(t, [
    { status: 202, body: stored }, { status: 202, body: stored },
  ]);
  const options = { knowledgeBase: "7", signal: new AbortController().signal, onProgress: () => {} };
  await uploadDocument(new File(["test"], "default.txt"), options);
  await uploadDocument(new File(["test"], "classified.txt"), { ...options, confidentiality: 2 });
  assert.equal(requests[0].body.get("confidentiality"), "1");
  assert.equal(requests[1].body.get("confidentiality"), "2");
});

test("knowledge-base client requests READ for browsing and keeps WRITE as its default", async () => {
  const urls = [];
  const client = new AuthClient({ fetcher: async (url) => {
    if (url.endsWith("/select-tenant")) return response(session("initial"));
    urls.push(url);
    return response([{ id: "7", name: "只读库", description: "", permission: "READ" }]);
  } });
  await client.selectTenant("selection", "10");
  setClient(client);
  assert.equal((await fetchKnowledgeBases("READ"))[0].permission, "READ");
  await fetchKnowledgeBases();
  assert.deepEqual(urls, ["/api/knowledge-bases?access=READ", "/api/knowledge-bases?access=WRITE"]);
});
