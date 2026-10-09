import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const moduleUrl = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText).toString("base64")}`;
const authUrl = moduleUrl(await readFile(new URL("../src/services/auth-client.ts", import.meta.url), "utf8"));
const { AuthClient } = await import(authUrl);
const dependencyUrl = moduleUrl(`
export let client;
export function setClient(value) { client = value; }
export const authenticatedFetch = (url, options) => client.fetch(url, options);
`);
const { setClient } = await import(dependencyUrl);
const source = (await readFile(new URL("../src/services/permissions.ts", import.meta.url), "utf8"))
  .replace('"./auth"', JSON.stringify(dependencyUrl)).replace('"./auth-client"', JSON.stringify(authUrl))
  .replaceAll("import.meta.env", "({})");
const { fetchPermissions, setPermission, revokePermission, setDocumentClassification, PermissionError } = await import(moduleUrl(source));
const json = (body, status = 200) => new Response(JSON.stringify(body), { status });
const view = {
  document: { id: "2", title: "文档", department_id: "3", confidentiality: 1 },
  grants: [{ id: "4", user_id: "5", department_id: null, permission: "READ", subject_name: "用户" }],
  subjects: { users: [{ id: "5", email: "user@example.com", display_name: "用户", department_id: "3", clearance: 1 }], departments: [{ id: "3", name: "部门" }] },
};
async function setup(handler = () => json(view)) {
  const calls = [];
  const client = new AuthClient({ fetcher: async (url, init) => {
    if (url.endsWith("/select-tenant")) return json({ access_token: "token", user: { id: "1", email: "admin@example.com", display_name: "管理人" }, tenant: { id: "10", name: "租户", code: "TEST", role: "ADMIN" } });
    calls.push({ url, init });
    return handler(url, init);
  } });
  await client.selectTenant("selection", "10");
  setClient(client);
  return { client, calls, signal: new AbortController().signal };
}

test("permission management uses authenticated read, user/department upsert and revoke endpoints", async () => {
  const { calls, signal } = await setup();
  const target = { kind: "knowledge-base", id: "库/id", name: "知识库" };
  assert.equal((await fetchPermissions(target, signal)).grants[0].id, "4");
  await setPermission(target, { user_id: "5" }, "WRITE", signal);
  await setPermission(target, { department_id: "3" }, "READ", signal);
  await revokePermission(target, "授权/id", signal);
  assert.deepEqual(calls.map(({ url, init }) => [url, init.method]), [
    ["/api/knowledge-bases/%E5%BA%93%2Fid/permissions", "GET"],
    ["/api/knowledge-bases/%E5%BA%93%2Fid/permissions", "POST"],
    ["/api/knowledge-bases/%E5%BA%93%2Fid/permissions", "POST"],
    ["/api/knowledge-bases/%E5%BA%93%2Fid/permissions/%E6%8E%88%E6%9D%83%2Fid", "DELETE"],
  ]);
  assert.deepEqual(JSON.parse(calls[1].init.body), { user_id: "5", permission: "WRITE" });
  assert.deepEqual(JSON.parse(calls[2].init.body), { department_id: "3", permission: "READ" });
  assert.ok(calls.every(({ init }) => init.credentials === "include" && init.headers.get("Authorization") === "Bearer token"));
  assert.ok(calls.slice(1).every(({ init }) => init.headers.get("X-Requested-With") === "XMLHttpRequest"));
});

test("document sharing and classification use separate endpoints with complete returned views", async () => {
  const { calls, signal } = await setup();
  const target = { kind: "document", id: "2", name: "文档" };
  await setPermission(target, { department_id: "3" }, "WRITE", signal);
  const result = await setDocumentClassification("2", 2, signal);
  assert.equal(result.document.title, "文档");
  assert.equal(calls[0].url, "/api/documents/2/permissions");
  assert.equal(calls[1].url, "/api/documents/2/classification");
  assert.deepEqual(JSON.parse(calls[1].init.body), { confidentiality: 2 });
});

test("last-admin conflict and clearance denial preserve identity and expose actionable server errors", async () => {
  const { client, signal } = await setup((url) => url.endsWith("/classification")
    ? json({ detail: { code: "CLEARANCE_EXCEEDED", message: "超出当前账号密级" } }, 403)
    : json({ detail: { code: "LAST_ADMIN_REQUIRED", message: "需要保留最后一位管理员" } }, 409));
  await assert.rejects(revokePermission({ kind: "knowledge-base", id: "1", name: "库" }, "4", signal),
    (error) => error instanceof PermissionError && error.status === 409 && error.code === "LAST_ADMIN_REQUIRED");
  await assert.rejects(setDocumentClassification("2", 2, signal),
    (error) => error.status === 403 && error.message === "超出当前账号密级");
  assert.ok(client.snapshot.identity);
});

test("closing a permissions dialog cancels reads before issuing a request", async () => {
  const { calls } = await setup();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(fetchPermissions({ kind: "document", id: "2", name: "文档" }, controller.signal),
    (error) => error.name === "AbortError");
  assert.equal(calls.length, 0);
});
