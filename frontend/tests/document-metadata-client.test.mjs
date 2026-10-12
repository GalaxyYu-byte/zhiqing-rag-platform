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
const dependencyUrl = moduleUrl(`
  export const authenticatedFetch = (url, options) => globalThis.metadataFetch(url, options);
  export const authClient = { authorizedRequest: (send, _decode, signal) => send("test", signal) };
`);
const source = (await readFile(new URL("../src/services/upload.ts", import.meta.url), "utf8"))
  .replace('"./auth"', JSON.stringify(dependencyUrl)).replaceAll("import.meta.env", "({})");
const { fetchDocumentMetadata, updateDocumentMetadata, uploadDocument } = await import(moduleUrl(source));
const values = { document_code: "A-001", business_status: "EFFECTIVE", event_date: "2026-10-01" };

test("metadata editing preserves bigint IDs and optimistic version in JSON", async (t) => {
  const requests = [];
  globalThis.metadataFetch = async (url, options) => {
    requests.push({ url, options });
    return Response.json({ document_id: "9007199254740993", ...values });
  };
  t.after(() => { delete globalThis.metadataFetch; });
  const signal = new AbortController().signal;
  const id = "9007199254740993";
  await fetchDocumentMetadata(id, signal);
  await updateDocumentMetadata(id, { ...values, title: "采购制度", expected_updated_at: "2026-10-11T01:00:00Z" }, signal);
  assert.ok(requests.every(({ url }) => url === `/api/documents/${id}/metadata`));
  assert.equal(requests[1].options.method, "PATCH");
  assert.equal(requests[1].options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(requests[1].options.body), {
    ...values, title: "采购制度", expected_updated_at: "2026-10-11T01:00:00Z",
  });
});

test("metadata conflicts display server's actionable error", async (t) => {
  globalThis.metadataFetch = async () => Response.json({ detail: {
    code: "METADATA_CONFLICT", message: "文档属性已变更，请重新加载后重试",
  } }, { status: 409 });
  t.after(() => { delete globalThis.metadataFetch; });
  await assert.rejects(updateDocumentMetadata("11", {
    ...values, title: "采购制度", expected_updated_at: "2026-10-11T01:00:00Z",
  }, new AbortController().signal), { message: "文档属性已变更，请重新加载后重试" });
});

test("upload sends each file's metadata and revision updates preserve existing metadata", async (t) => {
  const original = globalThis.XMLHttpRequest;
  const requests = [];
  globalThis.XMLHttpRequest = class {
    upload = {};
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader() {}
    send(body) {
      requests.push({ url: this.url, body });
      this.status = 202;
      this.response = { status: "STORED", processing_status: "PENDING", document_id: "11", task_id: "14" };
      queueMicrotask(() => this.onload());
    }
  };
  t.after(() => { globalThis.XMLHttpRequest = original; });
  const options = { knowledgeBase: "7", signal: new AbortController().signal, onProgress: () => {}, metadata: values };
  await uploadDocument(new File(["test"], "test.txt"), options);
  await uploadDocument(new File(["changed"], "test.txt"), { ...options, documentId: "11", expectedRevisionId: "12" });
  assert.equal(requests[0].body.get("document_code"), "A-001");
  assert.equal(requests[0].body.get("business_status"), "EFFECTIVE");
  assert.equal(requests[0].body.get("event_date"), "2026-10-01");
  assert.equal(requests[1].body.get("document_code"), null);
  assert.equal(requests[1].body.get("business_status"), null);
  assert.equal(requests[1].body.get("event_date"), null);
  assert.equal(requests[1].body.get("expected_revision_id"), "12");
});
