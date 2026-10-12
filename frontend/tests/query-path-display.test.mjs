import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { compileTemplate, parse } from "@vue/compiler-sfc";
import { createRenderer } from "vue";
import ts from "typescript";
import { graphPathFixture } from "./helpers/graph-path-fixture.mjs";
import { queryFixture } from "./helpers/query-fixture.mjs";

const moduleUrl = (source) => `data:text/javascript;base64,${Buffer.from(ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText).toString("base64")}`;
const dependencies = moduleUrl(`export const authenticatedFetch = () => { throw new Error('No network in display tests'); };
  export const authClient = { clearSession() {} }; export const responseCode = () => ''; export const responseMessage = () => '';`);
const querySource = (await readFile(new URL("../src/services/query.ts", import.meta.url), "utf8"))
  .replace('"./auth"', JSON.stringify(dependencies)).replace('"./auth-client"', JSON.stringify(dependencies))
  .replaceAll("import.meta.env", "({})");
const { validatedGraphPaths } = await import(moduleUrl(querySource));
const { graphPathViews } = await import(moduleUrl(await readFile(new URL("../src/services/query-diagnostics.ts", import.meta.url), "utf8")));
const { descriptor } = parse(await readFile(new URL("../src/QueryWorkspace.vue", import.meta.url), "utf8"));
const template = descriptor.template.content;
const start = template.indexOf('<section v-if="panel.paths.length"');
assert.ok(start >= 0);
const fragment = template.slice(start, template.indexOf("</section>", start) + "</section>".length);
const compiled = compileTemplate({ source: fragment, filename: "QueryWorkspace.vue", id: "graph-path-display" });
assert.deepEqual(compiled.errors, []);
const { render } = await import(moduleUrl(compiled.code.replace(/from ["']vue["']/g,
  `from ${JSON.stringify(import.meta.resolve("vue"))}`)));
const renderer = createRenderer({
  createElement: (type) => ({ type, props: {}, children: [] }),
  createText: (text) => ({ type: "text", text, children: [] }),
  createComment: (text) => ({ type: "comment", text, children: [] }),
  insert: (child, parent) => { parent.children.push(child); child.parent = parent; },
  remove: (child) => { if (child.parent) child.parent.children = child.parent.children.filter((item) => item !== child); },
  setText: (node, text) => { node.text = text; },
  setElementText: (node, text) => { node.text = text; node.children = []; },
  patchProp: (node, key, previous, value) => { node.props[key] = value; },
  parentNode: (node) => node.parent, nextSibling: () => null,
});
function mount(result) {
  const root = { children: [] };
  const panel = { id: "run-1", paths: graphPathViews(validatedGraphPaths(result), result.retrieval.evidence) };
  const app = renderer.createApp({ setup: () => ({ panel }), render });
  app.mount(root);
  const walk = (node) => [node, ...(node.children ?? []).flatMap(walk)];
  const nodes = walk(root);
  return { app, nodes, text: nodes.filter((node) => node.type !== "comment").map((node) => node.text ?? "").join(" ") };
}

test("actual path template renders both business assertions, condition and exact raw quotes", () => {
  const result = graphPathFixture({ astral: true });
  const { app, nodes, text } = mount(result);
  try {
    assert.match(text, /处理步骤/); assert.match(text, /形成材料/); assert.match(text, /负责人确认结论/);
    assert.match(text, /适用条件：关闭前/);
    assert.deepEqual(nodes.filter((node) => node.type === "a").map((node) => node.props.href),
      ["#query-evidence-run-1-E1", "#query-evidence-run-1-E2"]);
    assert.equal(nodes.find((node) => node.type === "blockquote").text, result.retrieval.evidence[0].content);
    assert.equal(nodes.filter((node) => node.type === "script").length, 0);
    assert.doesNotMatch(text, /private_projection|assertion-1|material-1|PRODUCES_MATERIAL/);
  } finally { app.unmount(); }
});

test("cross-document bridge displays its current original-source citation links", () => {
  const { app, nodes, text } = mount(graphPathFixture({ bridge: true }));
  try {
    assert.match(text, /对象对应：关闭摘要/); assert.match(text, /原文明示别名/);
    assert.deepEqual(nodes.filter((node) => node.type === "a").map((node) => node.props.href),
      ["#query-evidence-run-1-E1", "#query-evidence-run-1-E1", "#query-evidence-run-1-E3", "#query-evidence-run-1-E2"]);
  } finally { app.unmount(); }
});

test("invalid or partially missing paths produce no credible display or citation links", () => {
  for (const corruption of ["quote", "source", "scope"]) {
    const result = graphPathFixture({ bridge: true });
    if (corruption === "quote") result.retrieval.graph_paths[0].hops[0].source.quote_end++;
    if (corruption === "source") result.retrieval.evidence.pop();
    if (corruption === "scope") result.retrieval.graph_paths[0].bridges[0].sources[1].tenant_id = "99";
    const { app, nodes, text } = mount(result);
    try { assert.equal(nodes.filter((node) => node.type === "a").length, 0); assert.doesNotMatch(text, /关闭摘要|路径 1/); }
    finally { app.unmount(); }
  }
});

test("old history without graph_paths renders without a path section", () => {
  const { app, nodes } = mount(queryFixture());
  try { assert.equal(nodes.filter((node) => node.type === "section").length, 0); }
  finally { app.unmount(); }
});

test("shared UID without a bridge cannot connect different projections of the same document", () => {
  for (const key of ["namespace", "generation_id"]) {
    const result = graphPathFixture();
    const source = result.retrieval.graph_paths[0].hops[1].source;
    source[key] = key === "namespace" ? "other_projection" : "999";
    const projection = result.plan.params.projections.find((row) => row.projection_id === source.projection_id);
    projection[key] = source[key];
    if (key === "generation_id") result.retrieval.evidence[1].generation_id = source[key];
    assert.deepEqual(validatedGraphPaths(result), []);
  }
});
