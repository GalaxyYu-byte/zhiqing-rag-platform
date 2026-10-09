import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import { queryFixture } from "./helpers/query-fixture.mjs";

const source = await readFile(new URL("../src/services/query-diagnostics.ts", import.meta.url), "utf8");
const output = ts.transpileModule(source, { compilerOptions: {
  target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext,
} }).outputText;
const { analysisFailureLabel, queryStatus, slowestQueryStage } = await import(
  `data:text/javascript;base64,${Buffer.from(output).toString("base64")}`,
);

test("analysis failure remains visible when retrieval itself succeeds", () => {
  const result = queryFixture({ analysis: { status: "degraded", degradation_reason: "invalid_response" } });
  assert.deepEqual(queryStatus(result), { label: "分析降级 · 检索完成", warning: true });
  assert.deepEqual(queryStatus(queryFixture()), { label: "检索完成", warning: false });
});

test("analysis failure does not conceal skipped, empty, unsupported or degraded retrieval", () => {
  for (const [status, label] of [["skipped", "未执行检索"], ["empty", "没有找到原文"], ["unsupported", "暂不支持"], ["degraded", "检索完成 · 有降级"]]) {
    const result = queryFixture({ analysis: { status: "degraded" }, retrieval: { status } });
    assert.deepEqual(queryStatus(result), { label: `分析降级 · ${label}`, warning: true });
  }
});

test("route fallback and warnings prevent an entirely green success state", () => {
  assert.deepEqual(queryStatus(queryFixture({ plan: { requested_route: "hybrid_graph" } })), { label: "检索完成 · 路由回退", warning: true });
  assert.deepEqual(queryStatus(queryFixture({ plan: { warnings: ["GRAPH_NOT_READY"] } })), { label: "检索完成 · 有提示", warning: true });
});

test("all analyzer failure reasons have distinct explanations without assuming timeout", () => {
  const reasons = ["disabled", "not_configured", "timeout", "model_unavailable", "invalid_response", "invalid_grounding", "incomplete_response"];
  assert.equal(new Set(reasons.map(analysisFailureLabel)).size, reasons.length);
  assert.match(analysisFailureLabel("timeout"), /超时/);
  assert.doesNotMatch(analysisFailureLabel("invalid_response"), /超时/);
  assert.match(analysisFailureLabel("new_reason"), /new_reason/);
  assert.equal(analysisFailureLabel(null), "未提供具体原因");
});

test("slowest recorded stage includes analysis and excludes aggregate retrieval time", () => {
  const result = queryFixture({ analysis: { latency_ms: 10230 }, retrieval: { trace: { ...queryFixture().retrieval.trace, total_ms: 9610, rerank_ms: 8000 } } });
  assert.deepEqual(slowestQueryStage(result), { label: "问题分析", ms: 10230 });
  result.analysis.latency_ms = 100;
  assert.deepEqual(slowestQueryStage(result), { label: "重排", ms: 8000 });
  result.retrieval.trace.bm25_ms = 9000;
  assert.deepEqual(slowestQueryStage(result), { label: "关键词检索", ms: 9000 });
});

test("skipped zero-duration work does not invent a slow stage", () => {
  const result = queryFixture({ analysis: { latency_ms: 0 } });
  for (const key of Object.keys(result.retrieval.trace).filter((key) => key.endsWith("_ms"))) result.retrieval.trace[key] = 0;
  assert.equal(slowestQueryStage(result), null);
});
