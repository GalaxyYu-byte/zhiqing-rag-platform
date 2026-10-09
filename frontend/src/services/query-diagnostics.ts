import type { QuerySearchResponse } from "./query";

const retrievalLabels = {
  ok: "检索完成", degraded: "检索完成 · 有降级", empty: "没有找到原文",
  skipped: "未执行检索", unsupported: "暂不支持",
};
const analysisReasons: Record<string, string> = {
  disabled: "问题分析器未启用",
  not_configured: "分析模型凭据未配置",
  timeout: "问题分析超时",
  model_unavailable: "分析模型调用失败，请检查网络、凭据或模型服务",
  invalid_response: "模型输出不符合分析契约",
  invalid_grounding: "模型抽取的信息无法在输入原文中找到依据",
  incomplete_response: "模型输出未完整结束",
};

export function analysisFailureLabel(reason: string | null): string {
  return reason ? analysisReasons[reason] || `未知分析故障（${reason}）` : "未提供具体原因";
}

export function queryStatus(result: QuerySearchResponse) {
  const { analysis, retrieval, plan } = result;
  const routeFallback = plan.requested_route !== plan.route;
  let label = retrievalLabels[retrieval.status];
  if (analysis.status === "degraded") label = `分析降级 · ${label}`;
  else if (retrieval.status === "ok" && routeFallback) label += " · 路由回退";
  else if (retrieval.status === "ok" && (retrieval.degradation_codes.length || plan.warnings.length)) label += " · 有提示";
  return {
    label,
    warning: analysis.status === "degraded" || retrieval.status !== "ok" || routeFallback
      || retrieval.degradation_codes.length > 0 || plan.warnings.length > 0,
  };
}

export function slowestQueryStage(result: QuerySearchResponse): { label: string; ms: number } | null {
  const trace = result.retrieval.trace;
  const stages = [
    { label: "问题分析", ms: result.analysis.latency_ms },
    { label: "问题向量化", ms: trace.embedding_ms },
    { label: "向量检索", ms: trace.dense_ms },
    { label: "关键词检索", ms: trace.bm25_ms },
    { label: "候选融合", ms: trace.fusion_ms },
    { label: "授权校验", ms: trace.authorization_ms },
    { label: "重排", ms: trace.rerank_ms },
  ];
  const slowest = stages.reduce((max, stage) => stage.ms > max.ms ? stage : max);
  return slowest.ms > 0 ? slowest : null;
}
