import type { QuerySearchResponse, QueryAnswerResponse, GraphPathEvidence, QueryEvidence } from "./query";

const predicateLabels: Record<string, string> = {
  HAS_STEP: "包含步骤", NEXT_STEP: "下一步骤", RESPONSIBLE_FOR: "负责办理",
  REVIEWS: "审核或审批", CONFIRMS: "确认结果", APPLIES_TO: "适用于",
  REQUIRES_MATERIAL: "需要材料", PRODUCES_MATERIAL: "形成材料", OPERATES_ON: "操作对象",
  HAS_FIELD: "记录字段", HAS_STATE: "定义状态", GOVERNED_BY: "受规则约束",
  DEPENDS_ON: "依赖前提", ESCALATES_TO: "升级至", TRANSITIONS_TO: "状态转移",
};
const bridgeLabels = {
  explicit_document: "文档明确对应", shared_context: "共同上下文确认", explicit_alias: "原文明示别名",
};

export interface GraphPathView {
  id: string;
  steps: { subject: string; predicate: string; object: string; condition: string;
    quote: string; citation: string; document: string }[];
  bridges: { name: string; basis: string; sources: { citation: string; quote: string; document: string }[] }[];
}

/** Call only with validatedGraphPaths; the view contains readable facts and source links. */
export function graphPathViews(paths: GraphPathEvidence[], evidence: QueryEvidence[]): GraphPathView[] {
  const byChunk = new Map(evidence.map((item) => [item.chunk_id, item]));
  return paths.flatMap((path) => {
    const steps = path.hops.map((hop) => {
      const item = byChunk.get(hop.source.chunk_id), predicate = predicateLabels[hop.relation.predicate];
      if (!item || !predicate || !/^E[1-9][0-9]*$/.test(item.citation_id)) return null;
      return { subject: hop.subject_name || "文档中的对象", predicate, object: hop.object_name || "关联对象",
        condition: hop.relation.condition_text, quote: hop.source.source_quote, citation: item.citation_id,
        document: item.document_title || item.document_name };
    });
    const bridges = path.bridges.map((bridge) => {
      const basis = bridgeLabels[bridge.basis];
      const sources = bridge.sources.map((source) => {
        const item = byChunk.get(source.chunk_id);
        return item && /^E[1-9][0-9]*$/.test(item.citation_id)
          ? { citation: item.citation_id, quote: source.source_quote, document: item.document_title || item.document_name } : null;
      });
      if (!basis || sources.some((source) => source === null)) return null;
      return { name: bridge.canonical_name, basis, sources: sources.filter((source) => source !== null) };
    });
    if (steps.length !== 2 || steps.some((step) => step === null) || bridges.some((bridge) => bridge === null)) return [];
    return [{ id: path.path_id, steps: steps.filter((step) => step !== null), bridges: bridges.filter((bridge) => bridge !== null) }];
  });
}

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
  if (result.structured) return { label: "目录查询完成", warning: false };
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

export function slowestQueryStage(result: QuerySearchResponse | QueryAnswerResponse): { label: string; ms: number } | null {
  const trace = result.retrieval.trace;
  const stages = [
    { label: "问题分析", ms: result.analysis.latency_ms },
    { label: "问题向量化", ms: trace.embedding_ms },
    { label: "向量检索", ms: trace.dense_ms },
    { label: "关键词检索", ms: trace.bm25_ms },
    { label: "图关系召回", ms: trace.graph_ms ?? 0 },
    { label: "候选融合", ms: trace.fusion_ms },
    { label: "授权校验", ms: trace.authorization_ms },
    { label: "重排", ms: trace.rerank_ms },
  ];
  if ("answer" in result) stages.push(
    { label: "回答生成", ms: result.answer.trace.generation_ms },
    { label: "回答证据复核", ms: result.answer.trace.review_ms },
    { label: "回答授权校验", ms: result.answer.trace.authorization_ms },
  );
  const slowest = stages.reduce((max, stage) => stage.ms > max.ms ? stage : max);
  return slowest.ms > 0 ? slowest : null;
}
