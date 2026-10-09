<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { FileText, LoaderCircle, Search, RefreshCw, X, ShieldCheck } from "lucide-vue-next";
import { isDemoMode, type KnowledgeBaseOption } from "./services/upload";
import { searchQuery, QuerySearchError, type QuerySearchResponse, type QueryRoute } from "./services/query";
import { analysisFailureLabel, queryStatus, slowestQueryStage } from "./services/query-diagnostics";

const props = defineProps<{ bases: KnowledgeBaseOption[]; basesLoading: boolean; basesError: string }>();
const emit = defineEmits<{ reloadBases: [] }>();
const query = ref("");
const selectedBase = ref("");
const busy = ref(false);
const error = ref("");
const notice = ref("");
const result = ref<QuerySearchResponse | null>(null);
const submittedScope = ref("");
const requestElapsedMs = ref(0);
const resultStatus = computed(() => result.value ? queryStatus(result.value) : null);
const slowestStage = computed(() => result.value ? slowestQueryStage(result.value) : null);
let controller: AbortController | undefined;
let requestVersion = 0;
const canSearch = computed(() => !isDemoMode && !props.basesLoading && !props.basesError
  && props.bases.length > 0 && !!query.value.trim() && !busy.value);
const routeLabels: Record<QueryRoute, string> = {
  hybrid: "混合检索", hybrid_graph: "图谱检索", structured: "文档目录", clarify: "需要澄清", none: "无需检索",
};
const stageLabels: Record<string, string> = { succeeded: "成功", failed: "失败", skipped: "未执行", disabled: "未启用" };
const degradationLabels: Record<string, string> = {
  EMBEDDING_UNAVAILABLE: "问题向量化失败", DENSE_UNAVAILABLE: "向量检索不可用",
  BM25_UNAVAILABLE: "关键词检索不可用", RERANKER_UNAVAILABLE: "重排不可用，保留融合排序",
  SCOPE_CHANGED: "检索期间文档或权限发生变化",
};
const resultMessage = computed(() => {
  const value = result.value;
  if (!value) return "";
  if (value.plan.route === "clarify") return value.plan.params.question || value.plan.message || "请补充问题后重新检索。";
  if (value.plan.route === "none") return value.retrieval.message || "当前问题无需检索文档。";
  return value.retrieval.message || value.plan.message || (value.retrieval.status === "empty"
    ? (value.scope.published_document_count === 0 ? "当前范围内没有已发布的文档，请先上传文档并等待入库完成。" : "请尝试更具体的问题，或调整知识库范围。")
    : "");
});
const time = (ms: number) => ms >= 1000 ? `${(ms / 1000).toFixed(2)} 秒` : `${ms} 毫秒`;

function invalidate() {
  requestVersion++;
  controller?.abort();
  controller = undefined;
  busy.value = false;
  result.value = null;
  error.value = "";
}
function cancel() {
  invalidate();
  notice.value = "已取消本次检索。";
}
function reloadBases() {
  invalidate();
  notice.value = "";
  emit("reloadBases");
}
watch([query, selectedBase], () => { invalidate(); notice.value = ""; }, { flush: "sync" });
watch(() => props.bases, (bases) => {
  result.value = null;
  if (selectedBase.value && !bases.some((base) => base.id === selectedBase.value)) {
    selectedBase.value = "";
    notice.value = "原知识库已不可用，请重新确认检索范围。";
  }
}, { flush: "sync" });
onBeforeUnmount(invalidate);

async function submit() {
  if (!canSearch.value) return;
  invalidate();
  notice.value = "";
  const version = requestVersion;
  const request = new AbortController();
  controller = request;
  busy.value = true;
  const started = performance.now();
  submittedScope.value = props.bases.find((base) => base.id === selectedBase.value)?.name || "全部可读知识库";
  try {
    const response = await searchQuery(query.value, selectedBase.value, request.signal);
    if (version === requestVersion && !request.signal.aborted) {
      requestElapsedMs.value = Math.max(0, Math.round(performance.now() - started));
      result.value = response;
    }
  } catch (cause) {
    if (version !== requestVersion || request.signal.aborted) return;
    error.value = cause instanceof Error && cause.name === "TimeoutError" ? "检索等待超过 2 分钟，请稍后重试。"
      : cause instanceof Error ? cause.message : "检索失败，请检查网络后重试。";
    if (cause instanceof QuerySearchError && cause.status === 403) emit("reloadBases");
  } finally {
    if (version === requestVersion) { busy.value = false; controller = undefined; }
  }
}
</script>

<template>
  <div class="query-workspace">
    <div class="page-heading">
      <div><h1>智能问答 <span class="query-tag">检索测试</span></h1><p>输入问题，检查检索到的文档原文。当前阶段仅展示检索证据。</p></div>
    </div>
    <section class="card query-form-card" aria-labelledby="query-form-title">
      <h2 id="query-form-title">检索问题</h2>
      <p v-if="isDemoMode" class="query-message">当前为离线演示。请使用 API 模式并登录后测试真实检索。</p>
      <form @submit.prevent="submit">
        <label for="query-scope">知识库范围</label>
        <div class="query-scope-row">
          <select id="query-scope" v-model="selectedBase" :disabled="isDemoMode || basesLoading || busy || !!basesError">
            <option value="">全部可读知识库</option>
            <option v-for="base in bases" :key="base.id" :value="base.id">{{ base.name }}</option>
          </select>
          <button type="button" class="button secondary" :disabled="isDemoMode || basesLoading || busy" @click="reloadBases"><RefreshCw :size="15" />刷新知识库</button>
        </div>
        <p v-if="basesLoading" class="query-hint" role="status">正在加载可读知识库…</p>
        <p v-else-if="basesError" class="query-error" role="alert">{{ basesError }}</p>
        <p v-else-if="!isDemoMode && !bases.length" class="query-hint">当前没有可读知识库，请联系管理员授权。</p>
        <label for="query-text">你的问题</label>
        <textarea id="query-text" v-model="query" maxlength="2000" rows="4" placeholder="例如：工单关闭前需要谁确认？" @keydown.ctrl.enter.prevent="submit" @keydown.meta.enter.prevent="submit"></textarea>
        <div class="query-form-footer">
          <span class="query-hint">{{ query.length }}/2000 · Ctrl / ⌘ + Enter 检索</span>
          <div class="query-actions">
            <button v-if="busy" type="button" class="button secondary" @click="cancel"><X :size="15" />取消</button>
            <button type="submit" class="button primary" :disabled="!canSearch"><LoaderCircle v-if="busy" :size="17" class="spin" /><Search v-else :size="17" />{{ busy ? '正在检索…' : '开始检索' }}</button>
          </div>
        </div>
      </form>
      <p v-if="busy" class="query-hint" role="status" aria-live="polite">正在分析问题并检索文档，首次调用可能需要较长时间。</p>
      <p v-if="error" class="query-error" role="alert">{{ error }}</p>
      <p v-if="notice" class="query-message" role="status">{{ notice }}</p>
    </section>

    <section v-if="result" class="query-results" aria-labelledby="query-results-title" aria-live="polite">
      <div class="card query-summary">
        <div class="query-result-heading"><h2 id="query-results-title">检索结果</h2><span class="query-tag" :class="{ warning: resultStatus?.warning }">{{ resultStatus?.label }}</span></div>
        <p class="query-hint">{{ submittedScope }} · {{ routeLabels[result.plan.route] }} · {{ result.retrieval.evidence.length }} 条原文证据</p>
        <div class="query-timings"><span>本次等待 <strong>{{ time(requestElapsedMs) }}</strong></span><span>问题分析 <strong>{{ time(result.analysis.latency_ms) }}</strong></span><span>检索执行 <strong>{{ time(result.retrieval.trace.total_ms) }}</strong></span></div>
        <p v-if="slowestStage" class="query-hint">耗时最多的已记录阶段：{{ slowestStage.label }}（{{ time(slowestStage.ms) }}）。</p>
        <p v-if="resultMessage" class="query-message">{{ resultMessage }}</p>
        <div v-if="result.analysis.status === 'degraded'" class="query-message warning"><strong>问题分析已降级：{{ analysisFailureLabel(result.analysis.degradation_reason) }}</strong><p v-if="result.analysis.message">{{ result.analysis.message }}</p></div>
        <p v-for="code in result.retrieval.degradation_codes" :key="code" class="query-message warning">{{ degradationLabels[code] || code }}</p>
        <p v-if="result.plan.requested_route !== result.plan.route" class="query-message warning">计划使用{{ routeLabels[result.plan.requested_route] }}，本次实际使用{{ routeLabels[result.plan.route] }}。</p>
        <details class="query-details">
          <summary>查看检索详情</summary>
          <dl class="query-trace">
            <div><dt>问题分析状态</dt><dd>{{ result.analysis.status === 'ok' ? '成功' : '已降级' }} · {{ time(result.analysis.latency_ms) }}</dd></div>
            <div v-if="result.analysis.status === 'degraded'"><dt>分析降级原因码</dt><dd>{{ result.analysis.degradation_reason || '未提供' }}</dd></div>
            <div v-if="result.plan.params.candidate_k !== undefined"><dt>计划候选 / 最终上限</dt><dd>{{ result.plan.params.candidate_k }} / {{ result.plan.params.top_k }} · {{ result.plan.params.rerank ? '启用重排' : '关闭重排' }}</dd></div>
            <div><dt>可读 / 已发布文档</dt><dd>{{ result.scope.authorized_document_count }} / {{ result.scope.published_document_count }}</dd></div>
            <div><dt>向量检索</dt><dd>{{ stageLabels[result.retrieval.trace.dense_status] }} · {{ result.retrieval.trace.dense_count }} 条 · {{ time(result.retrieval.trace.dense_ms) }}</dd></div>
            <div><dt>关键词检索</dt><dd>{{ stageLabels[result.retrieval.trace.bm25_status] }} · {{ result.retrieval.trace.bm25_count }} 条 · {{ time(result.retrieval.trace.bm25_ms) }}</dd></div>
            <div><dt>融合 / 授权后候选</dt><dd>{{ result.retrieval.trace.fused_count }} / {{ result.retrieval.trace.authorized_count }}</dd></div>
            <div><dt>重排</dt><dd>{{ stageLabels[result.retrieval.trace.rerank_status] }} · {{ time(result.retrieval.trace.rerank_ms) }}</dd></div>
            <div><dt>问题向量化</dt><dd>{{ time(result.retrieval.trace.embedding_ms) }}</dd></div>
            <div><dt>融合 / 授权校验耗时</dt><dd>{{ time(result.retrieval.trace.fusion_ms) }} / {{ time(result.retrieval.trace.authorization_ms) }}</dd></div>
            <div><dt>路由原因</dt><dd>{{ result.plan.reason_code }}</dd></div>
          </dl>
          <p v-if="result.analysis.analysis" class="query-hint">问题意图：{{ result.analysis.analysis.intent }}</p>
          <p v-for="warning in result.plan.warnings" :key="warning" class="query-hint">{{ warning }}</p>
          <p class="query-hint">本次等待包括网络、身份与范围校验及分析和检索。召回分支并行执行，各阶段耗时不能直接相加；最耗时阶段用于定位等待位置。</p>
        </details>
      </div>
      <article v-for="item in result.retrieval.evidence" :key="item.chunk_id" class="card query-evidence">
        <div class="query-evidence-heading"><span class="query-citation">{{ item.citation_id }}</span><FileText :size="18" /><h3>{{ item.document_title || item.document_name }}</h3><span class="query-hint">第 {{ item.rank }} 条</span></div>
        <p class="query-source">{{ item.document_name }}<span v-if="item.page_number !== null"> · 第 {{ item.page_number }} 页</span><span v-if="item.section_title"> · {{ item.section_title }}</span> · 分块 {{ item.chunk_index + 1 }}</p>
        <div class="query-original">{{ item.content }}</div>
      </article>
    </section>
    <div v-else-if="!busy && !error" class="query-empty"><Search :size="30" /><p>检索结果会显示在这里</p><span>建议先用已入库文档中的具体问题进行测试。</span></div>
    <footer class="page-footer"><span>知擎 · 检索测试</span><span><ShieldCheck :size="13" />按当前账号权限检索已发布文档</span></footer>
  </div>
</template>

<style scoped>
.query-workspace { max-width: 1120px; margin: 0 auto; }
.query-form-card, .query-summary, .query-evidence { padding: 24px; }
h2 { font-size: 17px; margin: 0 0 18px; }
.query-tag { display: inline-flex; align-items: center; vertical-align: middle; font-size: 12px; font-weight: 500; padding: 5px 9px; border-radius: 6px; background: #e8f5ef; color: #167c70; }
h1 .query-tag { margin-left: 8px; }
.query-form-card label { display: block; font-size: 14px; font-weight: 600; margin: 18px 0 9px; }
.query-scope-row, .query-actions, .query-form-footer, .query-result-heading, .query-evidence-heading { display: flex; align-items: center; gap: 12px; }
.query-scope-row select, textarea { border: 1px solid #dce3df; border-radius: 8px; padding: 12px; background: #fff; color: #243c34; font: inherit; }
.query-scope-row select { flex: 1; min-width: 0; }
textarea { display: block; width: 100%; resize: vertical; min-height: 120px; box-sizing: border-box; line-height: 1.7; }
textarea:focus, select:focus { outline: 2px solid #167c70; outline-offset: 2px; }
.query-form-footer { justify-content: space-between; margin-top: 14px; flex-wrap: wrap; }
.query-hint, .query-source { color: #738079; font-size: 13px; line-height: 1.7; }
.query-error { color: #b33737; font-size: 14px; line-height: 1.7; overflow-wrap: anywhere; }
.query-message { margin: 10px 0; padding: 12px 14px; background: #f3f6f4; border-radius: 8px; font-size: 14px; line-height: 1.7; overflow-wrap: anywhere; }
.query-message p { margin-top: 6px; }
.warning { color: #925b17; background: #fff4df; }
.query-results { display: grid; gap: 16px; margin-top: 24px; }
.query-result-heading { justify-content: space-between; flex-wrap: wrap; }
.query-result-heading h2 { margin: 0; }
.query-timings { display: flex; gap: 24px; flex-wrap: wrap; font-size: 13px; color: #738079; margin: 18px 0; }
.query-timings strong { color: #243c34; margin-left: 8px; }
.query-details { border-top: 1px solid #e8ede9; padding-top: 16px; margin-top: 18px; font-size: 13px; }
.query-details summary { cursor: pointer; color: #167c70; }
.query-trace { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; }
.query-trace dt { color: #738079; margin-bottom: 5px; }
.query-trace dd { margin: 0; overflow-wrap: anywhere; }
.query-citation { font-size: 12px; color: #167c70; background: #e8f5ef; border-radius: 5px; padding: 5px 8px; }
.query-evidence-heading { color: #167c70; flex-wrap: wrap; }
.query-evidence-heading h3 { color: #243c34; margin: 0; font-size: 15px; flex: 1; min-width: 100px; overflow-wrap: anywhere; }
.query-source { overflow-wrap: anywhere; }
.query-original { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.85; font-size: 14px; border-top: 1px solid #e8ede9; padding-top: 16px; }
.query-empty { text-align: center; color: #8b9790; padding: 60px 16px; }
.query-empty p { color: #53675c; font-size: 15px; }
.query-empty span { font-size: 13px; }
@media (max-width: 640px) {
  .query-form-card, .query-summary, .query-evidence { padding: 18px; }
  .query-scope-row { flex-wrap: wrap; }
  .query-scope-row select { flex-basis: 100%; }
  .query-trace { grid-template-columns: 1fr; }
  .query-form-footer .query-actions { width: 100%; justify-content: flex-end; }
}
</style>
