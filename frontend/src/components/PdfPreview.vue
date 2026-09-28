<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { ChevronLeft, ChevronRight, LoaderCircle } from "lucide-vue-next";
import { getDocument, GlobalWorkerOptions, type PDFDocumentProxy, type RenderTask } from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = workerUrl;
const props = defineProps<{ url: string; name: string }>();
const container = ref<HTMLDivElement>();
const canvas = ref<HTMLCanvasElement>();
const pageNumber = ref(1);
const pageCount = ref(0);
const loading = ref(true);
const error = ref("");
let pdf: PDFDocumentProxy | undefined;
let task: ReturnType<typeof getDocument> | undefined;
let rendering: RenderTask | undefined;
let disposed = false;

async function renderPage(number: number) {
  if (!pdf || disposed) return;
  loading.value = true;
  error.value = "";
  try {
    const page = await pdf.getPage(number);
    if (disposed || !canvas.value) return;
    const target = canvas.value;
    const width = Math.max(240, Math.min(1000, (container.value?.clientWidth || 800) - 32));
    const viewport = page.getViewport({ scale: width / page.getViewport({ scale: 1 }).width });
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    target.width = Math.floor(viewport.width * ratio);
    target.height = Math.floor(viewport.height * ratio);
    target.style.width = `${viewport.width}px`;
    target.style.height = `${viewport.height}px`;
    rendering = page.render({
      canvas: target,
      viewport,
      transform: ratio === 1 ? undefined : [ratio, 0, 0, ratio, 0, 0],
    });
    await rendering.promise;
    if (!disposed) pageNumber.value = number;
  } catch {
    if (!disposed) error.value = "PDF 页面显示失败，请下载原文件查看。";
  } finally {
    if (!disposed) loading.value = false;
  }
}

onMounted(async () => {
  try {
    task = getDocument({ url: props.url });
    pdf = await task.promise;
    if (disposed) return;
    pageCount.value = pdf.numPages;
    await renderPage(1);
  } catch {
    if (!disposed) {
      error.value = "PDF 预览加载失败，文件可能已加密或损坏，请下载原文件查看。";
      loading.value = false;
    }
  }
});
onBeforeUnmount(() => {
  disposed = true;
  rendering?.cancel();
  void task?.destroy();
});
</script>

<template>
  <div class="pdf-viewer">
    <div class="pdf-toolbar">
      <button class="button secondary" :disabled="loading || pageNumber <= 1" @click="renderPage(pageNumber - 1)">
        <ChevronLeft :size="16" />上一页
      </button>
      <span aria-live="polite">第 {{ pageNumber }} / {{ pageCount }} 页</span>
      <button class="button secondary" :disabled="loading || pageNumber >= pageCount" @click="renderPage(pageNumber + 1)">
        下一页<ChevronRight :size="16" />
      </button>
    </div>
    <div v-if="loading" class="pdf-loading" role="status"><LoaderCircle :size="18" class="spin" />正在渲染页面…</div>
    <p v-if="error" role="alert">{{ error }}</p>
    <div ref="container" class="pdf-pages">
      <canvas ref="canvas" role="img" :aria-label="`${name} 第 ${pageNumber} 页`"></canvas>
    </div>
  </div>
</template>

<style scoped>
.pdf-toolbar {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
  font-size: 12px;
  color: #60786d;
}
.pdf-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 12px;
  color: #81968b;
}
.pdf-pages {
  max-height: 60vh;
  overflow: auto;
  padding: 16px;
  background: #eef2f0;
  border-radius: 8px;
}
canvas {
  display: block;
  margin: auto;
  max-width: 100%;
  height: auto !important;
  box-shadow: 0 2px 10px #29383515;
}
</style>
