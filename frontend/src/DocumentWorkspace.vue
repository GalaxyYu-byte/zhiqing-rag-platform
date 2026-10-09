<script setup lang="ts">
import { computed, defineAsyncComponent, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  CloudUpload,
  FileText,
  FolderOpen,
  Info,
  Layers3,
  LoaderCircle,
  LogOut,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
  X,
} from "lucide-vue-next";
import {
  allowedExtensions,
  fetchKnowledgeBases,
  fetchUploadSettings,
  fetchDocumentTasks,
  fetchDocumentRevisions,
  fetchDocumentTask,
  retryDocumentTask,
  deleteDocument,
  fetchOriginalDocument,
  fetchDocumentPreview,
  type DocumentPreview,
  isDemoMode,
  maxFileSize,
  uploadDocument,
  type KnowledgeBaseOption,
  type UploadResult,
  type DocumentTaskStatus,
  type DocumentTaskPage,
} from "./services/upload";
import type { AuthSnapshot } from "./services/auth-client";
import { authClient } from "./services/auth";
import PermissionDialog from "./components/PermissionDialog.vue";
import QueryWorkspace from "./QueryWorkspace.vue";
import { classificationLabels, type PermissionTarget } from "./services/permissions";

defineProps<{
  identity?: AuthSnapshot["identity"];
  loggingOut?: boolean;
  logoutError?: string;
}>();
const emit = defineEmits<{ logout: [] }>();

const PdfPreview = defineAsyncComponent(() => import("./components/PdfPreview.vue"));
const activeView = ref<"documents" | "query">("documents");
const queryAccessVersion = ref(0);

type Status =
  | "pending"
  | "uploading"
  | "queued"
  | "processing"
  | "process_error"
  | "done"
  | "error"
  | "cancelled";
interface DocumentItem {
  id: string;
  file: File | null;
  name: string;
  size: number;
  task?: DocumentTaskStatus;
  retrying?: boolean;
  updating?: boolean;
  removing?: boolean;
  pollError?: string;
  extension: string;
  status: Status;
  progress: number;
  error?: string;
  selected: boolean;
  knowledgeBase: string;
  result?: UploadResult;
  createdAt: string;
}

const documents = ref<DocumentItem[]>([]);
const historyDocuments = ref<DocumentItem[]>([]);
const historyPage = ref<DocumentTaskPage>({ items: [], total: 0, overall_total: 0, overall_size: 0, counts: {} });
const historyLoading = ref(false);
const historyError = ref("");
const expandedDocumentId = ref<string | null>(null);
const revisionHistory = ref<DocumentItem[]>([]);
const revisionHistoryLoading = ref(false);
const revisionHistoryError = ref("");
let revisionHistoryController: AbortController | undefined;
const currentPage = ref(1);
const pageDraft = ref("1");
const pageSize = ref(10);
const dateFrom = ref("");
const dateTo = ref("");
const dateError = computed(() => dateFrom.value && dateTo.value && dateFrom.value > dateTo.value ? "开始日期不能晚于结束日期" : "");
const allDocuments = computed(() => [...documents.value, ...historyDocuments.value]);
let historyController: AbortController | undefined;
let historyTimer: ReturnType<typeof setTimeout> | undefined;
let historyReady = false;
const fileInput = ref<HTMLInputElement>();
const updateInput = ref<HTMLInputElement>();
const updateTarget = ref<DocumentItem | null>(null);
const dragDepth = ref(0);
const search = ref("");
const filter = ref("all");
const baseFilter = ref("");
const knowledgeBase = ref("");
const bases = ref<KnowledgeBaseOption[]>([]);
const basesLoading = ref(false);
const basesError = ref("");
const writableBases = computed(() => bases.value.filter((base) => ["WRITE", "ADMIN"].includes(base.permission)));
const selectedBase = computed(() => bases.value.find((base) => base.id === knowledgeBase.value));
const confidentiality = ref(1);
const clearance = ref(isDemoMode ? 2 : 0);
const departmentId = ref<string | null>(null);
const settingsReady = ref(isDemoMode);
const settingsLoading = ref(false);
const settingsError = ref("");
const availableClassifications = computed(() => [0, 1, 2].filter((level) => level <= clearance.value));
const metadataController = new AbortController();
let basesController: AbortController | undefined;
let settingsController: AbortController | undefined;
let preferredKnowledgeBase = "";
const permissionTarget = ref<PermissionTarget | null>(null);
const permissionDialogVersion = ref(0);
let accessVersion = 0;
const permissionChannel = !isDemoMode && typeof BroadcastChannel !== "undefined"
  ? new BroadcastChannel("zhiqing-permissions") : null;
let accessRefreshQueue = Promise.resolve();
const chunkSize = ref(512);
const chunkOverlap = ref(48);
const modal = ref<"help" | "remove" | "preview" | null>(null);
const previewItem = ref<DocumentItem | null>(null);
const previewLoading = ref(false);
const previewError = ref("");
const previewData = ref<DocumentPreview | null>(null);
const previewUrl = ref("");
const previewSheet = ref("");
let previewController: AbortController | undefined;
const previewSheets = computed(() => [...new Set(
  previewData.value?.elements.map((element) => element.sheet_name).filter((name): name is string => !!name) || [],
)]);
const previewElements = computed(() => previewData.value?.elements.filter(
  (element) => !previewSheet.value || element.sheet_name === previewSheet.value,
) || []);
const removalTargets = ref<DocumentItem[]>([]);
const removing = ref(false);
const removalError = ref("");
const dialog = ref<HTMLDialogElement>();
const toast = ref("");
let toastTimer: ReturnType<typeof setTimeout>;
const controllers = new Map<string, AbortController>();
const running = ref(false);
const pending = computed(() =>
  documents.value.filter(
    (item) =>
      !item.result && !item.removing && ["pending", "error", "cancelled"].includes(item.status),
  ),
);
const inProgress = computed(
  () =>
    allDocuments.value.filter((item) =>
      ["uploading", "queued", "processing"].includes(item.status),
    ).length,
);
const totalSize = computed(() =>
  historyPage.value.overall_size + documents.value.reduce((total, item) => total + item.size, 0),
);
const totalDocuments = computed(() => historyPage.value.overall_total + documents.value.length);
function documentBaseId(item: DocumentItem): string {
  return (
    !item.result && ["pending", "error", "cancelled"].includes(item.status)
      ? knowledgeBase.value
      : item.knowledgeBase
  );
}
function documentBaseName(item: DocumentItem): string {
  const baseId = documentBaseId(item);
  if (!baseId) return "待选择知识库";
  return bases.value.find((base) => base.id === baseId)?.name || `知识库 #${baseId}`;
}
const baseFilterOptions = computed(() => {
  const options = new Map(bases.value.map((base) => [base.id, base.name]));
  for (const item of allDocuments.value) {
    const id = documentBaseId(item);
    if (id && !options.has(id)) options.set(id, documentBaseName(item));
  }
  return [...options].map(([id, name]) => ({ id, name }));
});
const baseDocuments = computed(() =>
  documents.value.filter(
    (item) => (!baseFilter.value || documentBaseId(item) === baseFilter.value) &&
      item.name.toLowerCase().includes(search.value.trim().toLowerCase()) &&
      !dateError.value &&
      (!dateFrom.value || localDate(item.createdAt) >= dateFrom.value) &&
      (!dateTo.value || localDate(item.createdAt) <= dateTo.value),
  ),
);
const localMatches = computed(() =>
  baseDocuments.value.filter(
    (item) =>
      (filter.value === "all" ||
        (filter.value === "processing"
          ? ["queued", "processing"].includes(item.status)
          : filter.value === "error"
            ? ["error", "process_error"].includes(item.status)
            : item.status === filter.value)),
  ),
);
const matchTotal = computed(() => historyPage.value.total + localMatches.value.length);
const totalPages = computed(() => Math.max(1, Math.ceil(matchTotal.value / pageSize.value)));
watch([currentPage, totalPages], () => {
  pageDraft.value = String(currentPage.value);
});
function jumpToPage() {
  if (historyLoading.value || dateError.value) return;
  const text = pageDraft.value.trim();
  const page = Number(text);
  if (!/^\d+$/.test(text) || !Number.isSafeInteger(page) || page < 1 || page > totalPages.value) {
    notify(`请输入 1–${totalPages.value} 之间的整数页码`);
    return;
  }
  currentPage.value = page;
  pageDraft.value = String(page);
}
const visible = computed(() => {
  const offset = (currentPage.value - 1) * pageSize.value;
  return [
    ...localMatches.value.slice(offset, offset + pageSize.value),
    ...(historyLoading.value || historyError.value || dateError.value ? [] : historyDocuments.value),
  ];
});
const queueCounts = computed(() => ({
  all: (historyPage.value.counts.all || 0) + baseDocuments.value.length,
  pending: baseDocuments.value.filter((item) => item.status === "pending").length,
  done: (historyPage.value.counts.done || 0) + baseDocuments.value.filter((item) => item.status === "done").length,
  processing: (historyPage.value.counts.processing || 0) + baseDocuments.value.filter((item) => ["queued", "processing"].includes(item.status)).length,
  error: (historyPage.value.counts.error || 0) + baseDocuments.value.filter((item) => ["error", "process_error"].includes(item.status)).length,
}));
function localDate(value: string) {
  return new Intl.DateTimeFormat("sv-SE", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(value));
}
function uploadTime(value: string) {
  return new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date(value));
}
const allSelected = computed(
  () =>
    visible.value.some(canSelect) && visible.value.filter(canSelect).every((item) => item.selected),
);
const selectedCount = computed(
  () => new Set(allDocuments.value.filter((item) => item.selected)
    .map((item) => item.result?.document_id || item.id)).size,
);
const statusLabels = computed<Record<Status, string>>(() => ({
  pending: "待上传",
  uploading: "上传中",
  done: isDemoMode ? "演示完成" : "已入库",
  queued: "等待处理",
  processing: "处理中",
  process_error: "处理失败",
  error: "上传失败",
  cancelled: "已取消",
}));

async function loadBases() {
  basesController?.abort();
  const controller = new AbortController();
  basesController = controller;
  basesLoading.value = true;
  basesError.value = "";
  try {
    const loadedBases: KnowledgeBaseOption[] = isDemoMode
      ? [{ id: "demo", name: "演示知识库", description: "仅用于本地交互演示", permission: "WRITE" }]
      : await fetchKnowledgeBases("READ", AbortSignal.any([controller.signal, metadataController.signal]));
    if (!mounted || controller.signal.aborted) return;
    bases.value = loadedBases;
    const preferred = knowledgeBase.value || preferredKnowledgeBase;
    if (writableBases.value.some((base) => base.id === preferred)) knowledgeBase.value = preferred;
    if (!writableBases.value.some((base) => base.id === knowledgeBase.value))
      knowledgeBase.value = writableBases.value[0]?.id || "";
    preferredKnowledgeBase = knowledgeBase.value;
    if (baseFilter.value && !bases.value.some((base) => base.id === baseFilter.value)) baseFilter.value = "";
    if (!bases.value.length)
      basesError.value = "当前账号没有可访问的知识库，请联系管理员授权。";
  } catch (error) {
    if (!mounted || controller.signal.aborted) return;
    bases.value = [];
    knowledgeBase.value = "";
    basesError.value =
      error instanceof Error
        ? error.message
        : "知识库加载失败，请启动后端服务后重试。";
  } finally {
    if (basesController === controller) basesLoading.value = false;
  }
}
async function loadSettings() {
  settingsController?.abort();
  const controller = new AbortController();
  settingsController = controller;
  settingsLoading.value = true;
  settingsError.value = "";
  try {
    const settings = await fetchUploadSettings(AbortSignal.any([controller.signal, metadataController.signal]));
    if (!mounted || controller.signal.aborted) return;
    chunkSize.value = settings.chunk_size;
    chunkOverlap.value = settings.chunk_overlap;
    clearance.value = settings.clearance;
    departmentId.value = settings.department_id;
    if (confidentiality.value > clearance.value) confidentiality.value = Math.min(1, clearance.value);
    settingsReady.value = true;
  } catch (error) {
    if (mounted && !controller.signal.aborted) {
      settingsReady.value = false;
      settingsError.value = error instanceof Error ? error.message : "上传权限配置暂不可用，请重试";
    }
  } finally { if (mounted && settingsController === controller) settingsLoading.value = false; }
}

function canWrite(item: DocumentItem) { return !item.result || !!item.task?.can_write; }
function canSelect(item: DocumentItem) {
  return canWrite(item) && !item.removing && !item.updating && item.status !== "uploading" && !item.retrying;
}
function manageBase() {
  if (isDemoMode || selectedBase.value?.permission !== "ADMIN" || removing.value) return;
  closeModal();
  permissionTarget.value = { kind: "knowledge-base", id: selectedBase.value.id, name: selectedBase.value.name };
}
function manageDocument(item: DocumentItem) {
  if (isDemoMode || !item.result || !item.task?.can_manage_permissions || removing.value) return;
  closeModal();
  permissionTarget.value = { kind: "document", id: item.result.document_id, name: item.name };
}
function clearAccessCaches() {
  queryAccessVersion.value++;
  accessVersion++;
  authClient.cancelRequests();
  historyReady = false;
  historyController?.abort();
  basesController?.abort();
  settingsController?.abort();
  if (knowledgeBase.value) preferredKnowledgeBase = knowledgeBase.value;
  knowledgeBase.value = "";
  bases.value = [];
  settingsReady.value = false;
  clearTimeout(historyTimer);
  historyDocuments.value = [];
  historyPage.value = { items: [], total: 0, overall_total: 0, overall_size: 0, counts: {} };
  closeRevisionHistory();
  if (modal.value !== "help") {
    dialog.value?.close();
    modal.value = null;
    removalTargets.value = [];
    removalError.value = "";
  }
  clearPreview();
  documents.value.forEach((item) => { item.selected = false; });
}
function refreshAccessState(broadcast = true) {
  if (!mounted || isDemoMode) return Promise.resolve();
  clearAccessCaches();
  if (broadcast) permissionChannel?.postMessage("changed");
  else if (permissionTarget.value) permissionDialogVersion.value++;
  accessRefreshQueue = accessRefreshQueue.catch(() => {}).then(async () => {
    if (!mounted) return;
    clearAccessCaches();
    await loadBases();
    if (!mounted) return;
    await loadSettings();
    if (!mounted) return;
    historyReady = true;
    await loadHistory();
    const target = permissionTarget.value;
    if (target?.kind === "knowledge-base" && bases.value.find((base) => base.id === target.id)?.permission !== "ADMIN")
      permissionTarget.value = null;
    if (target?.kind === "document" && !allDocuments.value.some((item) =>
      item.result?.document_id === target.id && item.task?.can_manage_permissions)) permissionTarget.value = null;
  });
  return accessRefreshQueue;
}
function permissionInaccessible() {
  permissionTarget.value = null;
  notify("访问权限已变化，正在刷新文档与知识库");
  void refreshAccessState();
}
if (permissionChannel) permissionChannel.onmessage = (event) => {
  if (event.data === "changed") void refreshAccessState(false);
};
let pollTimer: ReturnType<typeof setInterval> | undefined;
let pollBusy = false;
let mounted = true;
const stageLabels: Record<string, string> = {
  QUEUED: "等待处理",
  DOWNLOADING: "读取文件",
  PARSING: "解析与分块",
  EMBEDDING: "生成向量",
  SAVING: "向量入库",
  COMPLETED: "已入库",
  RETRY_WAIT: "等待重试",
};
const processingErrors: Record<string, string> = {
  PARSE_INCOMPLETE: "文档解析不完整，请检查 OCR 或公式内容",
  NO_USABLE_CONTENT: "文档没有可用正文",
  MODEL_CONFIGURATION_ERROR: "向量模型配置无效，请检查服务端凭据",
  MODEL_CONFIG_CHANGED: "向量模型配置已变更",
  ACCESS_REVOKED: "权限或账号状态已变更",
  FILE_INTEGRITY_FAILED: "原文件完整性校验失败",
  OBJECT_NOT_FOUND: "原文件不存在",
  PROCESSING_TIMEOUT: "处理超时",
  ATTEMPTS_EXHAUSTED: "已达到自动重试次数",
  REVISION_UNAVAILABLE: "文档已删除、撤回或修订已变更",
};
function applyTask(item: DocumentItem, task: DocumentTaskStatus) {
  item.task = task;
  if (item.result) item.result.generation_id = task.generation_id;
  item.progress = task.progress;
  item.pollError = undefined;
  if (task.document_status === "DELETED") {
    item.status = "process_error";
    item.error = "永久删除尚未完成，请点击移除按钮重试清理";
    return;
  }
  item.status =
    task.status === "SUCCEEDED"
      ? "done"
      : task.status === "FAILED"
        ? "process_error"
        : task.status === "CANCELLED"
          ? "cancelled"
          : task.status === "RUNNING"
            ? "processing"
            : "queued";
  item.error =
    task.status === "FAILED"
      ? processingErrors[task.error_code || ""] ||
        `处理失败（${task.error_code || "UNKNOWN"}），可重试处理`
      : undefined;
}
function fromTask(task: DocumentTaskStatus): DocumentItem {
  const item: DocumentItem = {
    id: `task-${task.task_id}`, file: null, name: task.original_filename,
    size: task.file_size, createdAt: task.created_at,
    extension: task.original_filename.split(".").pop()?.toLowerCase() || "",
    status: "queued", progress: 0, selected: false,
    knowledgeBase: task.knowledge_base_id,
    result: {
      document_id: task.document_id, revision_id: task.revision_id,
      generation_id: task.generation_id, task_id: task.task_id,
      status: "STORED", processing_status: "PENDING",
    },
  };
  applyTask(item, task);
  return item;
}
async function loadHistory() {
  if (!mounted) return;
  clearTimeout(historyTimer);
  historyController?.abort();
  historyError.value = "";
  if (!historyReady || isDemoMode || dateError.value) {
    historyLoading.value = false;
    return;
  }
  const controller = new AbortController();
  historyController = controller;
  historyLoading.value = true;
  const pageOffset = (currentPage.value - 1) * pageSize.value;
  const localCount = localMatches.value.slice(pageOffset, pageOffset + pageSize.value).length;
  const query = new URLSearchParams({
    offset: String(Math.max(0, pageOffset - localMatches.value.length)),
    limit: String(Math.max(1, pageSize.value - localCount)), status: filter.value,
    search: search.value.trim(),
  });
  if (baseFilter.value) query.set("knowledge_base_id", baseFilter.value);
  if (dateFrom.value) query.set("date_from", dateFrom.value);
  if (dateTo.value) query.set("date_to", dateTo.value);
  try {
    const page = await fetchDocumentTasks(query, controller.signal);
    if (!mounted || controller.signal.aborted) return;
    historyPage.value = page;
    historyDocuments.value = localCount < pageSize.value ? page.items.map(fromTask) : [];
    if (expandedDocumentId.value) closeRevisionHistory();
    if (currentPage.value > totalPages.value) currentPage.value = totalPages.value;
  } catch (error) {
    if (mounted && !controller.signal.aborted) {
      historyDocuments.value = [];
      historyError.value = error instanceof Error ? error.message : "文档列表查询失败";
    }
  } finally {
    if (!controller.signal.aborted) historyLoading.value = false;
  }
}
function closeRevisionHistory() {
  revisionHistoryController?.abort();
  expandedDocumentId.value = null;
  revisionHistory.value = [];
  revisionHistoryError.value = "";
  revisionHistoryLoading.value = false;
}
async function toggleRevisionHistory(item: DocumentItem) {
  const documentId = item.result?.document_id;
  if (!documentId) return;
  if (expandedDocumentId.value === documentId) {
    closeRevisionHistory();
    return;
  }
  closeRevisionHistory();
  expandedDocumentId.value = documentId;
  revisionHistoryLoading.value = true;
  const controller = new AbortController();
  revisionHistoryController = controller;
  try {
    const revisions = await fetchDocumentRevisions(documentId, controller.signal);
    if (controller.signal.aborted || expandedDocumentId.value !== documentId) return;
    revisionHistory.value = revisions.filter(
      (revision) => revision.revision_id !== item.result?.revision_id,
    ).map(fromTask);
  } catch (error) {
    if (!controller.signal.aborted) revisionHistoryError.value =
      error instanceof Error ? error.message : "修订历史查询失败";
  } finally {
    if (!controller.signal.aborted) revisionHistoryLoading.value = false;
  }
}
function scheduleHistory() {
  if (!historyReady) return;
  historyController?.abort();
  historyLoading.value = !isDemoMode && !dateError.value;
  clearTimeout(historyTimer);
  allDocuments.value.forEach((item) => item.selected = false);
  historyTimer = setTimeout(loadHistory, 250);
}
watch([baseFilter, search, filter, dateFrom, dateTo, pageSize, knowledgeBase, () => documents.value.length], () => {
  currentPage.value = 1;
  scheduleHistory();
});
watch(currentPage, scheduleHistory);
async function pollTasks() {
  if (pollBusy || !mounted) return;
  pollBusy = true;
  let changed = false;
  try {
    await Promise.all(
      allDocuments.value
        .filter(
          (item) =>
            item.result && !item.removing && ["queued", "processing"].includes(item.status),
        )
        .map(async (item) => {
          try {
            const task = await fetchDocumentTask(item.result!.task_id);
            if (mounted && !item.removing) {
              changed ||= item.task?.status !== task.status;
              applyTask(item, task);
            }
          } catch (error) {
            if (mounted && !item.removing)
              item.pollError =
                error instanceof Error
                  ? error.message
                  : "任务状态暂不可用，稍后自动重试";
          }
        }),
    );
    if (changed && !historyLoading.value) await loadHistory();
  } finally {
    pollBusy = false;
  }
}
async function retryProcessing(item: DocumentItem) {
  if (!item.result || !canWrite(item) || item.retrying || item.removing) return;
  item.retrying = true;
  try {
    applyTask(item, await retryDocumentTask(item.result.task_id));
    notify("已重新加入后台处理队列，无需重新上传");
    await loadHistory();
  } catch (error) {
    notify(error instanceof Error ? error.message : "处理任务重试失败");
  } finally {
    item.retrying = false;
  }
}
function canUpdate(item: DocumentItem): boolean {
  // 列表按处理任务展示，同一文档可能有多条历史修订。只允许从当前期望修订更新；
  // 服务端还会在事务里复核，防止页面数据过期时覆盖其他人的更新。
  return !isDemoMode && canWrite(item) && !!item.result && !!item.task &&
    item.task.desired_revision_id === item.task.revision_id &&
    ["DRAFT", "PUBLISHED"].includes(item.task.document_status);
}
function chooseUpdate(item: DocumentItem) {
  if (!canUpdate(item) || item.updating || item.removing || removing.value) return;
  // 文件选择器只有一个，用当前行保存目标文档及其修订号。
  updateTarget.value = item;
  updateInput.value?.click();
}
async function onUpdateFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  const item = updateTarget.value;
  // 清空 input 后，用户下次仍可选择同一个本地文件；取消选择不发起请求。
  input.value = "";
  updateTarget.value = null;
  if (!file || !item?.result || !canUpdate(item)) return;
  const extension = file.name.split(".").pop()?.toLowerCase() || "";
  if (!allowedExtensions.includes(extension) || !file.size || file.size > maxFileSize) {
    notify("更新文件格式无效、为空或超过 50 MB");
    return;
  }
  item.updating = true;
  try {
    // 更新请求携带原 document_id 和用户看到的修订 ID，不创建另一份业务文档。
    // 新任务入队后重新读服务端列表，让新旧修订的发布状态以数据库指针为准。
    await uploadDocument(file, {
      knowledgeBase: item.knowledgeBase,
      documentId: item.result.document_id,
      expectedRevisionId: item.task!.revision_id,
      signal: new AbortController().signal,
      onProgress: () => {},
    });
    notify("新修订已保存，后台正在处理；发布前旧版本继续可用");
    closeRevisionHistory();
    currentPage.value = 1;
    await loadHistory();
  } catch (error) {
    notify(error instanceof Error ? error.message : "更新文件失败，请重试");
  } finally {
    item.updating = false;
  }
}
onMounted(async () => {
  if (!isDemoMode) pollTimer = setInterval(pollTasks, 2500);
  await loadBases();
  if (!mounted) return;
  if (!isDemoMode) {
    await loadSettings();
  }
  historyReady = true;
  if (!isDemoMode && mounted && bases.value.length) {
    await loadHistory();
  }
});

function formatSize(size: number) {
  return size < 1024 * 1024
    ? `${(size / 1024).toFixed(1)} KB`
    : `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
function notify(message: string) {
  toast.value = message;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.value = "";
  }, 5000);
}
function addFiles(files: FileList | File[]) {
  const errors: string[] = [];
  let added = 0;
  for (const file of Array.from(files)) {
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    if (!allowedExtensions.includes(extension)) {
      errors.push(`${file.name}：不支持此格式`);
      continue;
    }
    if (file.size === 0) {
      errors.push(`${file.name}：文件为空`);
      continue;
    }
    if (file.size > maxFileSize) {
      errors.push(`${file.name}：超过 50 MB`);
      continue;
    }
    if (
      allDocuments.value.some(
        (item) =>
          item.name === file.name &&
          item.size === file.size &&
          (!item.file || item.file.lastModified === file.lastModified),
      )
    ) {
      errors.push(`${file.name}：已在列表中`);
      continue;
    }
    documents.value.push({
      id: crypto.randomUUID(),
      file,
      name: file.name,
      size: file.size,
      extension,
      status: "pending",
      progress: 0,
      selected: false,
      knowledgeBase: knowledgeBase.value,
      createdAt: new Date().toISOString(),
    });
    added++;
  }
  notify(
    [
      added ? `已添加 ${added} 个文件` : "",
      ...errors.slice(0, 3),
      errors.length > 3 ? `另有 ${errors.length - 3} 个文件未添加` : "",
    ]
      .filter(Boolean)
      .join("；"),
  );
}
function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement;
  if (input.files) addFiles(input.files);
  input.value = "";
}
function onDrop(event: DragEvent) {
  dragDepth.value = 0;
  if (event.dataTransfer?.files) addFiles(event.dataTransfer.files);
}
async function requestRemoval(items: DocumentItem[]) {
  if (removing.value) return;
  // 多条修订记录可能属于同一文档。更新进行中不删除该文档；批量移除时
  // 同一 document_id 只调用一次删除接口，避免重复删除造成误报。
  const updatingDocuments = new Set(allDocuments.value.filter((item) => item.updating)
    .map((item) => item.result?.document_id).filter((id): id is string => !!id));
  const seen = new Set<string>();
  removalTargets.value = items.filter(
    (item) => canWrite(item) && item.status !== "uploading" && !item.retrying && !item.removing &&
      !item.updating && (!item.result || !updatingDocuments.has(item.result.document_id)),
  ).filter((item) => {
    const key = item.result?.document_id || item.id;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (!removalTargets.value.length) return;
  if (!removalTargets.value.some((item) => item.result)) {
    await confirmRemoval();
    return;
  }
  removalError.value = "";
  modal.value = "remove";
  await nextTick();
  dialog.value?.showModal();
}
function removeSelected() {
  void requestRemoval(allDocuments.value.filter((item) => item.selected));
}
async function confirmRemoval() {
  if (removing.value) return;
  removing.value = true;
  removalError.value = "";
  let removed = 0;
  const failed: DocumentItem[] = [];
  for (const item of [...removalTargets.value]) {
    if (!mounted) break;
    if (!allDocuments.value.some((row) => row.id === item.id)) continue;
    item.removing = true;
    try {
      if (item.result) await deleteDocument(item.result.document_id);
      documents.value = documents.value.filter(
        (row) => row.id !== item.id &&
          (!item.result || row.result?.document_id !== item.result.document_id),
      );
      historyDocuments.value = historyDocuments.value.filter(
        (row) => row.id !== item.id && (!item.result || row.result?.document_id !== item.result.document_id),
      );
      removed++;
    } catch (error) {
      failed.push(item);
      removalError.value = error instanceof Error ? error.message : "文档移除失败，请重试";
      if (mounted && item.result) {
        try {
          applyTask(item, await fetchDocumentTask(item.result.task_id));
        } catch { /* 保留当前行，允许重试移除。 */ }
      }
    } finally {
      item.removing = false;
    }
  }
  removing.value = false;
  removalTargets.value = failed;
  if (removed) notify(`已移除 ${removed} 个文件`);
  if (!failed.length) closeModal();
  if (removed) await loadHistory();
}
function cancel(item: DocumentItem) {
  controllers.get(item.id)?.abort();
}
async function startUpload() {
  if (running.value || !pending.value.length || !knowledgeBase.value || !settingsReady.value ||
    !writableBases.value.some((base) => base.id === knowledgeBase.value)) return;
  running.value = true;
  const batch = [...pending.value];
  const batchBase = knowledgeBase.value;
  const batchConfidentiality = confidentiality.value;
  const batchAccessVersion = accessVersion;
  let success = 0;
  try {
    for (const item of batch) {
      if (!mounted || batchAccessVersion !== accessVersion) break;
      if (!documents.value.some((document) => document.id === item.id))
        continue;
      const controller = new AbortController();
      controllers.set(item.id, controller);
      item.status = "uploading";
      item.progress = 0;
      item.error = undefined;
      item.knowledgeBase = batchBase;
      try {
        item.result = await uploadDocument(item.file!, {
          knowledgeBase: batchBase,
          confidentiality: batchConfidentiality,
          signal: controller.signal,
          onProgress: (progress) => {
            item.progress = progress;
          },
        });
        item.status = isDemoMode ? "done" : "queued";
        item.progress = isDemoMode ? 100 : 0;
        success++;
        if (!isDemoMode) {
          documents.value = documents.value.filter((row) => row.id !== item.id);
        }
      } catch (error) {
        item.status =
          error instanceof DOMException && error.name === "AbortError"
            ? "cancelled"
            : "error";
        item.error =
          error instanceof Error ? error.message : "上传失败，请重试";
      } finally {
        controllers.delete(item.id);
      }
    }
    notify(
      isDemoMode
        ? `本地演示完成 ${success} 个文件，文件未发送至服务器`
        : `已接收 ${success} 个文件，后台正在异步解析与生成向量`,
    );
  } finally {
    running.value = false;
    if (!isDemoMode && mounted) {
      documents.value = documents.value.filter((item) => !item.result);
      currentPage.value = 1;
      await loadHistory();
    }
  }
}
function downloadExample() {
  const blob = new Blob(
    [
      "# 知擎知识库示例\n\n## 平台简介\n知擎将团队文档转化为可检索的知识。\n\n## 上传说明\n选择知识库，添加文档，然后点击开始上传。\n",
    ],
    { type: "text/markdown;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "知擎知识库示例.md";
  anchor.click();
  URL.revokeObjectURL(url);
}
async function openModal(type: "help") {
  if (removing.value) return;
  modal.value = type;
  await nextTick();
  dialog.value?.showModal();
}
function clearPreview() {
  previewController?.abort();
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value);
  previewUrl.value = "";
  previewData.value = null;
  previewSheet.value = "";
  previewItem.value = null;
  previewLoading.value = false;
  previewError.value = "";
}
async function openPreview(item: DocumentItem) {
  if (item.removing || (!item.file && !item.result)) return;
  clearPreview();
  const controller = new AbortController();
  previewController = controller;
  previewItem.value = item;
  previewLoading.value = true;
  modal.value = "preview";
  await nextTick();
  dialog.value?.showModal();
  try {
    const blob = item.file || await fetchOriginalDocument(item.result!, controller.signal);
    if (controller.signal.aborted) return;
    previewUrl.value = URL.createObjectURL(
      item.extension === "pdf" ? new Blob([blob], { type: "application/pdf" }) : blob,
    );
    if (item.extension !== "pdf") {
      if (["txt", "md", "markdown"].includes(item.extension) && isDemoMode) {
        previewData.value = {
          elements: [{ kind: "paragraph", text: await blob.text(), sheet_name: null, cell_range: null, table_rows: [] }],
          warnings: [], truncated: false,
        };
      } else {
        if (isDemoMode) throw new Error("演示模式暂不支持此格式的正文预览，请下载原文件查看。");
        const data = await fetchDocumentPreview(item.file, item.result, controller.signal);
        if (controller.signal.aborted) return;
        previewData.value = data;
        previewSheet.value = previewSheets.value[0] || "";
      }
    }
  } catch (error) {
    if (!controller.signal.aborted)
      previewError.value = error instanceof Error ? error.message : "文件预览失败，请重试";
  } finally {
    if (!controller.signal.aborted) previewLoading.value = false;
  }
}
function closeModal() {
  if (removing.value) return;
  dialog.value?.close();
  modal.value = null;
  clearPreview();
}
onBeforeUnmount(() => {
  metadataController.abort();
  permissionChannel?.close();
  closeRevisionHistory();
  historyController?.abort();
  clearTimeout(historyTimer);
  clearPreview();
  mounted = false;
  clearInterval(pollTimer);
  controllers.forEach((controller) => controller.abort());
  clearTimeout(toastTimer);
});
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <a class="brand" href="/" aria-label="知擎首页"
        ><span class="brand-icon"><Layers3 :size="23" /></span
        ><span>知擎<span class="brand-en">ZHIQING</span></span></a
      >
      <div class="workspace">
        <span class="workspace-icon">知</span>
        <div>
          <strong :title="identity?.tenant.name">{{ identity?.tenant.name || '知擎工作空间' }}</strong><small>文档与知识库管理</small>
        </div>
      </div>
      <div class="nav-caption">工作空间</div>
      <nav aria-label="主导航">
        <button class="nav-item" :class="{ active: activeView === 'query' }" :aria-current="activeView === 'query' ? 'page' : undefined" @click="closeModal(); activeView = 'query'">
          <MessageSquare :size="19" />智能问答<span v-if="activeView === 'query'" class="nav-dot"></span>
        </button>
        <button class="nav-item" disabled>
          <BookOpen :size="19" />知识库<span class="soon">即将上线</span>
        </button>
        <button class="nav-item" :class="{ active: activeView === 'documents' }" :aria-current="activeView === 'documents' ? 'page' : undefined" @click="closeModal(); activeView = 'documents'">
          <CloudUpload :size="19" />文档上传<span v-if="activeView === 'documents'" class="nav-dot"></span>
        </button>
      </nav>
      <div class="sidebar-bottom">
        <button @click="openModal('help')">
          <CircleHelp :size="18" />帮助与指南<ArrowRight :size="15" />
        </button>
        <div class="profile">
          <span class="avatar">{{ (identity?.user.display_name || identity?.user.email || '知').slice(0, 1).toUpperCase() }}</span>
          <div class="profile-identity"><strong :title="identity?.user.display_name || identity?.user.email">{{ identity?.user.display_name || identity?.user.email || '本地工作空间' }}</strong><small :title="identity?.user.email">{{ isDemoMode ? '演示环境' : identity?.user.email }}</small></div>
          <span class="online-dot"></span>
        </div>
      </div>
    </aside>

    <div class="main-shell">
      <header class="topbar">
        <div class="breadcrumbs">
          <span>工作空间</span><ChevronRight :size="14" /><strong
            >{{ activeView === 'query' ? '智能问答' : '文档上传' }}</strong
          >
        </div>
        <div class="topbar-right">
          <span class="mode"
            ><span></span
            >{{ isDemoMode ? "演示环境" : "服务已连接" }}</span
          ><button
            class="icon-button"
            aria-label="帮助"
            @click="openModal('help')"
          >
            <CircleHelp :size="19" /></button
          ><button v-if="!isDemoMode" class="button secondary auth-logout" :disabled="loggingOut" @click="emit('logout')"><LoaderCircle v-if="loggingOut" :size="15" class="spin" /><LogOut v-else :size="15" />{{ loggingOut ? '退出中…' : '退出所有设备' }}</button>
        </div>
      </header>
      <div v-if="identity" class="auth-mobile-identity"><span :title="identity.tenant.name">{{ identity.tenant.name }}</span><span :title="identity.user.email">{{ identity.user.display_name || identity.user.email }}</span></div>
      <div v-if="logoutError" class="auth-session-error" role="alert">{{ logoutError }}。请点击“退出所有设备”重试。</div>
      <main v-show="activeView === 'documents'">
        <div class="page-heading">
          <div>
            <h1>文档上传</h1>
            <p>选择知识库，上传并管理文档。</p>
          </div>
          <button class="button secondary" @click="downloadExample">
            <ArrowDownToLine :size="16" />下载示例文档
          </button>
        </div>

        <div class="content-grid">
          <section class="upload-card card" aria-labelledby="upload-title">
            <div class="section-heading">
              <div>
                <h2 id="upload-title">上传文档</h2>
                <span class="light-label">支持批量上传</span>
              </div>
            </div>
            <div
              class="dropzone"
              :class="{ dragging: dragDepth > 0 }"
              @dragenter.prevent="dragDepth++"
              @dragover.prevent
              @dragleave.prevent="dragDepth = Math.max(0, dragDepth - 1)"
              @drop.prevent="onDrop"
            >
              <CloudUpload class="dropzone-icon" :size="28" :stroke-width="1.5" aria-hidden="true" />
              <h3>
                {{ dragDepth > 0 ? "松开鼠标添加文件" : "拖拽文件到此处" }}
              </h3>
              <p>或选择本地文件</p>
              <button class="button primary" @click="fileInput?.click()">
                <Plus :size="18" />选择文档</button
              ><input
                ref="fileInput"
                type="file"
                class="sr-only"
                multiple
                :accept="
                  allowedExtensions
                    .map((extension) => `.${extension}`)
                    .join(',')
                "
                aria-label="选择上传文档"
                @change="onFileChange"
              />
              <small class="upload-formats">PDF、Word、Excel、Markdown、TXT · 单个文件 ≤ 50 MB</small>
            </div>
            <div class="upload-tip">
              <Info :size="16" /><span>{{
                isDemoMode
                  ? "当前为本地演示：上传进度仅供预览，不会入库或触发解析。"
                  : "上传后自动处理，处理结果可在下方列表查看。"
              }}</span>
            </div>
          </section>

          <aside class="config-column">
            <section class="card destination-card">
              <div class="section-heading">
                <div>
                  <h2>上传至知识库</h2>
                </div>
              </div>
              <label class="field-label" for="knowledge-base"
                >目标知识库<span>*</span></label
              >
              <div class="select-wrap">
                <BookOpen :size="17" /><select
                  id="knowledge-base"
                  v-model="knowledgeBase"
                  :disabled="running || basesLoading || !writableBases.length"
                >
                  <option v-if="!writableBases.length" value="">
                    {{ basesLoading ? "正在加载…" : "暂无可写入知识库" }}
                  </option>
                  <option v-for="base in writableBases" :key="base.id" :value="base.id">
                    {{ base.name }}
                  </option></select
                ><ChevronDown :size="15" />
              </div>
              <div v-if="basesError" class="base-load-error" role="alert">
                <p>{{ basesError }}</p>
                <button class="text-button" @click="loadBases">重新加载</button>
              </div>
              <p class="destination-hint">{{ bases.length && !writableBases.length ? '当前仅有读取权限，可浏览下方文档；上传需知识库写入权限。' : '新上传的文档将保存至所选知识库。' }}</p>
              <button v-if="!isDemoMode && selectedBase?.permission === 'ADMIN'" class="text-button manage-base-button" :disabled="removing" @click="manageBase"><ShieldCheck :size="15" />管理授权</button>
              <label class="field-label confidentiality-label" for="upload-confidentiality">文档密级<span>*</span></label>
              <div class="select-wrap"><ShieldCheck :size="17" /><select id="upload-confidentiality" v-model.number="confidentiality" :disabled="running || !settingsReady || settingsLoading"><option v-for="level in availableClassifications" :key="level" :value="level">{{ classificationLabels[level] }}{{ level === 1 ? '（默认）' : '' }}</option></select><ChevronDown :size="15" /></div>
              <p class="destination-hint">{{ confidentiality === 0 ? '内部公开文档仍仅对有知识库访问权限的账号开放。' : departmentId ? '默认仅本部门可见，跨部门需要明确共享。' : '未归属部门，非公开文档仅本人及明确共享对象可见。' }}</p>
              <div v-if="settingsError" class="base-load-error" role="alert"><p>{{ settingsError }}</p><button class="text-button" :disabled="settingsLoading" @click="loadSettings">重新加载上传权限</button></div>
              <div class="config-divider"></div>
              <div class="settings-heading">
                <strong>解析设置</strong
                ><span class="managed-settings">系统统一管理</span>
              </div>
              <dl class="settings-list">
                <div>
                  <dt>分块策略</dt>
                  <dd>智能分块</dd>
                </div>
                <div>
                  <dt>分块大小</dt>
                  <dd>{{ chunkSize }} tokens</dd>
                </div>
                <div>
                  <dt>重叠大小</dt>
                  <dd>{{ chunkOverlap }} tokens</dd>
                </div>
              </dl>
            </section>
          </aside>
        </div>

        <section class="card queue-card" aria-labelledby="queue-title">
          <div class="queue-heading">
            <div class="section-heading">
              <div>
                <h2 id="queue-title">上传列表</h2>
                <span class="count-badge">{{ totalDocuments }}</span>
              </div>
            </div>
            <div class="queue-actions">
              <span title="当前文档文件大小合计">{{ formatSize(totalSize) }}</span
              ><button
                class="text-button"
                :disabled="historyLoading || removing"
                @click="loadHistory"
              >
                <RefreshCw :size="15" />刷新列表
              </button>
            </div>
          </div>
          <div class="list-toolbar">
            <div class="filter-tabs" aria-label="文件状态筛选">
              <button
                :class="{ selected: filter === 'all' }"
                @click="filter = 'all'"
              >
                全部文件 <span>{{ queueCounts.all }}</span></button
              ><button
                :class="{ selected: filter === 'pending' }"
                @click="filter = 'pending'"
              >
                待上传
                <span>{{ queueCounts.pending }}</span></button
              ><button
                :class="{ selected: filter === 'done' }"
                @click="filter = 'done'"
              >
                已完成 <span>{{ queueCounts.done }}</span></button
              ><button
                :class="{ selected: filter === 'processing' }"
                @click="filter = 'processing'"
              >
                处理中
                <span>{{ queueCounts.processing }}</span></button
              ><button
                v-if="queueCounts.error || filter === 'error'"
                :class="{ selected: filter === 'error' }"
                @click="filter = 'error'"
              >
                失败
              </button>
            </div>
            <div class="queue-filters">
              <label class="base-filter-field">
                <BookOpen :size="15" aria-hidden="true" />
                <select v-model="baseFilter" aria-label="筛选所属知识库">
                  <option value="">全部知识库</option>
                  <option v-for="base in baseFilterOptions" :key="base.id" :value="base.id">
                    {{ base.name }}
                  </option>
                </select>
              </label>
              <label class="search-field"
              ><Search :size="16" /><input
                v-model="search"
                placeholder="搜索文件名称"
                aria-label="搜索文件名称"
            /></label>
              <span class="filter-result-count">共 {{ matchTotal }} 个匹配文件</span>
              <div class="date-filter-row">
                <span>上传日期</span>
                <label class="date-filter-field">
                  <input v-model="dateFrom" type="date" :max="dateTo || undefined" aria-label="上传开始日期" />
                </label>
                <span>至</span>
                <label class="date-filter-field">
                  <input v-model="dateTo" type="date" :min="dateFrom || undefined" aria-label="上传结束日期" />
                </label>
                <button v-if="dateFrom || dateTo || baseFilter || search || filter !== 'all'" class="text-button" @click="dateFrom = ''; dateTo = ''; baseFilter = ''; search = ''; filter = 'all'">重置筛选</button>
              </div>
              <p v-if="dateError" class="file-error" role="alert">{{ dateError }}</p>
            </div>
          </div>
          <div class="table-scroll">
            <table>
              <thead>
                <tr>
                  <th class="checkbox-cell">
                    <input
                      type="checkbox"
                      aria-label="选择当前列表全部文件"
                      :checked="allSelected"
                      :disabled="!visible.some(canSelect) || removing"
                      @change="
                        visible.forEach(
                          (item) =>
                            (item.selected = canSelect(item) && (
                              $event.target as HTMLInputElement
                            ).checked),
                        )
                      "
                    />
                  </th>
                  <th>文件名称</th>
                  <th class="base-column">所属知识库</th>
                  <th>文件大小</th>
                  <th>上传时间</th>
                  <th>状态</th>
                  <th class="action-cell">操作</th>
                </tr>
              </thead>
              <tbody>
                <template v-for="item in visible" :key="item.id">
                <tr>
                  <td class="checkbox-cell">
                    <input
                      v-model="item.selected"
                      type="checkbox"
                      :disabled="!canSelect(item) || removing"
                      :aria-label="`选择 ${item.name}`"
                    />
                  </td>
                  <td>
                    <div class="file-cell">
                      <span class="file-icon" :class="item.extension"
                        ><FileText :size="21"
                      /></span>
                      <button
                        type="button"
                        class="file-details"
                        :aria-label="`预览 ${item.name}`"
                        :disabled="item.removing || (!item.file && !item.result)"
                        @click="openPreview(item)"
                      >
                        <strong :title="item.name">{{ item.name }}</strong
                        ><small
                          >{{ item.extension.toUpperCase() }}<span v-if="item.task"> · 第 {{ item.task.revision_no }} 版 · {{ classificationLabels[item.task.confidentiality] || '未知密级' }}</span><span v-if="item.task && !item.task.can_write"> · 只读</span><span v-if="item.task?.status === 'SUCCEEDED'"> · {{ item.task.chunk_count }} 个分块</span></small
                        ><small v-if="item.task && item.task.active_generation_id === item.task.generation_id">当前发布版本</small
                        ><small v-else-if="item.task?.active_generation_id && item.task?.desired_revision_id === item.task?.revision_id">旧版本索引仍在使用</small
                        ><span v-if="item.pollError" class="file-error"
                          >{{ item.pollError }}；将自动重试查询</span
                        ><span
                          v-if="
                            item.error &&
                            ['error', 'process_error'].includes(item.status)
                          "
                          class="file-error"
                          >{{ item.error }}</span
                        >
                      </button>
                      <button
                        v-if="item.task && item.task.revision_no > 1"
                        class="text-button revision-toggle"
                        :aria-expanded="expandedDocumentId === item.result?.document_id"
                        :aria-label="`${expandedDocumentId === item.result?.document_id ? '收起' : '查看'} ${item.name} 的历史版本`"
                        @click="toggleRevisionHistory(item)"
                      ><ChevronDown v-if="expandedDocumentId === item.result?.document_id" :size="13" /><ChevronRight v-else :size="13" />历史版本（{{ item.task.revision_no - 1 }}）</button>
                    </div>
                  </td>
                  <td class="base-cell">
                    <span class="base-tag" :title="documentBaseName(item)">
                      <BookOpen :size="13" aria-hidden="true" />
                      <span>{{ documentBaseName(item) }}</span>
                    </span>
                  </td>
                  <td class="size-cell">{{ formatSize(item.size) }}</td>
                  <td class="upload-time-cell">
                    <time v-if="item.result" :datetime="item.createdAt" :title="`${uploadTime(item.createdAt)}（北京时间）`">
                      <span>{{ uploadTime(item.createdAt).split(" ")[0] }}</span>
                      <span>{{ uploadTime(item.createdAt).split(" ")[1] }}</span>
                    </time>
                    <span v-else title="尚未上传">—</span>
                  </td>
                  <td>
                    <span class="status" :class="item.status"
                      ><LoaderCircle
                        v-if="['uploading', 'processing'].includes(item.status)"
                        :size="13"
                        class="spin"
                      /><Check
                        v-else-if="item.status === 'done'"
                        :size="13"
                      /><span v-else class="status-dot"></span
                      >{{
                        item.task?.error_code === 'REVISION_SUPERSEDED'
                          ? '已被新版本替代'
                          : item.task && ["queued", "processing"].includes(item.status)
                            ? stageLabels[item.task.stage || "QUEUED"] || statusLabels[item.status]
                            : statusLabels[item.status]
                      }}<span
                        v-if="['uploading', 'processing'].includes(item.status)"
                        >{{ item.progress }}%</span
                      ></span
                    >
                    <div
                      v-if="['uploading', 'processing'].includes(item.status)"
                      class="progress-track"
                      role="progressbar"
                      :aria-valuenow="item.progress"
                      aria-valuemin="0"
                      aria-valuemax="100"
                      :aria-label="`${item.name} 上传进度`"
                    >
                      <span :style="{ width: `${item.progress}%` }"></span>
                    </div>
                  </td>
                  <td class="action-cell">
                    <div class="row-actions">
                      <button v-if="!isDemoMode && item.result && item.task?.can_manage_permissions" class="text-button" :disabled="item.updating || item.retrying || item.removing || removing" :aria-label="`管理 ${item.name} 的权限`" @click="manageDocument(item)"><ShieldCheck :size="13" />权限</button>
                      <button
                        v-if="item.status === 'process_error' && canUpdate(item)"
                        class="text-button"
                        :disabled="item.retrying || item.removing"
                        :aria-label="`重试处理 ${item.name}`"
                        @click="retryProcessing(item)"
                      >{{ item.retrying ? "提交中" : "重试处理" }}</button>
                      <button
                        v-if="canUpdate(item)"
                        class="text-button"
                        :disabled="item.updating || item.retrying || item.removing || removing"
                        :aria-label="`更新原文件 ${item.name}`"
                        @click="chooseUpdate(item)"
                      >{{ item.updating ? "上传中…" : "更新文件" }}</button>
                      <button
                        v-if="item.status === 'uploading'"
                        class="icon-button"
                        :aria-label="`取消上传 ${item.name}`"
                        @click="cancel(item)"
                      ><X :size="17" /></button>
                      <button
                        v-else-if="canWrite(item)"
                        class="icon-button delete-button"
                        :disabled="item.removing || item.retrying || item.updating || removing"
                        :aria-label="`移除 ${item.name}`"
                        @click="requestRemoval([item])"
                      >
                        <LoaderCircle v-if="item.removing" :size="16" class="spin" />
                        <Trash2 v-else :size="16" />
                      </button>
                    </div>
                  </td>
                </tr>
                <tr v-if="item.result && expandedDocumentId === item.result.document_id" class="revision-history-row">
                  <td colspan="7">
                    <div class="revision-history">
                      <strong>历史版本</strong>
                      <span v-if="revisionHistoryLoading">正在查询…</span>
                      <span v-else-if="revisionHistoryError" role="alert">{{ revisionHistoryError }}</span>
                      <span v-else-if="!revisionHistory.length">暂无历史版本</span>
                      <div v-for="revision in revisionHistory" :key="revision.id" class="revision-history-item">
                        <span>第 {{ revision.task?.revision_no }} 版 · {{ revision.name }}</span>
                        <span>{{ formatSize(revision.size) }} · {{ uploadTime(revision.createdAt) }}</span>
                        <span v-if="revision.task?.active_generation_id === revision.task?.generation_id">当前发布的索引</span>
                        <button class="text-button" :aria-label="`预览第 ${revision.task?.revision_no} 版 ${revision.name}`" @click="openPreview(revision)">预览原文件</button>
                      </div>
                    </div>
                  </td>
                </tr>
                </template>
              </tbody>
            </table>
            <input ref="updateInput" type="file" class="sr-only" :accept="allowedExtensions.map((extension) => `.${extension}`).join(',')" aria-label="选择更新文件" @change="onUpdateFileChange" />
          </div>
          <div v-if="historyLoading" class="list-message" role="status"><LoaderCircle :size="20" class="spin" />正在查询文档…</div>
          <div v-else-if="historyError" class="list-message" role="alert">
            <span>{{ historyError }}</span><button class="text-button" @click="loadHistory">重新查询</button>
          </div>
          <div v-else-if="!visible.length" class="empty-state">
            <span><FolderOpen :size="28" /></span
            ><strong>{{
              totalDocuments ? "没有匹配的文档" : "暂无文档"
            }}</strong>
            <p>
              {{
                totalDocuments
                  ? "试试其他关键词，或调整知识库、状态和日期筛选"
                  : "添加文档后，你可以在这里查看上传进度"
              }}
            </p>
            <button
              v-if="!totalDocuments"
              class="text-button"
              @click="fileInput?.click()"
            >
              选择文档 <ArrowRight :size="14" />
            </button>
          </div>
          <div class="pagination-bar">
            <span>{{ matchTotal ? (currentPage - 1) * pageSize + 1 : 0 }}–{{ Math.min(currentPage * pageSize, matchTotal) }} / {{ matchTotal }} 个文件</span>
            <label class="page-size-field">每页
              <select v-model.number="pageSize" aria-label="每页文件数量"><option :value="10">10 条</option><option :value="20">20 条</option><option :value="50">50 条</option></select>
            </label>
            <div class="page-buttons">
              <button class="button secondary" :disabled="currentPage <= 1 || historyLoading || !!dateError" @click="currentPage--"><ChevronLeft :size="14" />上一页</button>
              <form class="page-jump" @submit.prevent="jumpToPage">
                <span>第</span>
                <input v-model="pageDraft" type="text" inputmode="numeric" maxlength="10" aria-label="跳转页码" title="输入页码，按回车或点击跳转" :disabled="historyLoading || !!dateError" @keydown.esc="pageDraft = String(currentPage)" />
                <span>/ {{ totalPages }} 页</span>
                <button type="submit" class="button secondary" :disabled="historyLoading || !!dateError">跳转</button>
              </form>
              <button class="button secondary" :disabled="currentPage >= totalPages || historyLoading || !!dateError" @click="currentPage++">下一页<ChevronRight :size="14" /></button>
            </div>
          </div>
          <div class="queue-footer">
            <div>
              <button
                v-if="selectedCount"
                class="text-button danger"
                :disabled="removing || running"
                @click="removeSelected"
              >
                <Trash2 :size="15" />移除所选 {{ selectedCount }} 项</button
              ><span v-else
                ><span class="footer-dot"></span
                >{{
                  inProgress
                    ? `${inProgress} 个文件上传或处理中`
                    : `共 ${totalDocuments} 个文件，${pending.length} 个待上传`
                }}</span
              >
            </div>
            <div class="footer-right">
              <span>{{
                isDemoMode ? "文件仅保留在当前页面" : "后台任务不受页面关闭影响"
              }}</span
              ><button
                class="button primary"
                :disabled="
                  !pending.length || running || removing || !knowledgeBase || basesLoading || !settingsReady
                "
                @click="startUpload"
              >
                <LoaderCircle
                  v-if="running"
                  :size="16"
                  class="spin"
                /><CloudUpload v-else :size="17" />{{
                  running
                    ? "上传中…"
                    : isDemoMode
                      ? "开始演示上传"
                      : "开始上传"
                }}<span
                  v-if="pending.length && !running"
                  class="button-count"
                  >{{ pending.length }}</span
                >
              </button>
            </div>
          </div>
        </section>
        <footer class="page-footer">
          <span>知擎 · 文档管理</span
          ><span
            ><ShieldCheck :size="13" />{{
              isDemoMode ? "演示环境" : "服务已连接"
            }}</span
          >
        </footer>
      </main>
      <main v-if="activeView === 'query'">
        <QueryWorkspace :key="queryAccessVersion" :bases="bases" :bases-loading="basesLoading" :bases-error="basesError" @reload-bases="loadBases" />
      </main>
    </div>
    <div v-if="toast" class="toast" role="status" aria-live="polite">
      <Info :size="18" /><span>{{ toast }}</span
      ><button class="icon-button" aria-label="关闭提示" @click="toast = ''">
        <X :size="16" />
      </button>
    </div>
    <dialog
      ref="dialog"
      :class="{ 'preview-dialog': modal === 'preview' }"
      aria-labelledby="modal-title"
      @cancel.prevent="closeModal"
      @click="$event.target === dialog && closeModal()"
    >
      <div class="modal-content">
        <div class="modal-heading">
          <h2 id="modal-title">{{ modal === "preview" ? "文件预览" : modal === "remove" ? "移除文档" : "文档上传指南" }}</h2>
          <button
            class="icon-button"
            aria-label="关闭对话框"
            :disabled="removing"
            @click="closeModal"
          >
            <X :size="20" />
          </button>
        </div>
        <div v-if="modal === 'preview' && previewItem" class="preview-content">
          <div class="preview-summary">
            <div class="preview-file">
              <div class="preview-file-icon" :class="previewItem.extension"><FileText :size="26" /></div>
              <div class="preview-file-info">
                <strong>{{ previewItem.name }}</strong>
                <div class="preview-file-meta">
                  <span class="preview-base"><BookOpen :size="13" />{{ documentBaseName(previewItem) }}</span>
                  <span class="preview-format">{{ previewItem.extension.toUpperCase() }}</span>
                  <span>{{ formatSize(previewItem.size) }}</span>
                </div>
              </div>
            </div>
            <a v-if="previewUrl" class="button primary preview-download" :href="previewUrl" :download="previewItem.name">
              <ArrowDownToLine :size="17" /><span>下载原文件</span>
            </a>
          </div>
          <div v-if="previewLoading" class="preview-message" role="status">
            <LoaderCircle :size="24" class="spin" />正在加载预览…
          </div>
          <div v-else-if="previewError" class="preview-message" role="alert">
            <p>{{ previewError }}</p>
            <button class="button secondary" @click="openPreview(previewItem)">重新加载</button>
          </div>
          <PdfPreview v-else-if="previewItem.extension === 'pdf' && previewUrl" :key="previewUrl" :url="previewUrl" :name="previewItem.name" />
          <template v-else-if="previewData">
            <div v-if="previewSheets.length" class="preview-sheet-tabs" aria-label="工作表选择">
              <button v-for="sheet in previewSheets" :key="sheet" :class="{ selected: previewSheet === sheet }" :aria-pressed="previewSheet === sheet" @click="previewSheet = sheet"><Layers3 :size="14" />{{ sheet }}</button>
            </div>
            <p v-if="previewData.truncated" class="preview-notice">文件内容较多，当前仅展示部分内容。下载原文件可查看完整内容。</p>
            <p v-for="(warning, index) in previewData.warnings" :key="index" class="preview-notice">{{ warning.message }}</p>
            <div class="preview-body">
              <p v-if="!previewElements.length" class="preview-message">此文件没有可预览的正文内容。</p>
              <div v-for="(element, index) in previewElements" :key="index" class="preview-element">
                <template v-if="element.table_rows.length">
                  <span v-if="element.cell_range" class="preview-range">{{ element.cell_range }}</span>
                  <div class="preview-table-scroll">
                    <table class="preview-table">
                      <tbody><tr v-for="(row, rowIndex) in element.table_rows" :key="rowIndex"><td v-for="(cell, cellIndex) in row" :key="cellIndex">{{ cell }}</td></tr></tbody>
                    </table>
                  </div>
                </template>
                <h3 v-else-if="element.kind === 'heading'">{{ element.text }}</h3>
                <pre v-else-if="element.kind === 'code'">{{ element.text }}</pre>
                <p v-else>{{ element.text }}</p>
              </div>
            </div>
          </template>
        </div>
        <div v-else-if="modal === 'remove'" class="help-content">
          <p>确认移除 {{ removalTargets.length }} 个文件？</p>
          <ul>
            <li v-for="item in removalTargets" :key="item.id">{{ item.name }}</li>
          </ul>
          <p>此操作不可恢复。已上传文档的原文件、全部修订、分块、向量、索引及处理历史将永久删除，后台任务会停止。未上传的文件只从列表移除。</p>
          <p v-if="removalError" class="file-error" role="alert">{{ removalError }}。未移除的文件已保留，请重试。</p>
          <div class="modal-actions">
            <button class="button secondary" :disabled="removing" @click="closeModal">取消</button>
            <button class="button primary" :disabled="removing" @click="confirmRemoval">
              <LoaderCircle v-if="removing" :size="16" class="spin" />
              {{ removing ? "移除中…" : "确认移除" }}
            </button>
          </div>
        </div>
        <div v-else class="help-content">
          <p>
            将 PDF、DOCX、XLS/XLSX、Markdown 或 TXT 文档添加到列表，每份不超过
            50 MB。旧版 DOC 文件暂不支持。
          </p>
          <ol>
            <li>选择目标知识库，解析参数由系统统一管理。</li>
            <li>拖拽或选择文档，检查文件列表。</li>
            <li>按知识库、状态和上传日期查询文档，翻页查看历史记录。处理失败可直接重试，移除文档会停止后台任务并从知识库移除。</li>
          </ol>
          <div class="help-note">
            {{
              isDemoMode
                ? "当前为本地演示模式。文件不会发送至服务器，刷新后列表清空。知识库选项为演示配置；解析、索引与权限将在后端接入后实现。"
                : "列表展示当前账号可读取的知识库文档；上传需要写入授权。文档默认部门内部可见，跨部门需明确共享，共享对象还需知识库访问权限与对应密级。更新文件保留原密级。权限入口可管理文档共享与密级，知识库管理员可管理知识库授权。退出登录会结束所有设备上的会话。"
            }}
          </div>
          <button class="button primary" @click="closeModal">我知道了</button>
        </div>
      </div>
    </dialog>
    <PermissionDialog v-if="permissionTarget" :key="`${permissionTarget.kind}:${permissionTarget.id}:${permissionDialogVersion}`" :target="permissionTarget" :clearance="clearance" @close="permissionTarget = null" @changed="refreshAccessState()" @inaccessible="permissionInaccessible" />
  </div>
</template>
