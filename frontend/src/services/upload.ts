import { authClient, authenticatedFetch } from "./auth";

export const uploadEndpoint =
  import.meta.env.VITE_UPLOAD_URL?.trim() || "/api/documents/upload";
export const isDemoMode = import.meta.env.VITE_UPLOAD_MODE === "demo";
export const allowedExtensions = [
  "pdf",
  "docx",
  "xls",
  "xlsx",
  "md",
  "markdown",
  "txt",
];
export const maxFileSize = 50 * 1024 * 1024;

export interface KnowledgeBaseOption {
  id: string;
  name: string;
  description: string;
  permission: "READ" | "WRITE" | "ADMIN";
}

export interface UploadResult {
  document_id: string;
  revision_id: string;
  generation_id: string;
  task_id: string;
  status: "STORED";
  processing_status: "PENDING";
}

export interface DocumentTaskStatus {
  task_id: string;
  document_id: string;
  revision_id: string;
  revision_no: number;
  generation_id: string;
  desired_revision_id: string | null;
  active_generation_id: string | null;
  knowledge_base_id: string;
  original_filename: string;
  file_size: number;
  status:
    "PENDING" | "RUNNING" | "RETRY_WAIT" | "SUCCEEDED" | "FAILED" | "CANCELLED";
  stage: string | null;
  progress: number;
  attempt_count: number;
  max_attempts: number;
  error_code: string | null;
  chunk_count: number;
  document_status: string;
  created_at: string;
  confidentiality: number;
  department_id: string | null;
  can_write: boolean;
  can_manage_permissions: boolean;
}

const tasksEndpoint =
  import.meta.env.VITE_DOCUMENT_TASKS_URL || "/api/document-tasks";

async function taskRequest<T>(url: string, method = "GET", signal?: AbortSignal): Promise<T> {
  const response = await authenticatedFetch(url, {
    method,
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15_000)]) : AbortSignal.timeout(15_000),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      errorMessage(body, `处理任务请求失败（${response.status}）`),
    );
  return body as T;
}

export interface DocumentTaskPage {
  items: DocumentTaskStatus[];
  total: number;
  overall_total: number;
  overall_size: number;
  counts: Record<string, number>;
}
export async function fetchDocumentTasks(query: URLSearchParams, signal: AbortSignal): Promise<DocumentTaskPage> {
  const response = await authenticatedFetch(`${tasksEndpoint}/page?${query}`, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(body, "文档列表查询失败，请重试"));
  return body as DocumentTaskPage;
}
export async function fetchDocumentRevisions(documentId: string, signal: AbortSignal): Promise<DocumentTaskStatus[]> {
  const response = await authenticatedFetch(
    `${documentsEndpoint}/${encodeURIComponent(documentId)}/revisions`,
    { signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]) },
  );
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(body, "修订历史查询失败，请重试"));
  return body as DocumentTaskStatus[];
}
export const fetchUploadSettings = (signal?: AbortSignal) =>
  taskRequest<{ chunk_size: number; chunk_overlap: number; clearance: number; department_id: string | null }>("/api/document-upload-settings", "GET", signal);
export const fetchDocumentTask = (id: string) =>
  taskRequest<DocumentTaskStatus>(`${tasksEndpoint}/${encodeURIComponent(id)}`);
export const retryDocumentTask = (id: string) =>
  taskRequest<DocumentTaskStatus>(
    `${tasksEndpoint}/${encodeURIComponent(id)}/retry`,
    "POST",
  );

export async function deleteDocument(id: string): Promise<void> {
  const endpoint =
    import.meta.env.VITE_DOCUMENTS_URL?.trim() || "/api/documents";
  const response = await authenticatedFetch(`${endpoint}/${encodeURIComponent(id)}`, {
    method: "DELETE",
    signal: AbortSignal.timeout(15_000),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(errorMessage(body, `文档移除失败（${response.status}）`));
  }
}

export interface DocumentPreview {
  elements: {
    kind: string;
    text: string;
    sheet_name: string | null;
    cell_range: string | null;
    table_rows: string[][];
  }[];
  warnings: { message: string }[];
  truncated: boolean;
}

const documentsEndpoint = import.meta.env.VITE_DOCUMENTS_URL?.trim() || "/api/documents";

export async function fetchOriginalDocument(result: UploadResult, signal: AbortSignal): Promise<Blob> {
  const response = await authenticatedFetch(
    `${documentsEndpoint}/${encodeURIComponent(result.document_id)}/content?revision_id=${encodeURIComponent(result.revision_id)}`,
    { signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]) },
  );
  if (!response.ok) throw new Error(errorMessage(await response.json().catch(() => null), "原文件读取失败"));
  return response.blob();
}

export async function fetchDocumentPreview(
  file: File | null, result: UploadResult | undefined, signal: AbortSignal,
): Promise<DocumentPreview> {
  const body = file ? new FormData() : undefined;
  if (file) body!.append("file", file);
  const url = file
    ? `${documentsEndpoint}/preview`
    : `${documentsEndpoint}/${encodeURIComponent(result!.document_id)}/preview?revision_id=${encodeURIComponent(result!.revision_id)}`;
  const response = await authenticatedFetch(url, {
    method: file ? "POST" : "GET", body,
    signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(data, "文件预览失败，请下载原文件查看"));
  return data as DocumentPreview;
}

function errorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (
      detail &&
      typeof detail === "object" &&
      "message" in detail &&
      typeof detail.message === "string"
    )
      return detail.message;
    if (Array.isArray(detail)) return "上传参数无效，请检查知识库和分块设置";
  }
  return fallback;
}

export async function fetchKnowledgeBases(access: "READ" | "WRITE" = "WRITE", signal?: AbortSignal): Promise<KnowledgeBaseOption[]> {
  const endpoint = import.meta.env.VITE_KNOWLEDGE_BASES_URL || "/api/knowledge-bases";
  const response = await authenticatedFetch(
    `${endpoint}${endpoint.includes("?") ? "&" : "?"}access=${access}`,
    { signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(15_000)]) : AbortSignal.timeout(15_000) },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(errorMessage(body, `知识库加载失败（${response.status}）`));
  }
  return response.json();
}

export interface UploadOptions {
  knowledgeBase: string;
  signal: AbortSignal;
  onProgress: (progress: number) => void;
  confidentiality?: number;
  // 存在 documentId 时改走新增修订接口；expectedRevisionId 用于并发版本校验。
  documentId?: string;
  expectedRevisionId?: string;
}

export function uploadDocument(
  file: File,
  options: UploadOptions,
): Promise<UploadResult | undefined> {
  if (options.signal.aborted)
    return Promise.reject(new DOMException("已取消", "AbortError"));
  if (isDemoMode) {
    return new Promise((resolve, reject) => {
      let progress = 0;
      const abort = () => {
        clearInterval(timer);
        options.signal.removeEventListener("abort", abort);
        reject(new DOMException("已取消", "AbortError"));
      };
      const timer = setInterval(() => {
        progress = Math.min(100, progress + 12);
        options.onProgress(progress);
        if (progress === 100) {
          clearInterval(timer);
          options.signal.removeEventListener("abort", abort);
          resolve(undefined);
        }
      }, 180);
      options.signal.addEventListener("abort", abort, { once: true });
    });
  }

  return authClient.authorizedRequest(
    (token, signal) => sendUpload(file, options, token, signal),
    async (response) => response.body,
    options.signal,
  ).then((response) => {
    const result = response.body as Partial<UploadResult> | null;
    if (response.status >= 200 && response.status < 300 && result?.status === "STORED" &&
      result.document_id && result.task_id && result.processing_status === "PENDING")
      return result as UploadResult;
    throw new Error(errorMessage(response.body, `服务器返回 ${response.status}，未确认文件保存成功`));
  });
}

function sendUpload(
  file: File, options: UploadOptions, token: string, signal: AbortSignal,
): Promise<{ status: number; body: unknown }> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("已取消", "AbortError"));
      return;
    }
    const request = new XMLHttpRequest();
    const abort = () => request.abort();
    const cleanup = () => signal.removeEventListener("abort", abort);
    request.open(
      "POST",
      options.documentId
        ? `${documentsEndpoint}/${encodeURIComponent(options.documentId)}/revisions`
        : uploadEndpoint,
    );
    request.withCredentials = true;
    request.setRequestHeader("Authorization", `Bearer ${token}`);
    request.setRequestHeader("X-Requested-With", "XMLHttpRequest");
    request.responseType = "json";
    request.timeout = 120_000;
    options.onProgress(0);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable)
        options.onProgress(Math.round((event.loaded / event.total) * 100));
    };
    request.onload = () => {
      cleanup();
      resolve({ status: request.status, body: request.response });
    };
    request.onerror = () => {
      cleanup();
      reject(new Error("连接失败，请检查网络和接口跨域配置"));
    };
    request.ontimeout = () => {
      cleanup();
      reject(new Error("上传超时，请重试"));
    };
    request.onabort = () => {
      cleanup();
      reject(new DOMException("已取消", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
    const body = new FormData();
    body.append("file", file);
    if (options.documentId) {
      // 知识库由后端按 documentId 查询，不信任浏览器传入的目标知识库。
      if (!options.expectedRevisionId) {
        cleanup();
        reject(new Error("缺少当前修订编号，请刷新列表后重试"));
        return;
      }
      body.append("expected_revision_id", options.expectedRevisionId);
    } else {
      body.append("knowledge_base_id", options.knowledgeBase);
      body.append("confidentiality", String(options.confidentiality ?? 1));
    }
    request.send(body);
  });
}
