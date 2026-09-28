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
  generation_id: string;
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
}

const tasksEndpoint =
  import.meta.env.VITE_DOCUMENT_TASKS_URL || "/api/document-tasks";

async function taskRequest<T>(url: string, method = "GET"): Promise<T> {
  const response = await fetch(url, {
    method,
    signal: AbortSignal.timeout(15_000),
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
  const response = await fetch(`${tasksEndpoint}/page?${query}`, {
    signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]),
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(errorMessage(body, "文档列表查询失败，请重试"));
  return body as DocumentTaskPage;
}
export const fetchUploadSettings = () =>
  taskRequest<{ chunk_size: number; chunk_overlap: number }>("/api/document-upload-settings");
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
  const response = await fetch(`${endpoint}/${encodeURIComponent(id)}`, {
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
  const response = await fetch(
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
  const response = await fetch(url, {
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

export async function fetchKnowledgeBases(): Promise<KnowledgeBaseOption[]> {
  const response = await fetch(
    import.meta.env.VITE_KNOWLEDGE_BASES_URL || "/api/knowledge-bases",
    { signal: AbortSignal.timeout(15_000) },
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

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    const abort = () => request.abort();
    const cleanup = () => options.signal.removeEventListener("abort", abort);
    request.open("POST", uploadEndpoint);
    request.responseType = "json";
    request.timeout = 120_000;
    request.upload.onprogress = (event) => {
      if (event.lengthComputable)
        options.onProgress(Math.round((event.loaded / event.total) * 100));
    };
    request.onload = () => {
      cleanup();
      if (
        request.status >= 200 &&
        request.status < 300 &&
        request.response?.status === "STORED" &&
        request.response?.document_id &&
        request.response?.task_id &&
        request.response?.processing_status === "PENDING"
      )
        resolve(request.response);
      else
        reject(
          new Error(
            errorMessage(
              request.response,
              `服务器返回 ${request.status}，未确认文件保存成功`,
            ),
          ),
        );
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
    options.signal.addEventListener("abort", abort, { once: true });
    const body = new FormData();
    body.append("file", file);
    body.append("knowledge_base_id", options.knowledgeBase);
    request.send(body);
  });
}
