/**
 * RAG Lab 前端演示层
 *
 * 默认连接同源真实接口；访问 /?demo=1 时使用内置数据独立演示界面。
 */
const DEMO_MODE = new URLSearchParams(window.location.search).get("demo") === "1";

const api = {
  upload: "/documents/upload",
  indexTask: (taskId) => `/documents/index-tasks/${taskId}`,
  documents: "/documents",
  knowledgeBases: "/knowledge-bases",
  retrieval: "/retrieval/search",
};

const demoKnowledgeBases = [
  { id: 1, name: "人力资源知识库", document_count: 8 },
  { id: 2, name: "产品文档库", document_count: 11 },
  { id: 3, name: "客户服务知识库", document_count: 5 },
];

const documents = [
  {
    id: 1042,
    name: "员工手册_2026版.pdf",
    type: "pdf",
    size: "4.8 MB",
    status: "DONE",
    chunks: 86,
    storage: "rag-documents/hr/1042.pdf",
    updated: "8 分钟前",
  },
  {
    id: 1041,
    name: "产品需求说明_v3.md",
    type: "md",
    size: "342 KB",
    status: "PROCESSING",
    stage: "CHUNKING",
    stageLabel: "清洗分块",
    progress: 25,
    chunks: null,
    storage: "rag-documents/product/1041.md",
    updated: "正在处理",
  },
  {
    id: 1040,
    name: "客户服务常见问题.docx",
    type: "docx",
    size: "1.2 MB",
    status: "DONE",
    chunks: 42,
    storage: "rag-documents/support/1040.docx",
    updated: "昨天 16:42",
  },
  {
    id: 1039,
    name: "差旅报销管理办法.pdf",
    type: "pdf",
    size: "2.6 MB",
    status: "DONE",
    chunks: 58,
    storage: "rag-documents/finance/1039.pdf",
    updated: "09 月 08 日",
  },
];

const resultFixtures = [
  {
    chunkId: 2841,
    document: "员工手册_2026版.pdf",
    section: "第四章 · 休假管理",
    page: 18,
    score: 0.9284,
    content: "正式员工工作满一年后，每个自然年度可享受 10 天带薪年假。工作满五年后增加至 15 天。年假原则上应在当年度内使用，确因工作需要可延至次年第一季度。",
    highlights: ["10 天带薪年假", "15 天"],
  },
  {
    chunkId: 2843,
    document: "员工手册_2026版.pdf",
    section: "第四章 · 请假流程",
    page: 19,
    score: 0.8741,
    content: "员工申请年假时，应至少提前三个工作日在系统中提交申请。连续休假超过五个工作日的，需要经部门负责人和人力资源部门共同审批。",
    highlights: ["申请年假", "五个工作日"],
  },
  {
    chunkId: 3116,
    document: "差旅报销管理办法.pdf",
    section: "附则 · 出差期间休假",
    page: 27,
    score: 0.7316,
    content: "出差任务与个人休假相邻时，应分别记录出差日期与休假日期。个人休假期间发生的住宿、交通等费用不纳入差旅报销范围。",
    highlights: ["个人休假", "休假日期"],
  },
  {
    chunkId: 2829,
    document: "员工手册_2026版.pdf",
    section: "第三章 · 考勤制度",
    page: 14,
    score: 0.6847,
    content: "公司实行每日八小时工作制。员工因病或因私不能正常出勤时，应按照对应假别完成线上申请，并在规定时间内提供证明材料。",
    highlights: ["假别", "线上申请"],
  },
  {
    chunkId: 2850,
    document: "员工手册_2026版.pdf",
    section: "第四章 · 假期结转",
    page: 21,
    score: 0.6129,
    content: "未使用的法定年假按照公司年度结转规则处理。员工离职时，人力资源部门将根据已使用天数核算剩余年假。",
    highlights: ["剩余年假", "已使用天数"],
  },
  {
    chunkId: 3022,
    document: "客户服务常见问题.docx",
    section: "服务时间",
    page: 6,
    score: 0.4812,
    content: "客户服务团队在国家法定工作日提供在线支持，服务时间为上午九点至下午六点，节假日安排以公告为准。",
    highlights: ["法定工作日"],
  },
];

const state = {
  selectedFiles: [],
  queryRuns: 0,
  results: DEMO_MODE ? resultFixtures : [],
  knowledgeBases: DEMO_MODE ? demoKnowledgeBases : [],
};

const els = {
  rows: document.getElementById("documentRows"),
  search: document.getElementById("documentSearch"),
  status: document.getElementById("statusFilter"),
  visibleCount: document.getElementById("visibleCount"),
  results: document.getElementById("retrievalResults"),
  loading: document.getElementById("loadingState"),
  resultSummary: document.getElementById("resultSummary"),
  question: document.getElementById("questionInput"),
  topK: document.getElementById("topK"),
  topKValue: document.getElementById("topKValue"),
  minScore: document.getElementById("minScore"),
  minScoreValue: document.getElementById("minScoreValue"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

const stageLabels = {
  PENDING: "等待处理",
  RETRY_WAIT: "等待重试",
  PARSING: "文档解析",
  CLEANING: "内容清洗",
  CHUNKING: "清洗分块",
  EMBEDDING: "向量化",
  PERSISTING: "索引入库",
  DONE: "已完成",
  FAILED: "处理失败",
};

function normalizeDocument(document, task = document.latest_task) {
  const type = String(document.file_type || fileType(document.file_name)).toLowerCase();
  return {
    id: document.id,
    name: document.file_name,
    type,
    size: formatFileSize(document.file_size),
    status: task?.status || document.status,
    stage: task?.stage || document.status,
    stageLabel: stageLabels[task?.stage || document.status] || "处理中",
    progress: task?.progress_percent || 0,
    chunks: document.chunk_count || task?.total_chunks || null,
    storage: document.minio_path,
    updated: formatDate(document.indexed_at || document.uploaded_at),
    taskId: task?.task_id,
  };
}

function fileType(name) {
  return name.split(".").pop()?.toLowerCase() || "file";
}

function knowledgeBaseOptions(includeAll = false) {
  if (!state.knowledgeBases.length) return `<option value="">暂无可用知识库</option>`;
  const allOption = includeAll && state.knowledgeBases.length > 1
    ? `<option value="${state.knowledgeBases.map((item) => item.id).join(",")}">全部知识库</option>`
    : "";
  return allOption + state.knowledgeBases.map((item) => (
    `<option value="${item.id}">${escapeHtml(item.name)} · ${item.document_count || 0} 份文档</option>`
  )).join("");
}

function renderKnowledgeBaseOptions() {
  const retrievalSelect = document.getElementById("kbSelect");
  retrievalSelect.innerHTML = knowledgeBaseOptions(true);
  retrievalSelect.disabled = state.knowledgeBases.length === 0;
  const uploadSelect = document.getElementById("uploadKb");
  if (uploadSelect) {
    uploadSelect.innerHTML = knowledgeBaseOptions(false);
    uploadSelect.disabled = state.knowledgeBases.length === 0;
  }
}

function statusMarkup(doc) {
  if (doc.status === "PROCESSING") {
    return `
      <div class="progress-cell">
        <span class="status-badge processing">${escapeHtml(doc.stageLabel || "处理中")}</span>
        <div class="mini-progress"><span style="width:${doc.progress || 0}%"></span></div>
        <div class="mini-progress-label"><span>${escapeHtml(doc.stage || "PROCESSING")}</span><b>${doc.progress || 0}%</b></div>
      </div>`;
  }
  if (doc.status === "FAILED") {
    return `<span class="status-badge failed">处理失败</span>`;
  }
  return `<span class="status-badge">已完成</span>`;
}

function renderDocuments() {
  const keyword = els.search.value.trim().toLowerCase();
  const status = els.status.value;
  const rows = documents.filter((doc) => {
    const nameMatched = doc.name.toLowerCase().includes(keyword);
    const statusMatched = status === "all" || doc.status === status;
    return nameMatched && statusMatched;
  });

  els.visibleCount.textContent = rows.length;
  if (!rows.length) {
    els.rows.innerHTML = `<tr><td class="empty-row" colspan="6">没有找到符合条件的文档</td></tr>`;
    return;
  }

  els.rows.innerHTML = rows.map((doc) => `
    <tr data-document-id="${doc.id}">
      <td>
        <div class="file-cell">
          <span class="file-icon ${escapeHtml(doc.type)}">${escapeHtml(doc.type.toUpperCase())}</span>
          <div class="file-meta"><strong>${escapeHtml(doc.name)}</strong><small>${doc.id} · ${escapeHtml(doc.size)}</small></div>
        </div>
      </td>
      <td>${statusMarkup(doc)}</td>
      <td>${doc.chunks == null ? "—" : `${doc.chunks} 块`}</td>
      <td><span class="storage-code">${escapeHtml(doc.storage)}</span></td>
      <td>${escapeHtml(doc.updated)}</td>
      <td><button class="row-action" data-action="inspect" aria-label="查看 ${escapeHtml(doc.name)}"><svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button></td>
    </tr>
  `).join("");
}

function highlightContent(result) {
  let content = escapeHtml(result.content);
  result.highlights.forEach((term) => {
    const safeTerm = escapeHtml(term);
    content = content.replaceAll(safeTerm, `<mark>${safeTerm}</mark>`);
  });
  return content;
}

function currentResults() {
  const topK = Number(els.topK.value);
  const threshold = Number(els.minScore.value) / 100;
  return state.results.filter((item) => item.score >= threshold).slice(0, topK);
}

function renderResults(latency = 42) {
  const results = currentResults();
  els.resultSummary.textContent = `找到 ${results.length} 个相关分块 · ${latency} ms`;

  if (!results.length) {
    els.results.innerHTML = `
      <div class="no-results">
        <strong>${state.queryRuns ? "当前阈值下没有召回结果" : "输入问题开始第一次召回测试"}</strong>
        <p>${state.queryRuns ? "可以适当调低最低相似度，或换一个更贴近文档表述的问题。" : "结果会展示分块原文、文档位置和余弦相似度。"}</p>
      </div>`;
    return;
  }

  els.results.innerHTML = results.map((result, index) => `
    <article class="result-item" style="animation-delay:${index * 45}ms">
      <div class="rank">${String(index + 1).padStart(2, "0")}</div>
      <div class="result-content">
        <div class="result-source">
          <strong>${escapeHtml(result.document)}</strong><i></i>
          <span>${escapeHtml(result.section)}</span><i></i>
          <span>${result.page == null ? "页码未知" : `第 ${result.page} 页`}</span><i></i>
          <span>Chunk #${result.chunkId}</span>
        </div>
        <p>${highlightContent(result)}</p>
      </div>
      <div class="score-block">
        <strong>${result.score.toFixed(4)}</strong>
        <span>余弦相似度</span>
        <div class="score-bar"><i style="width:${result.score * 100}%"></i></div>
      </div>
    </article>
  `).join("");
}

function switchView(viewName) {
  const retrieval = viewName === "retrieval";
  document.getElementById("documentsView").classList.toggle("active", !retrieval);
  document.getElementById("retrievalView").classList.toggle("active", retrieval);
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === viewName));
  document.getElementById("pageCrumb").textContent = retrieval ? "召回测试" : "文档处理";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showToast(title, detail) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<i></i><div><strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span></div>`;
  document.getElementById("toastRegion").appendChild(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function modalMarkup() {
  return `
    <div class="modal-backdrop" id="uploadModal" role="presentation">
      <section class="modal" role="dialog" aria-modal="true" aria-labelledby="uploadTitle">
        <div class="modal-header">
          <div><p class="eyebrow">NEW DOCUMENT</p><h2 id="uploadTitle">上传文档</h2></div>
          <button class="icon-button" id="closeUpload" aria-label="关闭"><svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg></button>
        </div>
        <div class="drop-zone" id="dropZone" role="button" tabindex="0" aria-label="拖入文件或打开文件选择器">
          <input id="fileInput" type="file" accept=".pdf,.docx,.txt,.md,.xlsx,.xls" multiple />
          <span class="upload-orbit"><svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M5 20h14"/></svg></span>
          <strong>拖拽文件到此处，或 <button type="button" class="browse-button" id="browseFiles">点击选择</button></strong>
          <span>支持 PDF、DOCX、TXT、Markdown、Excel，单文件最大 50 MB</span>
        </div>
        <div class="selected-files" id="selectedFiles"></div>
        <div class="upload-options">
          <label><span>目标知识库</span><select id="uploadKb" ${state.knowledgeBases.length ? "" : "disabled"}>${knowledgeBaseOptions(false)}</select></label>
          <label><span>分块大小（服务端）</span><div class="input-suffix"><input id="chunkSize" type="number" value="512" disabled /><b>tokens</b></div></label>
          <label><span>重叠长度（服务端）</span><div class="input-suffix"><input id="chunkOverlap" type="number" value="64" disabled /><b>tokens</b></div></label>
        </div>
        <div class="async-callout">
          <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/></svg>
          <p><strong>上传和索引采用异步任务</strong><span>文件写入 MinIO 后页面即可关闭；解析、清洗、分块、向量化会在后台继续执行。</span></p>
        </div>
        <div class="modal-footer">
          <button class="secondary-button" id="cancelUpload">取消</button>
          <button class="primary-button" id="submitUpload" disabled>上传并开始处理</button>
        </div>
      </section>
    </div>`;
}

function openUploadModal() {
  document.body.insertAdjacentHTML("beforeend", modalMarkup());
  state.selectedFiles = [];
  const modal = document.getElementById("uploadModal");
  const zone = document.getElementById("dropZone");
  const input = document.getElementById("fileInput");
  const listeners = new AbortController();
  const listenerOptions = { signal: listeners.signal };
  let dragDepth = 0;

  const close = () => {
    listeners.abort();
    modal.remove();
  };
  const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  const resetDragState = () => {
    dragDepth = 0;
    zone.classList.remove("dragover");
  };
  const extractFiles = (dataTransfer) => {
    const itemFiles = [...(dataTransfer?.items || [])]
      .filter((item) => item.kind === "file")
      .map((item) => item.getAsFile())
      .filter(Boolean);
    return itemFiles.length ? itemFiles : [...(dataTransfer?.files || [])];
  };

  document.getElementById("closeUpload").addEventListener("click", close, listenerOptions);
  document.getElementById("cancelUpload").addEventListener("click", close, listenerOptions);
  modal.addEventListener("click", (event) => { if (event.target === modal) close(); }, listenerOptions);
  input.addEventListener("change", () => addFiles(input.files), listenerOptions);
  zone.addEventListener("click", (event) => {
    if (!event.target.closest(".browse-button")) input.click();
  }, listenerOptions);
  document.getElementById("browseFiles").addEventListener("click", () => input.click(), listenerOptions);
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  }, listenerOptions);

  // 在 document 级别阻止浏览器把外部文件当成页面打开；内嵌 WebView 中
  // dragenter 的 target 经常不是上传框本身，因此不能只监听 zone。
  document.addEventListener("dragenter", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    dragDepth += 1;
    zone.classList.add("dragover");
  }, listenerOptions);
  document.addEventListener("dragover", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    zone.classList.add("dragover");
  }, listenerOptions);
  document.addEventListener("dragleave", (event) => {
    if (!hasFiles(event)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) zone.classList.remove("dragover");
  }, listenerOptions);
  document.addEventListener("drop", (event) => {
    if (!hasFiles(event)) return;
    event.preventDefault();
    const droppedFiles = extractFiles(event.dataTransfer);
    resetDragState();
    if (droppedFiles.length) addFiles(droppedFiles);
  }, listenerOptions);
  document.getElementById("submitUpload").addEventListener("click", async () => {
    try {
      await submitUpload();
    } catch (error) {
      showToast("上传失败", error.message);
      const button = document.getElementById("submitUpload");
      if (button) {
        button.disabled = state.selectedFiles.length === 0;
        button.textContent = "上传并开始处理";
      }
    }
  }, listenerOptions);
  renderKnowledgeBaseOptions();
  document.addEventListener("keydown", function onEscape(event) {
    if (event.key === "Escape" && document.getElementById("uploadModal")) {
      close();
    }
  }, listenerOptions);
}

function addFiles(fileList) {
  const allowed = ["pdf", "docx", "txt", "md", "xlsx", "xls"];
  [...fileList].forEach((file) => {
    const type = fileType(file.name);
    if (!allowed.includes(type)) {
      showToast("文件格式不支持", `${file.name} 未加入上传队列`);
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      showToast("文件超过 50 MB", `${file.name} 未加入上传队列`);
      return;
    }
    if (!state.selectedFiles.some((item) => item.name === file.name && item.size === file.size)) state.selectedFiles.push(file);
  });
  renderSelectedFiles();
}

function renderSelectedFiles() {
  const container = document.getElementById("selectedFiles");
  if (!container) return;
  container.innerHTML = state.selectedFiles.map((file, index) => {
    const type = fileType(file.name);
    return `
      <div class="selected-file">
        <span class="file-icon ${escapeHtml(type)}">${escapeHtml(type.toUpperCase())}</span>
        <div><strong>${escapeHtml(file.name)}</strong><small>${formatFileSize(file.size)}</small></div>
        <button class="remove-file" data-index="${index}" aria-label="移除 ${escapeHtml(file.name)}"><svg viewBox="0 0 24 24"><path d="m6 6 12 12M18 6 6 18"/></svg></button>
      </div>`;
  }).join("");
  container.querySelectorAll(".remove-file").forEach((button) => button.addEventListener("click", () => {
    state.selectedFiles.splice(Number(button.dataset.index), 1);
    renderSelectedFiles();
  }));
  document.getElementById("submitUpload").disabled = state.selectedFiles.length === 0 || state.knowledgeBases.length === 0;
}

async function submitUpload() {
  const selectedKbId = document.getElementById("uploadKb").value;
  if (!selectedKbId) {
    showToast("没有可用知识库", "请先在数据库中创建知识库");
    return;
  }
  const button = document.getElementById("submitUpload");
  button.disabled = true;
  button.textContent = "正在写入 MinIO…";

  if (!DEMO_MODE) {
    const form = new FormData();
    state.selectedFiles.forEach((file) => form.append("files", file));
    form.append("kb_id", selectedKbId);
    // 认证模块接入前使用测试用户 1；正式环境应由服务端从 JWT 获取用户 ID。
    form.append("uploaded_by", "1");
    const response = await fetch(api.upload, { method: "POST", body: form });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(error?.detail || `上传失败：${response.status}`);
    }
    const payload = await response.json();
    const newDocs = payload.items.map((item) => normalizeDocument(item.document, item.index_task));
    documents.unshift(...newDocs);
    document.getElementById("uploadModal")?.remove();
    renderDocuments();
    showToast("文件已写入 MinIO", `已创建 ${newDocs.length} 个异步索引任务`);
    newDocs.forEach((doc) => {
      if (doc.taskId) pollIndexTask(doc.taskId, doc.id);
    });
    return;
  } else {
    await new Promise((resolve) => window.setTimeout(resolve, 720));
  }

  const files = [...state.selectedFiles];
  document.getElementById("uploadModal")?.remove();
  const newDocs = files.map((file, index) => ({
    id: 1043 + index,
    name: file.name,
    type: fileType(file.name),
    size: formatFileSize(file.size),
    status: "PROCESSING",
    stage: "PARSING",
    stageLabel: "文档解析",
    progress: 5,
    chunks: null,
    storage: `rag-documents/uploads/${Date.now()}-${file.name}`,
    updated: "刚刚",
  }));
  documents.unshift(...newDocs);
  renderDocuments();
  document.getElementById("documentCount").textContent = 24 + newDocs.length;
  document.getElementById("processingCount").textContent = documents.filter((item) => item.status === "PROCESSING").length;
  showToast("文件已写入 MinIO", `已创建 ${newDocs.length} 个异步索引任务，可在列表中查看进度`);
  newDocs.forEach((doc, index) => simulatePipeline(doc, index * 360));
}

async function pollIndexTask(taskId, documentId) {
  const maxAttempts = 900;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    try {
      const response = await fetch(api.indexTask(taskId));
      if (!response.ok) return;
      const task = await response.json();
      const doc = documents.find((item) => item.id === documentId);
      if (!doc) return;
      doc.status = task.status;
      doc.stage = task.stage;
      doc.stageLabel = stageLabels[task.stage] || task.stage;
      doc.progress = task.progress_percent;
      doc.chunks = task.total_chunks || doc.chunks;
      doc.updated = task.status === "DONE" ? "刚刚" : "正在处理";
      renderDocuments();
      document.getElementById("processingCount").textContent = documents.filter((item) => item.status === "PROCESSING" || item.status === "PENDING").length;
      if (["DONE", "FAILED", "CANCELED"].includes(task.status)) {
        if (task.status === "DONE") showToast("文档索引完成", `${doc.name} 已生成 ${task.total_chunks} 个向量分块`);
        if (task.status === "FAILED") showToast("文档索引失败", task.error_msg || doc.name);
        return;
      }
    } catch {
      // 临时网络失败时继续轮询；任务状态以服务端数据库为准。
    }
  }
}

async function loadDocuments() {
  if (DEMO_MODE) return;
  try {
    const response = await fetch(`${api.documents}?limit=100`);
    if (!response.ok) throw new Error(`文档列表加载失败：${response.status}`);
    const payload = await response.json();
    documents.splice(0, documents.length, ...payload.items.map((item) => normalizeDocument(item)));
    document.getElementById("documentCount").textContent = payload.total;
    document.getElementById("chunkCount").textContent = documents.reduce((sum, item) => sum + (item.chunks || 0), 0).toLocaleString("zh-CN");
    document.getElementById("processingCount").textContent = documents.filter((item) => item.status === "PROCESSING" || item.status === "PENDING").length;
    renderDocuments();
    documents.filter((item) => item.taskId && ["PENDING", "PROCESSING"].includes(item.status)).forEach((item) => pollIndexTask(item.taskId, item.id));
  } catch (error) {
    showToast("文档列表暂时不可用", error.message);
  }
}

async function loadKnowledgeBases() {
  if (DEMO_MODE) {
    renderKnowledgeBaseOptions();
    return;
  }
  try {
    const response = await fetch(api.knowledgeBases);
    if (!response.ok) throw new Error(`知识库列表加载失败：${response.status}`);
    const payload = await response.json();
    state.knowledgeBases = payload.items;
    renderKnowledgeBaseOptions();
    if (!payload.total) showToast("暂无知识库", "请先创建知识库后再上传文档");
  } catch (error) {
    state.knowledgeBases = [];
    renderKnowledgeBaseOptions();
    showToast("知识库列表暂时不可用", error.message);
  }
}

/** 演示任务状态流转；真实模式下应每 1~2 秒轮询 api.indexTask(taskId)。 */
function simulatePipeline(doc, delay = 0) {
  const stages = [
    { stage: "CLEANING", stageLabel: "内容清洗", progress: 15 },
    { stage: "CHUNKING", stageLabel: "清洗分块", progress: 25 },
    { stage: "EMBEDDING", stageLabel: "向量化", progress: 63 },
    { stage: "PERSISTING", stageLabel: "索引入库", progress: 91 },
    { stage: "DONE", stageLabel: "已完成", progress: 100 },
  ];
  stages.forEach((next, index) => window.setTimeout(() => {
    Object.assign(doc, next);
    if (next.stage === "DONE") {
      doc.status = "DONE";
      doc.chunks = 28 + Math.floor(Math.random() * 45);
      doc.updated = "刚刚";
      const processing = documents.filter((item) => item.status === "PROCESSING").length;
      document.getElementById("processingCount").textContent = processing;
      showToast("文档索引完成", `${doc.name} 已生成 ${doc.chunks} 个向量分块`);
    }
    renderDocuments();
  }, delay + (index + 1) * 1450));
}

async function runRetrieval() {
  const question = els.question.value.trim();
  if (!question) {
    showToast("请输入测试问题", "问题不能为空");
    els.question.focus();
    return;
  }
  if (!document.getElementById("kbSelect").value) {
    showToast("没有可用知识库", "请先创建知识库并完成文档索引");
    return;
  }

  const button = document.getElementById("runRetrieval");
  button.disabled = true;
  els.results.hidden = true;
  els.loading.hidden = false;

  try {
    let latency = 36 + Math.floor(Math.random() * 32);
    if (!DEMO_MODE) {
      const startedAt = performance.now();
      const kbIds = document.getElementById("kbSelect").value.split(",").map(Number);
      const response = await fetch(api.retrieval, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: question,
          kb_ids: kbIds,
          top_k: Number(els.topK.value),
          min_score: Number(els.minScore.value) / 100,
          metric: "cosine",
        }),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => null);
        throw new Error(error?.detail || `召回失败：${response.status}`);
      }
      latency = Math.round(performance.now() - startedAt);
      const payload = await response.json();
      state.results = payload.results.map((item) => ({
        chunkId: item.chunk_id,
        document: item.document,
        section: item.section || `Chunk ${item.chunk_index}`,
        page: item.page,
        score: item.score,
        content: item.content,
        highlights: [],
      }));
      latency = payload.latency_ms ?? latency;
    } else {
      await new Promise((resolve) => window.setTimeout(resolve, 620));
    }
    state.queryRuns += 1;
    renderResults(latency);
  } catch (error) {
    showToast("召回失败", error.message);
  } finally {
    els.loading.hidden = true;
    els.results.hidden = false;
    button.disabled = false;
  }
}

document.querySelectorAll(".nav-item").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));
document.getElementById("openUpload").addEventListener("click", openUploadModal);
document.getElementById("runRetrieval").addEventListener("click", runRetrieval);
els.search.addEventListener("input", renderDocuments);
els.status.addEventListener("change", renderDocuments);
els.topK.addEventListener("input", () => {
  els.topKValue.textContent = els.topK.value;
  if (state.queryRuns) renderResults();
});
els.minScore.addEventListener("input", () => {
  els.minScoreValue.textContent = (Number(els.minScore.value) / 100).toFixed(2);
  if (state.queryRuns) renderResults();
});
els.question.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") runRetrieval();
});
els.rows.addEventListener("click", (event) => {
  const button = event.target.closest('[data-action="inspect"]');
  if (!button) return;
  const row = button.closest("tr");
  const doc = documents.find((item) => item.id === Number(row.dataset.documentId));
  if (doc?.status === "DONE") {
    switchView("retrieval");
    showToast("已限定测试范围", `将优先查看 ${doc.name} 的召回结果`);
  } else {
    showToast("文档仍在处理中", "向量入库完成后即可参与召回测试");
  }
});

document.getElementById("modeBadge").lastChild.textContent = DEMO_MODE ? " 演示数据" : " 真实接口";
renderDocuments();
renderResults();
loadKnowledgeBases();
loadDocuments();
