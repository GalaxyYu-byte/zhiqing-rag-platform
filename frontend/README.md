# 知擎前端

Vue 3 + TypeScript + Vite 文档上传页面，默认接入本项目 FastAPI 后端。

## 启动

先在项目根目录运行 `uv run python -m zhiqing_rag.main`，再在本目录执行：

```powershell
npm ci
npm run dev
```

同时在项目根目录另开终端运行 `uv run python -m zhiqing_rag.workers.ingestion`。
Worker 不运行时任务会持久化等待，启动后自动领取；上传 HTTP 请求不等待模型调用。

访问 `http://127.0.0.1:5173`。Vite 将 `/api` 代理至 `http://127.0.0.1:8000`。
`npm run build` 执行 TypeScript 检查和生产构建，`npm run preview` 预览构建。
正式部署需要 Web 服务器代理 `/api` 或配置接口绝对 URL 并设置后端 CORS 来源。

## 功能

- 文件选择、拖拽和批量添加，支持 PDF、DOCX、XLS/XLSX、MD/Markdown、TXT。
- 50 MiB 上限、空文件、扩展名和重复项校验，后端继续校验内容与 MIME。
- 后端可写知识库列表，加载失败时显示错误并提供重试，未选知识库时禁止提交。
- 实际网络上传进度、后台处理阶段、服务端错误、失败处理重试、取消上传请求。
- 后端分页查询历史记录，默认每页 10 条，可切换 20/50 条，后台任务不受页面关闭影响。
- 文件名、知识库、状态、上传日期范围组合筛选，切换条件回到第一页。
- 批量移除、原文件预览与下载、只读解析配置、示例文档下载。
- 响应式布局、键盘焦点、原生模态框、无障碍状态提示。

## 接口

`GET /api/knowledge-bases` 返回有写权限的知识库（实际 ID 以字符串传输）。
`POST /api/documents/upload` 使用 multipart/form-data，字段为 `file`、
`knowledge_base_id`。解析参数由服务端 `UPLOAD_CHUNK_SIZE` 和 `UPLOAD_CHUNK_OVERLAP`
管理，用户传入的分块参数不会覆盖配置。`GET /api/document-upload-settings` 返回只读值。
只有响应确认 `status=STORED`
且包含文档及任务 ID 才显示“等待处理”。原文件保存至 MinIO，数据库原子创建草稿、
修订、索引批次和持久化任务，接口返回 202。独立 Worker 自动解析、分块、向量化，
完整入库并发布索引后，页面显示“已入库”和分块数。

`GET /api/document-tasks/page` 支持 `offset`、`limit`、`knowledge_base_id`、`search`、
`status`、`date_from`、`date_to`，返回当前页、匹配总数、全部记录总数及状态数量。
日期按北京时间计算，包含开始和结束日期当天；分页不再受最近 50 条限制。
`GET /api/document-tasks/{id}` 查询状态，
`POST /api/document-tasks/{id}/retry` 重新处理失败任务，无需再次上传原文件。
进度为处理阶段进度，并非按文档字节或模型 token 计算的精确完成百分比。

`DELETE /api/documents/{id}` 在验证租户、密级和知识库写权限后永久删除文档，返回 204。
先撤销发布指针并取消任务租约，再清理全部修订的 MinIO 原文件（含历史版本和删除标记），
最后在数据库事务中删除引用来源、图谱投影记录、任务事件、执行尝试、任务、分块及向量、
索引批次、文件修订和文档记录。原文件清理失败时返回 503，保留停用文档及修订用于重试；
页面刷新后仍显示待完成删除的记录。全部清理完成后才返回成功。操作不可恢复。
可通过 `VITE_DOCUMENTS_URL` 配置文档接口前缀（默认 `/api/documents`）。

接口接入层位于 `src/services/upload.ts`；后端当前使用仅限本机 dev 环境的开发身份，
正式登录与生产授权会话需后续接入。上传详情见根目录 README。

## 配置与演示

复制 `.env.example` 为 `.env.local`，按需修改代理目标或接口地址，修改后重启 Vite。
默认 `VITE_UPLOAD_MODE=api`；仅在需要离线预览时设为 `demo`，此时使用演示知识库和
模拟进度，不向服务器发送文件。演示模式会明确显示在页面中。

未提交的本地文件列表仅保留在当前页面内存；已提交任务在服务器持久化。
取消上传请求不保证服务器尚未保存文件和任务。“移除”对未上传文件只清理列表；
对已上传文档先弹出不可恢复的永久删除确认框，再调用后端彻底删除，刷新后不再恢复。
批量移除中的失败项会保留，可重试。“刷新列表”重新查询当前页，不删除文档。
