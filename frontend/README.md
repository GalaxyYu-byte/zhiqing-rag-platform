# 知擎前端

Vue 3 + TypeScript + Vite 登录、文档管理与带引用的多轮问答页面，默认接入本项目 FastAPI 后端。

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
`npm test` 验证登录状态、并发刷新、会话撤销、租户切换、上传及检索的鉴权重试、检索范围与取消请求。
正式部署建议通过同一 HTTPS 来源代理 `/api`。如果使用接口绝对 URL，需设置后端
CORS 精确来源并允许凭据，以及 `Authorization`、`X-Requested-With` 请求头；
浏览器还需允许刷新 Cookie 的站点关系。

## 功能

- 邮箱与密码登录，登录后确认租户；即使只有一个租户也需确认后进入工作空间。
- 刷新页面时自动恢复登录；接口明确返回登录凭据过期后自动刷新并重试一次。
- 显示当前用户和租户；“退出所有设备”成功后清除本页状态并通知同来源其他标签页。
- 文件选择、拖拽和批量添加，支持 PDF、DOCX、XLS/XLSX、MD/Markdown、TXT。
- 50 MiB 上限、空文件、扩展名和重复项校验，后端继续校验内容与 MIME。
- 可读知识库筛选与文档浏览；上传只列出 WRITE/ADMIN 知识库，加载失败时可重试。
- 上传密级选择：内部公开、部门内部（默认）、机密，仅可选择当前账号密级范围内的值。
- 知识库管理员按用户或部门管理 READ/WRITE/ADMIN 授权；文档有管理权限的成员可按用户或部门明确共享、调整密级。
- 实际网络上传进度、后台处理阶段、服务端错误、失败处理重试、取消上传请求。
- 后端按文档分页，主列表只显示最新修订；可展开历史版本，默认每页 10 个文档，可切换 20/50 个。
- 文件名、知识库、状态、上传日期范围组合筛选，切换条件回到第一页。
- 批量移除、原文件预览与下载、只读解析配置、示例文档下载。
- 已上传文档可“更新文件”：保留文档 ID，新增修订；显示当前发布版本和修订历史。
- 响应式布局、键盘焦点、原生模态框、无障碍状态提示。
- 智能问答：输入问题、选择可读知识库，生成回答或仅检索证据；SSE 实时显示处理阶段，通过核验后逐段显示回答，每段引用可跳到原文。
- 会话创建与切换、会话和消息分页、刷新与断线恢复；稳定请求 ID 重试，历史回答由服务器按当前权限重新确认。

## 用前端测试检索

登录后点击左侧“智能问答”，输入已入库文档中的问题，点击“生成回答”；也可点击“仅检索证据”。
可选择单个知识库缩小范围，默认检索全部可读知识库；READ 权限即可使用。
回答展示逐段引用和逐字引文；资料不足或检查失败时不展示草稿。页码和章节只在原文解析提供时显示。

“生成回答”调用 `POST /api/chat/sessions/{session_id}/answer/stream`，提交 `query`、
稳定 `client_request_id` 和可选的 `knowledge_base_ids`；“仅检索证据”调用
`POST /api/query/search`，只提交问题与范围。
候选数、最终数量和重排开关由服务端 QueryPlan 决定。可通过
`VITE_QUERY_SEARCH_URL` 配置检索接口，聊天使用同来源 `/api/chat/sessions`。
旧单轮客户端及 `VITE_QUERY_ANSWER_URL`、`VITE_QUERY_ANSWER_STREAM_URL` 配置继续保留。
`VITE_CHAT_STREAM_TIMEOUT_MS` 默认 180000，可调整到 600000，代理读取预算需相应配置。
后端需配置 Analyzer、Embedding、Reranker、回答模型和检索数据库；
回答模型与权限、引用检查说明见 [回答接口](../QUERY_ANSWERING.md)。

查看“检索详情”可检查各召回分支、融合候选数量、重排状态及路由原因。
澄清问题、闲聊、空结果和降级均单独提示；HTTP 200 不代表一定找到原文。
“元数据筛选”支持部门、文件名、业务编号、文件类型、业务状态、上传时间和业务日期。
文档计数显示精确总数，文档清单可分页；分页沿用原请求的范围和条件。
上传可填写业务属性，具有 WRITE 权限的文档可在详情中编辑元数据；并发修改会提示刷新。
具体口径和接口见 [统计与元数据过滤](../QUERY_METADATA.md)。
分析耗时与检索耗时分开显示，检索内部分阶段并行执行，耗时不能直接相加。
“本次等待”记录浏览器从提交到接收结果的完整等待，包含网络、身份刷新、范围校验以及回答生成和复核。
页面指出耗时最多的已记录阶段；该指标帮助定位等待位置，不将检索总耗时与细分阶段重复比较。
分析降级时，结果标签同时显示“分析降级”，并展示具体原因说明；详情保留
`analysis.degradation_reason` 原因码，避免把格式或原文依据校验失败误判为超时。
可取消等待；修改问题、范围、切换页面、退出登录或收到权限变更通知后清除旧结果并中止在途请求。
问题、完整回答和证据保存于服务器；浏览器只按用户、租户缓存当前会话 ID，不持久化回答。
聊天流默认最多等待 3 分钟，连接前凭据过期沿用统一登录刷新策略。
流开始后不自动重连；错误、断线或缺少完成事件会清除临时结果，查询该轮保存状态。
已成功轮次直接恢复；执行中轮次有限轮询，失败、取消和中断明确显示，新提问才生成新请求 ID。
模型草稿完整复核后才发送回答段落，等待期间显示分析、检索、生成和复核阶段。
生产反向代理需关闭 SSE 响应缓冲，并允许覆盖聊天执行和发送预算的读取等待，默认建议至少 180 秒。
离线 demo 模式不调用检索接口，需切换 API 模式并登录才能实测。

服务端表准备、追问改写、幂等、撤权与恢复边界详见 [多轮会话与面试文档](../MULTI_TURN_CHAT_INTERVIEW.md)。

## 接口

`GET /api/knowledge-bases?access=READ` 返回可读知识库及 `permission`，供浏览筛选。
不传 `access` 的服务端接口仍默认 WRITE；前端服务函数也保留 WRITE 默认值。
实际 ID 以字符串传输。上传下拉框只保留 WRITE/ADMIN。
`POST /api/documents/upload` 使用 multipart/form-data，字段为 `file`、
`knowledge_base_id`、`confidentiality`（0 内部公开、1 部门内部、2 机密）。
解析参数由服务端 `UPLOAD_CHUNK_SIZE` 和 `UPLOAD_CHUNK_OVERLAP`
管理，用户传入的分块参数不会覆盖配置。`GET /api/document-upload-settings` 返回只读值、
当前账号 `clearance` 与 `department_id`；权限配置加载成功前禁用上传。
只有响应确认 `status=STORED`
且包含文档及任务 ID 才显示“等待处理”。原文件保存至 MinIO，数据库原子创建草稿、
修订、索引批次和持久化任务，接口返回 202。独立 Worker 自动解析、分块、向量化，
完整入库并发布索引后，页面显示“已入库”和分块数。

`POST /api/documents/{id}/revisions` 提交 `file` 和 `expected_revision_id`，
返回 202 和新修订的任务 ID。若列表中的修订已过期，服务端返回 409，刷新列表后
再更新。更新修订保留文档原密级，不提交新的 `confidentiality`。
后台处理期间旧发布批次继续可用；处理失败可直接重试该修订的任务。
相同内容的文件无需再次上传，可使用任务重试。Redis 会复用未变化分块的向量。

`GET /api/document-tasks/page` 支持 `offset`、`limit`、`knowledge_base_id`、`search`、
`status`、`date_from`、`date_to`，按文档最新修订返回当前页、匹配总数、文档总数及状态数量。
日期按北京时间计算，包含开始和结束日期当天；分页不再受最近 50 条限制。
`GET /api/documents/{id}/revisions` 返回该文档的修订历史，供展开和预览旧原文件。
`GET /api/document-tasks/{id}` 查询状态，
`POST /api/document-tasks/{id}/retry` 重新处理失败任务，无需再次上传原文件。
进度为处理阶段进度，并非按文档字节或模型 token 计算的精确完成百分比。

`DELETE /api/documents/{id}` 在验证租户、密级和知识库写权限后永久删除文档，返回 204。
先撤销发布指针并取消任务租约，再清理全部修订的 MinIO 原文件（含历史版本和删除标记），
最后在数据库事务中删除引用来源、图谱投影记录、任务事件、执行尝试、任务、分块及向量、
索引批次、文件修订和文档记录。原文件清理失败时返回 503，保留停用文档及修订用于重试；
页面刷新后仍显示待完成删除的记录。全部清理完成后才返回成功。操作不可恢复。
可通过 `VITE_DOCUMENTS_URL` 配置文档接口前缀（默认 `/api/documents`）。

接口接入层位于 `src/services/upload.ts`，所有业务接口与实际 XHR 上传都通过
`src/services/auth-client.ts` 的统一鉴权策略。上传详情见根目录 README。

## 部门范围与授权管理

列表、历史版本、预览和下载使用 READ 权限；更新、删除、重试使用服务端返回的
`can_write` 控制入口，权限管理使用 `can_manage_permissions`。按钮控制只改善交互，
所有 API 都继续由后端校验租户、知识库权限、密级与文档范围。

新文档默认部门内部，仅所属部门可见，跨部门需明确共享。账号没有部门时，非公开
文档默认仅本人及明确共享对象可见。内部公开文档仍需知识库访问权限；共享对象
还需知识库访问权限与对应密级，明确共享不会绕过这两个条件。

选中 ADMIN 知识库后可通过“管理授权”打开知识库授权弹窗，按当前租户的用户或部门
添加、调整或撤销 READ/WRITE/ADMIN。服务端阻止删除或降级最后一位可用管理员。
文档行“权限”弹窗可按用户或部门共享 READ/WRITE，并修改文档密级（不超过本人密级）。

权限接口位于 `src/services/permissions.ts`：

- `GET/POST /api/knowledge-bases/{id}/permissions`，`DELETE /api/knowledge-bases/{id}/permissions/{grant_id}`。
- `GET/POST /api/documents/{id}/permissions`，`DELETE /api/documents/{id}/permissions/{grant_id}`。
- `POST /api/documents/{id}/classification` 提交 `{confidentiality}`。

添加授权提交用户或部门其中一个 ID 和 `permission`；变更返回完整授权视图。
保存后关闭旧预览、撤销浏览器 Blob URL、中止在途资源请求、清除列表与历史缓存，
再刷新知识库、当前账号权限配置和文档列表。本人管理权限失效后关闭管理弹窗。
同来源标签页通过 BroadcastChannel 同步清理与重新加载，其他设备在后续 API 请求时
由服务端执行最新授权。弹窗关闭或登录身份变更会取消其在途请求。

## 登录与会话

账号由管理员开通，采用邮箱与密码登录，当前不提供注册或找回密码页面。
`POST /api/auth/login` 提交 `{email,password}`，返回短期 `selection_token`、用户及租户列表。
前端仅在内存保留票据，选择租户后调用 `POST /api/auth/select-tenant`，提交
`{selection_token,tenant_id}`。票据过期或已消费时回到邮箱登录。

选择租户成功后，服务端返回 15 分钟有效的 Access Token，并设置 HttpOnly Refresh
Cookie。Access Token 仅存在 JavaScript 内存，通过 `Authorization: Bearer` 携带；
不写入 localStorage 或 sessionStorage。Refresh Token 由浏览器 Cookie 管理，
JavaScript 不读取，服务端 Redis 保存刷新校验信息，数据库保留会话审计。
刷新轮换 Refresh Token，固定登录 7 天后需重新登录，不随刷新延长。

页面重载调用 `POST /api/auth/refresh` 恢复身份。在业务请求中，仅
`401 detail.code=ACCESS_TOKEN_EXPIRED` 触发一次刷新和一次原请求重试；403 或其他
401 不触发刷新。并发请求共用同一个刷新 Promise；迟到的旧凭据 401 直接使用已更新
的 Access Token。支持 Web Locks 的浏览器对刷新、租户确认和退出的 Cookie 操作
进行跨标签串行协调，减少共享 Cookie 轮换冲突；不支持 Web Locks 的浏览器只保证
单标签内协调，建议使用现代 Chrome、Edge、Firefox 或 Safari。

`POST /api/auth/logout` 撤销当前用户所有设备的会话。成功后业务页面卸载，取消轮询、
上传和预览请求，并清空当前页面文档与身份状态。同来源其他标签页通过 BroadcastChannel
收到退出通知；其他设备的 Access Token 在下一次请求时被服务端立即拒绝。
若自动恢复、刷新或退出遇到服务依赖暂不可用（503），显示可重试错误；已登录的页面
保留会话状态，退出重试成功后才清除身份。租户或用户在恢复时改变会卸载旧工作空间，
避免旧租户请求使用新身份重试。

所有 Cookie 请求使用 `credentials: include`，认证写请求携带
`X-Requested-With: XMLHttpRequest`。前端仅在身份恢复或租户确认成功后挂载业务页面，
所以登录前不会加载知识库或文档列表。

## 配置与演示

复制 `.env.example` 为 `.env.local`，按需修改代理目标或接口地址，修改后重启 Vite。
默认 `VITE_UPLOAD_MODE=api`；仅在需要离线预览时设为 `demo`，此时使用演示知识库和
模拟进度，不向服务器发送文件或认证请求，直接进入独立演示页面。
演示模式会明确显示在页面中。生产构建保持 `api` 模式。

未提交的本地文件列表仅保留在当前页面内存；已提交任务在服务器持久化。
取消上传请求不保证服务器尚未保存文件和任务。“移除”对未上传文件只清理列表；
对已上传文档先弹出不可恢复的永久删除确认框，再调用后端彻底删除，刷新后不再恢复。
批量移除中的失败项会保留，可重试。“刷新列表”重新查询当前页，不删除文档。
