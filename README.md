# 知擎 RAG 平台

新项目位于 `D:\code\zhiqing-rag-platform`，Python 固定为 **3.12**，使用 uv 管理环境和依赖。
当前已实现登录与租户选择、部门与知识库权限、文档上传与版本管理、后台解析与向量入库、离线图谱构建、问题分析与混合检索，以及带原文引用的单轮回答。Vue 页面支持文档管理、检索与 SSE 进度和核验后逐段回答。多轮会话、在线图谱召回及 Alembic 迁移仍待完善。

实体与技术方案文档单独维护；各业务模块将分阶段实现。

## 开始开发

在 PowerShell 中执行：

```powershell
cd D:\code\zhiqing-rag-platform
uv sync --locked
uv run python -m zhiqing_rag.main
```

启动后访问 `http://127.0.0.1:8000/docs`。
`GET /health/live` 检查进程；`GET /health/ready` 只读检查数据库、Redis、MinIO 和已启用的 Neo4j。
服务启动不连接中间件；就绪接口会在依赖不可用时返回 503。

## 配置

### 文档上传与前端联调

后端启动后，另开终端运行：

```powershell
cd D:\code\zhiqing-rag-platform\frontend
npm ci
npm run dev
```

访问 `http://127.0.0.1:5173`，Vite 默认把 `/api` 代理到 `http://127.0.0.1:8000`。
页面从 `GET /api/knowledge-bases?access=READ` 读取当前身份可读取的知识库，上传选项只保留 WRITE/ADMIN 知识库，向
`POST /api/documents/upload` 提交 `multipart/form-data`：`file`、实际数字
`knowledge_base_id`、`confidentiality`（默认 1，部门内部）。解析参数由服务端 `UPLOAD_CHUNK_SIZE`（默认 512）和
`UPLOAD_CHUNK_OVERLAP`（默认 48）统一管理，浏览器不能覆盖。

上传校验实际文件内容、声明 MIME、完整性、文件名和大小；默认上限 50 MiB，
请求体在 multipart 解析前限制为文件上限加 1 MiB。文件存入现有 MinIO 桶的
项目/租户/知识库范围内独立对象键，不覆盖同名文件。数据库事务创建 DRAFT 文档、
第一份不可变修订、期望修订指针、索引批次和持久化后台任务，并保存分块参数。
返回 **202 Accepted**，包含字符串形式的文档/修订/索引批次/任务 ID、SHA-256、MIME、
大小、创建时间以及 `STORED / PENDING` 状态。请求不执行解析或调用向量模型。

更新已有文档使用 `POST /api/documents/{document_id}/revisions`，提交
`multipart/form-data` 的 `file` 和 `expected_revision_id`。页面的“更新文件”按钮
会使用当前期望修订作为并发校验值；过期修订返回 409，相同文件内容也返回 409。
更新保留原 `document_id`，将新原文件写入独立对象键，新增递增的不可变修订、
索引批次和处理任务；尚未完成的旧修订任务会取消。处理新修订期间，原
`active_generation_id` 保持不变；成功后 Worker 原子切换，失败时旧批次继续可用。
解析和切片仍会重新执行，向量化按实际 `embedding_content` 和模型配置查 Redis 缓存，
只对未命中的分块调用模型。主列表按文档显示最新修订；“历史版本”展开后可查看和预览旧原文件。

再开一个终端启动独立后台进程：

```powershell
cd D:\code\zhiqing-rag-platform
uv run python -m zhiqing_rag.workers.ingestion
```

Worker 从项目独立 PostgreSQL 的 `background_task` 表领取任务，下载并校验 MinIO
原文件，随后解析、清洗、分块，先将正文和来源引用写入 `document_chunk`。图任务
此时即可开始；文档 Worker 同时调用 Embedding 模型生成 1024 维向量并补写到这些分块。
仅在向量完整性校验成功后，同一事务将
索引批次置为 READY、文档置为 PUBLISHED 并切换活动索引指针，任务置为 SUCCEEDED。
这表示文档已可供混合检索与单轮证据回答使用。

当前采用 **PostgreSQL 持久化任务队列 + 独立 Worker**，任务与上传记录同事务提交，
无需跨 Redis 写入和 Outbox 投递。该 Worker 不使用 ARQ；Redis 仅沿用现有向量缓存。
这是为了在当前 `REDIS_DB=0` 与旧项目共用的情况下避免启动共享 ARQ Worker。
将来切换到 ARQ 需分配独立逻辑库并实现 Outbox 投递，不能同时无租约地执行同一任务。

默认并发 2、每 2 秒检查队列、租约 90 秒、每 30 秒续约、单任务预算 1200 秒，
暂时性失败最多尝试 3 次，并按 15/30 秒退避。Worker 退出或宕机后，其他进程可在
租约过期后恢复任务；模型结果通过既有向量缓存复用。确定性解析/权限/模型配置错误
直接标记失败。旧租约和非期望修订不允许发布。解析在可终止子进程中执行，
避免 OCR 或 CPU 解析阻塞异步事件循环。失败后重试处理会创建新索引批次，保留
不可变失败批次和历史尝试记录，不覆盖已完成批次。

解析与分块完成后，Worker 先提交不含向量的 Chunk，同时写入独立的 `GRAPH` 任务和
`graph_projection` 记录。随后向量生成和 Graph Worker 抽取可并行进行；图谱结果不会
阻塞向量入库与文档发布，向量失败也不会取消图谱任务。文档仅在完整向量发布后可用。
配置 `NEO4J_ENABLED=true`、Neo4j 连接、
`DEEPSEEK_API_KEY` 后，单独启动 Graph Worker：

```powershell
uv run python -m zhiqing_rag.workers.graph
```

Graph Worker 按 Chunk 使用 DeepSeek JSON 模式抽取实体与关系，再严格校验 Schema、
关系端点、实体名称和逐字证据。无效事实逐条丢弃，只有校验通过的结果才入图。
实体名先做 Unicode、空格及大小写归一化；Neo4j 的
实体、断言、证据采用稳定键，重试同一分块会替换该分块旧断言。证据保存租户、知识库、
文档、修订、索引批次、Chunk、页码、章节、原文片段及引用位置。图谱只作为可重建投影；
查询时仍需按 PostgreSQL 当前文档/批次和授权范围过滤。`start-dev.ps1` 在
`NEO4J_ENABLED=true` 时会另开 Graph Worker 窗口。

图谱增量更新按同一文档的 Chunk 正文 SHA-256 筛选，并再次比较完整正文；
`chunk_index` 变化不会使相同正文重新抽取。命中相同 DeepSeek 模型和抽取规则版本时，
Graph Worker 从 PostgreSQL 的 Chunk 元数据读取已校验的抽取结果，跳过模型调用，
仍按新批次的 Chunk ID、页码和来源位置重新写入 Neo4j 证据。未命中的 Chunk 才请求
DeepSeek；成功的抽取结果先存入 PostgreSQL，以便图写入失败后重试复用。分块边界
变化导致正文变化时会重新抽取。上线前已经完成的旧图任务没有这份缓存，首次更新
这些 Chunk 时需要抽取一次。该功能不使用 Redis。

文档更新期间旧活动批次继续可用。新批次发布时，旧图投影在同一事务中标为
`STALE`，并向 outbox 写入可重试的旧批次清理事件；未完成的旧图任务在更新时即取消。
文档删除会先撤销可见性与租约，再留下独立于文档外键的图清理事件，因此文档记录
删除后仍可继续清理 Neo4j。Graph Worker 同时消费清理事件；清理只删目标文档或
批次的断言和证据，共享实体只有在无任何关系时才删除。图查询必须先调用
`authorized_graph_projection(session, identity, document_id)` 获取当前用户可读且 `READY` 的活动批次，再限定 Neo4j 结果的
租户、命名空间和批次；新图未就绪时仅使用向量检索。

Graph Worker 每完成一个 Chunk 就保存游标和实体/关系键，失败后从未完成的
Chunk 继续。模型连接异常、超时、429 或 5xx 使用带随机抖动的退避和 Worker
冷却，不会紧接着请求下一任务；402 会暂停该 Worker 的模型请求，需恢复额度后
重启 Worker 并显式重排失败任务。清理事件与模型调用独立，Neo4j 暂时不可用时
按延迟重试。修改上述 Worker 代码后需要重启运行中的 Graph Worker 进程。

数据库中旧的已发布分块可使用补录程序。默认只为当前活动索引中尚无图任务的批次
幂等创建任务，运行中的 Graph Worker 会继续处理；`--run` 可在程序内立即转换，
`--workers` 控制并发。先预览范围：

```powershell
uv run python scripts/backfill_graph.py --dry-run
uv run python scripts/backfill_graph.py --run --workers 2
```

可用 `--tenant-id`、`--kb-id`、`--document-id`、`--limit` 缩小范围。
失败任务可显式加 `--retry-failed` 重新排队；默认不重复处理已完成的图谱批次。
DeepSeek 返回额度错误时，回填程序停止派发后续任务；额度恢复后运行
`uv run python scripts/backfill_graph.py --retry-failed --run --workers 2` 继续处理。
任务失败并排查原因后，可加 `--retry-failed --run` 重试；原任务与尝试历史保留。

接口：页面通过 `GET /api/document-tasks/page` 分页查询有访问权限的文档最新修订，支持
`offset`、`limit`（默认 10、最多 100）、`knowledge_base_id`、`search`、`status`、
`date_from` 和 `date_to`。日期按北京时间计算，包含结束日期当天；响应返回
`items`、`total`、`overall_total`、`overall_size` 和状态数量 `counts`。
`GET /api/document-tasks` 保留最近 50 个任务的兼容接口；
`GET /api/document-tasks/{task_id}` 返回阶段、进度、重试次数、错误码、分块数和
文档发布状态；`POST /api/document-tasks/{task_id}/retry` 重试失败任务。
前端每 2.5 秒查询活跃任务，展示等待、解析分块、生成向量、入库、完成与失败，
页面关闭不停止后台任务。移除未上传文件仅清理本地列表；移除已上传文档需确认后
永久删除服务器文档并停止后台处理。`GET /api/document-upload-settings` 提供只读解析配置。

页面采用邮箱和密码登录，随后选择当前账号可用的租户。选定租户后签发 15 分钟的
Access Token 与 7 天的 Refresh Token；会话从选定租户起按固定期限到期，刷新不延长
7 天期限。现有用户密码沿用初始化数据的 scrypt 哈希格式，不提供注册和找回密码。
本机测试账号及密码可在已生成的 `data/dev-seed` 私有资料中查看，禁止提交这些资料。

认证接口使用 `/api/auth` 前缀：

- `POST /login`：提交 `email`、`password`，返回用户、可用租户及 5 分钟的一次性
  `selection_token`。这个票据只能用于选租户，不能调用业务接口。
- `POST /select-tenant`：提交 `selection_token`、`tenant_id`，返回 Access Token、用户
  和当前租户，设置 Refresh Token Cookie。客户端不能选取未加入或已停用的租户。
- `POST /refresh`：浏览器携带 Refresh Cookie，返回新 Access Token，并轮换 Refresh
  Token。旧 Refresh Token 立即失效，新 Cookie 仍沿用原始会话到期时间。
- `GET /me`：使用 Bearer Token 查询当前用户和租户。
- `POST /logout`：使用当前 Access Token 或 Refresh Cookie，撤销这个账号在所有租户、
  所有设备的登录会话，并清除当前 Cookie。

Access Token 只保存在浏览器内存，请求通过 `Authorization: Bearer` 携带。
Refresh Token 放 `HttpOnly` Cookie，Redis 仅保存摘要、会话信息和到期时间；数据库
`auth_session` 保存会话及摘要审计记录，刷新校验以 Redis 为准。页面重新打开时通过
Refresh Cookie 恢复登录。认证写接口要求 `X-Requested-With: XMLHttpRequest`，并校验
浏览器 Origin。建议前端与 API 同站部署；生产环境必须使用 HTTPS，并启用
`REFRESH_COOKIE_SECURE=true`。

只有 `401 / ACCESS_TOKEN_EXPIRED` 才会触发前端共享的刷新请求，成功后原请求重试
一次。其他 401 回到登录页，403 保留权限错误。所有文档、任务、知识库接口都要求正式
登录，HTTP 不再提供开发身份绕过。每次请求读取数据库会话及账号、租户、成员和部门
状态，校验授权版本，因此全设备退出后已有 Access Token 立即失效。知识库显式
READ/WRITE/ADMIN 授权、文档部门范围及密级限制由业务模块实时校验。健康检查仍无需登录。

登录前需配置至少 32 字节的随机 `JWT_SECRET_KEY`，并保证 PostgreSQL、Redis 可用；
凭据与连接错误不会返回到客户端。Redis 丢失的 Refresh 会话要求重新登录，数据库
或 Redis 故障返回脱敏的 503。其他配置见 `.env.example` 与 [前端说明](frontend/README.md)。

认证集成测试使用真实 PostgreSQL 和 Redis，在唯一临时 schema 与 Redis 前缀中
创建测试账号和会话，完成后清理；不改项目已有账号和会话，不调用模型。显式运行：

```powershell
$env:RUN_AUTH_INTEGRATION = '1'
.\.venv\Scripts\pytest.exe -q tests\test_auth_integration.py
Remove-Item Env:\RUN_AUTH_INTEGRATION
```

存储不可用或数据库失败返回脱敏的 503，不返回存储凭据或数据库连接信息。
对象写入成功但数据库事务失败或提交结果未知时不删除对象，以免损坏已提交修订；
孤儿对象需后续按项目对象前缀和已提交修订核对后清理。当前尚未实现孤儿清理任务。

```powershell
.\.venv\Scripts\pytest.exe -q tests\test_document_upload_api.py
# 显式启用真实 PostgreSQL/MinIO 实测，测试会清理自己创建的数据。
$env:RUN_UPLOAD_INTEGRATION = '1'
.\.venv\Scripts\pytest.exe -q tests\test_upload_integration.py
Remove-Item Env:\RUN_UPLOAD_INTEGRATION
# 持久化队列、租约恢复、重试、原子发布实测（自动清理测试数据）。
$env:RUN_INGESTION_INTEGRATION = '1'
.\.venv\Scripts\pytest.exe -q tests\test_ingestion_integration.py
# 同时启用此项可用小文档实际调用向量模型验证完整闭环。
$env:RUN_INGESTION_MODEL_INTEGRATION = '1'
.\.venv\Scripts\pytest.exe -q tests\test_ingestion_integration.py -k real_embedding_model
Remove-Item Env:\RUN_INGESTION_MODEL_INTEGRATION
Remove-Item Env:\RUN_INGESTION_INTEGRATION
```

### 中间件与模型配置

本机 `.env` 已迁移旧项目的中间件及模型配置，并生成新的 JWT 密钥。
新机器先将 `.env.example` 复制为 `.env`，填写实际凭据并生成独立的 JWT 密钥。
系统环境变量优先于 `.env`。本地缓存保存在 `.uv-cache`。

```powershell
uv run python -m zhiqing_rag.checks
uv run ruff check .
uv run ruff format --check .
```

环境检查仅验证中间件连接、pgvector 扩展和存储桶存在性；不会建表、建桶或写入队列。
当前连接独立数据库 `zhiqing_rag_platform`，项目 schema 为 `zhiqing_rag`。环境检查中的 `project_schema: present` 只表示 schema 存在，不代表业务表已初始化。
模型项显示凭据是否配置，不代表已完成模型 API 调用验证。
查询分析器已实现 `POST /api/query/analyze`，需要登录，接收当前问题并返回带原文依据的
意图、实体、关键词、条件和上下文依赖。DeepSeek 未配置或分析失败时返回显式降级建议；
不执行检索或回答生成。配置与接口详见 [查询分析器](docs/查询分析器.md)。
QueryPlan 与确定性 Router 已实现 `POST /api/query/plan`，先确认当前用户可读资源范围，
再调用 Analyzer 生成五类类型化计划；图谱未接入时回到基础路径，目录执行器未接入时
明确返回能力限制。计划成功不表示已执行检索，详见 [查询路由](docs/查询路由.md)。
基础混合检索已实现 `POST /api/query/search`，按服务端计划并行召回向量和 BM25，
融合去重、可选重排后返回最多 5 条原文证据。发送原文给重排模型前及返回结果前复查
当前权限和活动批次；失败有明确降级记录。配置与接口详见 [混合检索](docs/混合检索.md)。

单轮回答已实现 `POST /api/query/answer`：复用现有检索，生成带引用的草稿，完成逐字引文与
独立证据支持复核后发布。生成前、复核前和发布前实时检查权限与活动批次，资料不足时
明确返回不足。页面默认“生成回答”，也可选择“仅检索证据”。配置与限制见
[单轮回答说明](QUERY_ANSWERING.md)，固定问题验收见 [检索验收](eval/query_retrieval/README.md)。

历史 XLSX 表头索引可用 [安全重建工具](TABLE_REINDEX.md) 预览并重建。显式并列问题使用同一授权候选池分面重排，保持原召回预算；验收见 [检索与回答记录](eval/query_retrieval/ACCEPTANCE.md)。

HTTP 鉴权、范围确认和检索仓储已接入进程内数据库连接池。沿用 `DB_POOL_SIZE`、
`DB_MAX_OVERFLOW`、`DB_POOL_TIMEOUT`；每次事务结束归还连接，模型等待期间不占用
检索连接，权限复核仍实时查询。启动不预先连接数据库，应用关闭时释放空闲连接。
只读对比短连接与连接池（不读文档、不调用模型、不写业务数据）：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_database_pool.py --samples 5
```

脚本分别输出首个冷连接、短连接与热池的耗时，以及事务恢复、断连恢复、并发独占
检查。数值只反映数据库最小操作，不代表完整检索或模型耗时；前端复测应在后端
加载新代码后连续查询同一个问题，对比第二、三次的检索执行和授权校验耗时。

同一轮的会话、用户、租户成员、租户及部门状态已合并为一次 SELECT；各次复核仍
独立实时读取，重排前后及 BM25 评分后复核均保留。只读核对原查询与新查询结果和耗时：

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_authorization_queries.py --samples 5
```

此脚本只读取一条当前会话的认证状态，不输出会话标识、用户信息或凭据；若无当前
会话则跳过。它不调用模型、不读取文档原文、不写业务库。

## 技术组成

| 用途 | 组件 |
| --- | --- |
| HTTP 服务 / 流式输出 | FastAPI、Uvicorn、SSE |
| 配置 | Pydantic Settings、dotenv |
| 关系与向量数据 | PostgreSQL、pgvector、SQLAlchemy 2、psycopg 3 |
| 关键词检索 | PostgreSQL pg_search（BM25，已确认扩展存在） |
| 数据迁移 | Alembic（已安装，待表设计后配置迁移） |
| 缓存与异步任务 | Redis、ARQ |
| 文件存储 | MinIO |
| 图谱存储 | Neo4j |
| 模型接口 | OpenAI SDK、LangChain、DashScope；预留 DeepSeek |
| 文档解析 | PDF、DOCX、XLS/XLSX、Markdown、TXT（格式校验和结构化解析已实现） |
| 开发工具 | pytest、pytest-asyncio、Ruff |

使用 `src/zhiqing_rag` 布局，uv 以 editable 模式安装本项目。

### 离线文档格式校验

解析前可调用 `zhiqing_rag.document_processing.format_validation.validate_document_format`。
目前支持 PDF、DOCX、XLS/XLSX、Markdown（`.md`、`.markdown`）和 TXT；旧版 `.doc` 不支持。
校验包括文件大小、扩展名、可选的声明 MIME、实际文件结构和基本完整性，
通过后返回统一格式、MIME、大小、SHA-256 和文本编码。默认文件大小上限为 50 MiB，
调用方可通过 `max_file_size` 调整；临时文件没有原始扩展名时传入 `original_filename`。

```python
from zhiqing_rag.document_processing.format_validation import validate_document_format

result = validate_document_format("/path/to/upload.tmp", original_filename="manual.pdf")
print(result.format, result.file_sha256)
```

校验失败会抛出带 `code` 的 `DocumentFormatError`。这一步不提取正文，也不判断 PDF
是否为扫描件；后续解析阶段负责处理这些情况。

### 离线文档解析与分块

`process_document` 依次执行格式校验、解析、清洗和分块，返回原始结构化元素、
清洗后的元素及可入库的分块草稿。PDF 先按页尝试直接提取文字：有可靠文字层且
没有图形、图片或明显对齐分栏的页面使用轻量提取；扫描页、表格等复杂页面交给
本地 MinerU Basic 的 `auto` 模式。混合 PDF 只把需要处理的页面交给 MinerU，
结果仍保留原始页码及来源位置。其他格式沿用各自的解析策略。

```python
from zhiqing_rag.document_processing import ChunkConfig, process_document

result = process_document(
    "/path/to/upload.tmp",
    original_filename="manual.pdf",
    chunk_config=ChunkConfig(max_tokens=512, overlap_tokens=48, min_chunk_tokens=96),
)
for chunk in result.chunks:
    print(chunk.chunk_index, chunk.page_number, chunk.content, chunk.source_refs)
print(result.cleaned.warnings, result.cleaned.is_complete)
```

清洗阶段去除页眉、页脚和页码，保留表格及代码格式。分块阶段先按章节和页面组织正文，
长文本再按中英文段落、句子等边界递归切分；同页同章节或同一父章节下相邻的小正文块
在预算允许时合并。表格优先保持整体，超限时按行切分并重复表头；单行或复杂表格仍超限时
继续拆为片段，元数据标记 `table_fragment`。每块带有来源引用、标题路径、token 数和
正文 SHA-256，最终的 `embedding_content` 不超过 `max_tokens`。

可以用任意受支持的本地文件运行完整流程测试（PDF、DOCX、XLS、XLSX、MD、TXT）：

```powershell
$env:DOCUMENT_TEST_FILE = 'D:\documents\example.md'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\pytest.exe -q -s tests\test_user_document.py
Remove-Item Env:\DOCUMENT_TEST_FILE
Remove-Item Env:\PYTHONIOENCODING
```

测试会检查解析、清洗和分块结果，并打印格式、元素数量、告警及前 20 个分块预览。
PDF 的复杂版面或扫描页会按需调用本地 MinerU。

PDF 中图片没有识别出文字时产生 `OCR_NO_TEXT` 告警，影响 `is_complete`；
XLSX 公式缺少缓存值时产生 `FORMULA_CACHE_MISSING` 告警。Excel 按连续非空行形成
表格区域；单个区域超过 500 行时会拆成多个带原始单元格范围的解析元素。

### 分块向量化与入库

`await embed_chunks(chunks)` 使用配置中的百炼 OpenAI 兼容接口，每批最多 10 个
`embedding_content`，明确请求 1024 维浮点向量。返回结果按响应 `index` 对应原分块，
逐个检查维度、有限数值、float32 范围和非零向量；全部批次成功才返回
`tuple[EmbeddedChunk, ...]`。20 个未命中的不同分块会自动拆成两批。

`EmbeddedChunk` 保存原始分块 `chunk`、不可变向量 `embedding` 和 `model_name`。
客户端默认超时 30 秒，使用 SDK 的最多两次重试；不叠加业务层重试。

向量缓存默认启用，保存到已配置的 Redis，TTL 为 7 天。缓存键使用
`zhiqing-rag:embedding:v2:float32:<模型配置哈希>:<embedding_content SHA-256>`，
模型配置包含供应商、接口地址、模型名、维度、缓存修订及输出格式，不包含凭据。
缓存值只保存小端序 float32 二进制向量，1024 维固定为 4096 字节，
不保存文档正文、文档 ID 或分块来源。缓存命中返回的数值采用 float32 精度，
与模型原始 Python float 可能有微小舍入差异。旧版 JSON 缓存通过键版本隔离，按原 TTL 自动过期。
命中向量仍执行完整校验；仅对未命中的不同文本调用模型，随后按原始顺序恢复全部
`EmbeddedChunk`。标题上下文变化会产生不同缓存键。同一次调用内相同输入只计算一次。
Redis 读取或写入失败不会阻断向量生成及后续数据库入库，非法缓存值按未命中处理。

通过以下配置调整缓存；模型快照或影响向量的请求参数变化时，应更新缓存修订：

```dotenv
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_TTL_SECONDS=604800
EMBEDDING_CACHE_REVISION=local-config-v1
EMBEDDING_CACHE_TIMEOUT_SECONDS=1
```

可传入 `EmbeddingStats` 查看 `cache_hits`、`model_chunks` 和 `model_batches`；
缓存命中的分块仍按原流程入库。导入报告中的 `batch_count` 现在记录实际提交的逻辑模型批次数。

`save_embedded_chunks(session, scope, embedded)` 在调用方事务内保存完整批次，
核对租户、知识库、文档、修订、模型配置及 `BUILDING` 状态，保留正文、实际向量输入、
页码、标题、哈希及来源引用。全部写入后将批次置为 `READY`；不自行提交事务。
已有分块或已完成的批次会被拒绝，不覆盖原数据。文档发布指针由后续发布流程维护。

运行真实 20 块示例（知识库 1、成员 1 为当前默认租户的调试数据）：

```powershell
.\.venv\Scripts\python.exe scripts\embed_document.py --demo-20 --kb-id 1 --member-id 1
```

该命令生成 20 个章节的 Markdown 测试文档，真实调用 `text-embedding-v3`，
将原文件上传 MinIO，并在同一数据库事务内新建文档、修订、索引批次和全部分块。
提交后用新会话核对分块与向量，结果报告位于 `data/embedding-inputs/<run_id>/result.json`。
批次为 `READY`，文档为 `DRAFT`。每次执行新建一份文档，并产生模型调用费用；
数据库写入失败时事务回滚，已上传的原文件保留在此次独立对象键下，便于排查。

也可导入受支持的真实文件，分块数由现有解析和分块规则决定：

```powershell
.\.venv\Scripts\python.exe scripts\embed_document.py --file 'D:\documents\example.md' --kb-id 1 --member-id 1
```

此入口是独立本地开发导入工具，不复用 HTTP 上传任务或知识库权限鉴权；页面上传
使用上文的独立后台 Worker，二者共享解析、向量生成和完整批次入库函数。
可单独验证同一文件连续两次向量化：第一次填充未命中缓存，第二次应全部命中；
该命令不创建数据库文档，首次未命中会产生模型调用费用：

```powershell
.\.venv\Scripts\python.exe scripts\check_embedding_cache.py 'D:\documents\example.md'
```

普通测试不会调用模型或写入数据库；入库集成测试需显式启用，测试数据自动回滚：

```powershell
$env:RUN_EMBEDDING_DB_INTEGRATION = '1'
$testTemp = Join-Path '.uv-cache' ('pytest-embedding-db-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\pytest.exe -q tests\test_chunk_storage_integration.py --basetemp $testTemp
Remove-Item Env:\RUN_EMBEDDING_DB_INTEGRATION
```

### 本地 MinerU Basic 环境

MinerU 4.0.7 要求 `openai<3`，主项目要求 `openai>=3.6`，因此 MinerU 安装在
独立的 `.venv-mineru`。模型位于项目内 `.mineru/models`，两者已加入 Git 忽略。
首次在 Windows PowerShell 准备环境：

```powershell
uv venv --python 3.12 .venv-mineru
uv pip install --python .venv-mineru\Scripts\python.exe "mineru==4.0.7"
$env:MINERU_HOME = (Join-Path (Get-Location) '.mineru')
.\.venv-mineru\Scripts\mineru-kit.exe models download --tier basic --small-backend onnx --source modelscope
.\.venv-mineru\Scripts\mineru-kit.exe models verify --tier basic --small-backend onnx
```

运行时默认查找项目 `.venv-mineru` 和 `.mineru`。从其他目录启动或在服务器部署时，
设置 `MINERU_PYTHON` 为独立环境的 Python 绝对路径、`MINERU_HOME` 为模型目录的
上一级目录；可用 `MINERU_TIMEOUT_SECONDS` 调整单份 PDF 的解析超时。解析进程
始终使用本地模型，不自动联网下载。Linux 下独立环境的 Python 路径为
`.venv-mineru/bin/python`。

本地模型的端到端转换测试会生成一页包含标题和表格的 PDF，验证复杂页面的
MinerU 输出已转换成统一元素和分块。简单文字 PDF 不需要 MinerU 环境。
普通测试默认跳过模型实测；需要实测时运行：

```powershell
$env:RUN_MINERU_INTEGRATION = '1'
$testTemp = Join-Path '.uv-cache' ('pytest-mineru-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\pytest.exe -q tests\test_mineru_integration.py --basetemp $testTemp
Remove-Item Env:\RUN_MINERU_INTEGRATION
```

也可以指定自己的 PDF；`-s` 会打印前 10 个转换后的元素，便于核对页码、类型和文字：

```powershell
$env:RUN_MINERU_INTEGRATION = '1'
$env:MINERU_TEST_PDF = 'D:\documents\example.pdf'
$testTemp = Join-Path '.uv-cache' ('pytest-mineru-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\pytest.exe -q -s tests\test_mineru_integration.py -k user_pdf --basetemp $testTemp
Remove-Item Env:\MINERU_TEST_PDF
Remove-Item Env:\RUN_MINERU_INTEGRATION
```

在 PowerShell 中设置路径时，请使用键盘输入的普通单引号或双引号。复制路径时如果带入
弯引号（如 `“`、`”`），测试入口也会去掉成对的外层引号。

依赖沿用旧项目主技术栈，通过 `uv.lock` 固定实际版本；评估工具将在评估方案确定后单独配置。
新依赖使用 `uv add 包名`，开发工具使用 `uv add --dev 包名`。
修改依赖后提交 `pyproject.toml` 和 `uv.lock`；不要提交 `.env`、`.venv` 和缓存目录。

## 后续数据隔离约定

现阶段沿用已有服务实例与连接凭据。PostgreSQL 已新建独立数据库 `zhiqing_rag_platform`，其中 schema 为 `zhiqing_rag`；
Redis/ARQ 使用 `zhiqing-rag:` 前缀；MinIO 对象使用 `zhiqing-rag-platform/` 前缀；
Neo4j 图谱投影使用 `GRAPH_NAMESPACE` 隔离项目；图谱任务复用现有 PostgreSQL 任务表。
数据库、schema 与业务表已创建；身份与知识库调试数据已录入，文档可通过页面上传或本地开发工具导入。

## 部门知识权限

本次登录、租户选择、令牌刷新和部门文档授权的完整整理见
[登录与知识权限设计及实现说明](D:/code/zhiqing-rag-platform/docs/登录与知识权限设计及实现说明.md)。

授权顺序为当前登录身份 → 租户 → 知识库 ACL → 文档密级与部门/显式共享。
知识库对用户和当前部门的授权取并集，等级为 READ < WRITE < ADMIN；部门树不自动继承，
租户管理员也不能绕过知识库和文档权限。共享文档不会自动授予知识库权限或提高成员密级。

新上传文档默认 `confidentiality=1`，所属部门和创建者由服务端填写。可选择的密级不能
高于成员 `clearance`：0 内部公开、1 部门内部、2 机密。内部公开仍要求同租户和知识库 ACL；
1/2 额外要求当前部门与文档部门相同，或对用户/当前部门显式共享。无部门归属的非公开文档
默认仅创建者可见，可另外共享。有部门归属的文档在上传者调离部门后不再默认对其开放。
历史文档保留原密级，新规则会收紧其可见范围；更新文件不会修改文档密级或所属部门。

列表、统计、任务状态、修订历史、预览和原文件下载使用 READ 校验；新增、更新、删除、
重试使用 WRITE 校验。跨部门 READ 共享只允许读取，即使知识库有 WRITE 权限也不能修改；
还需要文档 WRITE 共享才可写入。权限管理要求知识库 ADMIN，并同时通过文档 WRITE 范围及密级校验。

页面提供知识库“管理授权”和文档“权限”入口，支持用户/部门授权、撤销和文档密级调整。
权限写入与 `knowledge_base.acl_version` 递增、审计记录在同一事务提交。撤销后下一次请求
立即按新规则拒绝，前端同时清理列表/历史缓存、关闭旧预览并刷新同来源标签页。
服务端不会将 ACL 长期写进 JWT；已有下载到本地的文件无法收回。
后台解析、向量发布及图谱任务同样复查当前修订上传者的 WRITE 权限；调岗或撤销写权限后，后续处理与发布
会被拒绝。权限事务和后台发布统一先锁知识库，再锁任务、文档和投影，避免反向等待。

- `GET/POST /api/knowledge-bases/{kb_id}/permissions` 查看/添加或调整授权；
  `DELETE .../permissions/{grant_id}` 撤销授权，必须保留至少一个可用 ADMIN。
- `GET/POST /api/documents/{document_id}/permissions` 查看/添加或调整 READ/WRITE 共享；
  `DELETE .../permissions/{grant_id}` 撤销共享。
- `POST /api/documents/{document_id}/classification` 调整文档密级，提交 `{ "confidentiality": 1 }`。

已有数据库只需执行幂等增量迁移，不回填授权或改动现有文档：

```powershell
.\.venv\Scripts\python.exe scripts\migrate_document_permissions.py --dry-run
.\.venv\Scripts\python.exe scripts\migrate_document_permissions.py
```

基础向量/关键词检索已复用 `authorized_chunk_statement(identity)` 所依赖的 READ 授权规则，
在 SQL 中限制可见文档和当前 READY 批次，再执行检索排序；单轮回答与引用复核已接入，详见 `QUERY_ANSWERING.md`。
后续图检索使用 `authorized_graph_projection`，并限制其租户、namespace、generation。
结果缓存需要包含当前身份/部门/密级和 ACL 版本；模型上下文、引用与会话历史也需重新检查权限。
内部 `active_graph_projection` 仅查询发布状态，保留给可信后台任务使用，不可作为用户请求的授权入口。

权限集成测试使用独立临时 schema 和 Redis 前缀，自动清理，不访问业务文件或调用模型：

```powershell
$env:RUN_PERMISSION_INTEGRATION = '1'
.\.venv\Scripts\python.exe -m pytest tests\test_permissions_integration.py -q -p no:cacheprovider
Remove-Item Env:\RUN_PERMISSION_INTEGRATION
```
