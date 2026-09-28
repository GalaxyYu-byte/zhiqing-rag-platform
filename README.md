# 知擎 RAG 平台

新项目位于 `D:\code\zhiqing-rag-platform`，Python 固定为 **3.12**，使用 uv 管理环境和依赖。
当前已完成开发环境、健康检查入口、文档上传接口、Vue 3 上传页面、独立数据库和 27 张业务表的设计与初始化；身份与知识库调试数据已录入。其他业务 API 与 Alembic 将后续完善。

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
页面从 `GET /api/knowledge-bases` 读取当前身份可写入的知识库，向
`POST /api/documents/upload` 提交 `multipart/form-data`：`file`、实际数字
`knowledge_base_id`。解析参数由服务端 `UPLOAD_CHUNK_SIZE`（默认 512）和
`UPLOAD_CHUNK_OVERLAP`（默认 48）统一管理，浏览器不能覆盖。

上传校验实际文件内容、声明 MIME、完整性、文件名和大小；默认上限 50 MiB，
请求体在 multipart 解析前限制为文件上限加 1 MiB。文件存入现有 MinIO 桶的
项目/租户/知识库范围内独立对象键，不覆盖同名文件。数据库事务创建 DRAFT 文档、
第一份不可变修订、期望修订指针、索引批次和持久化后台任务，并保存分块参数。
返回 **202 Accepted**，包含字符串形式的文档/修订/索引批次/任务 ID、SHA-256、MIME、
大小、创建时间以及 `STORED / PENDING` 状态。请求不执行解析或调用向量模型。

再开一个终端启动独立后台进程：

```powershell
cd D:\code\zhiqing-rag-platform
uv run python -m zhiqing_rag.workers.ingestion
```

Worker 从项目独立 PostgreSQL 的 `background_task` 表领取任务，下载并校验 MinIO
原文件，随后解析、清洗、分块，调用现有 Embedding 模型生成 1024 维向量，并将分块
正文、向量和来源引用写入 `document_chunk`。仅在完整性校验成功后，同一事务将
索引批次置为 READY、文档置为 PUBLISHED 并切换活动索引指针，任务置为 SUCCEEDED。
这表示文档已可供后续检索模块使用，问答检索 API 本身仍需后续实现。

当前采用 **PostgreSQL 持久化任务队列 + 独立 Worker**，任务与上传记录同事务提交，
无需跨 Redis 写入和 Outbox 投递。该 Worker 不使用 ARQ；Redis 仅沿用现有向量缓存。
这是为了在当前 `REDIS_DB=0` 与旧项目共用的情况下避免启动共享 ARQ Worker。
将来切换到 ARQ 需分配独立逻辑库并实现 Outbox 投递，不能同时无租约地执行同一任务。

默认并发 2、每 2 秒检查队列、租约 90 秒、每 30 秒续约、单任务预算 1200 秒，
暂时性失败最多尝试 3 次，并按 15/30 秒退避。Worker 退出或宕机后，其他进程可在
租约过期后恢复任务；模型结果通过既有向量缓存复用。确定性解析/权限/模型配置错误
直接标记失败。旧租约和非期望修订不允许发布。解析在可终止子进程中执行，
避免 OCR 或 CPU 解析阻塞异步事件循环。失败后重试处理会创建新索引批次，保留
不可变失败批次和历史尝试记录，不覆盖已完成批次。当前没有图谱任务的自动投递。

接口：页面通过 `GET /api/document-tasks/page` 分页查询有访问权限的任务，支持
`offset`、`limit`（默认 10、最多 100）、`knowledge_base_id`、`search`、`status`、
`date_from` 和 `date_to`。日期按北京时间计算，包含结束日期当天；响应返回
`items`、`total`、`overall_total`、`overall_size` 和状态数量 `counts`。
`GET /api/document-tasks` 保留最近 50 个任务的兼容接口；
`GET /api/document-tasks/{task_id}` 返回阶段、进度、重试次数、错误码、分块数和
文档发布状态；`POST /api/document-tasks/{task_id}/retry` 重试失败任务。
前端每 2.5 秒查询活跃任务，展示等待、解析分块、生成向量、入库、完成与失败，
页面关闭不停止后台任务。移除未上传文件仅清理本地列表；移除已上传文档需确认后
永久删除服务器文档并停止后台处理。`GET /api/document-upload-settings` 提供只读解析配置。

正式登录 API 尚未实现，本机联调使用服务端配置的 `admin@zhiqing.test` / `DEFAULT`
开发身份；不会接收浏览器指定的用户或租户。必须为 `APP_ENV=dev`、回环地址绑定和
本机请求，同时检查用户、租户、成员、部门启用状态以及知识库 WRITE/ADMIN 显式授权。
可以使用 `DEV_UPLOAD_USER_EMAIL`、`DEV_UPLOAD_TENANT_CODE` 切换已有测试身份。
关闭 `DEV_UPLOAD_IDENTITY_ENABLED` 或在非 dev 环境调用时返回 401；正式上线前需
替换身份依赖为完整登录鉴权。其他配置见 `.env.example` 与 [前端说明](frontend/README.md)。

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
DeepSeek 为后续查询分析预留，可暂不配置。

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
Neo4j 预留 `GRAPH_NAMESPACE`。这些配置需要后续存储层、任务层和迁移代码落实。
数据库、schema 与 27 张业务表已创建；身份与知识库调试数据已录入，文档类数据待后续离线导入。
