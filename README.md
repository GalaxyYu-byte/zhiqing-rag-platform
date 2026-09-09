# zq-rag-py

基于 Python、FastAPI 和 PostgreSQL/pgvector 的知识库 RAG 服务。项目目前包含文档解析、文档清洗、结构感知递归分块，以及后续向量检索所需的基础模型和配置。

## 环境要求

- Python 3.12+
- PostgreSQL（需要启用 pgvector）
- Redis
- MinIO
- DashScope 兼容 OpenAI API 的 Embedding/LLM 服务

## 安装

项目使用 `uv` 管理依赖：

```powershell
uv sync
```

本地开发也可以直接使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 配置

在项目根目录创建 `.env`，至少配置以下敏感项：

```dotenv
DASHSCOPE_API_KEY=your-api-key
RERANKER_ENDPOINT=https://your-reranker-endpoint
JWT_SECRET_KEY=replace-with-a-random-secret
```

数据库、Redis、MinIO、模型和 RAG 参数也可以通过同名环境变量覆盖，具体字段见 [`zq_rag_app/core/config.py`](zq_rag_app/core/config.py)。不要把 `.env` 提交到版本库。

## 文档处理流程

当前文档处理流程为：

```text
文档解析 → 文档清洗 → 结构感知分段 → 语言感知递归分块 → Embedding/入库
```

解析器支持 TXT、Markdown、DOCX、PDF 和 Excel，并尽量保留页码、章节、标题路径和表格行结构。

分块模块位于 [`zq_rag_app/document_processing/chunking`](zq_rag_app/document_processing/chunking)，默认配置为 512 tokens、64 tokens Overlap：

```python
from zq_rag_app.document_processing.chunking import (
    ChunkingConfig,
    chunk_cleaned_document,
)
from zq_rag_app.document_processing.cleaning import clean_parsed_blocks
from zq_rag_app.utils.document_parser import parse_document

parsed = parse_document("tests/fixtures/documents/hr-handbook.txt")
cleaned = clean_parsed_blocks(parsed)
chunked = chunk_cleaned_document(
    cleaned,
    ChunkingConfig(chunk_size=512, chunk_overlap=64),
)

for chunk in chunked.chunks:
    print(chunk.content)
    print(chunk.embedding_content)
```

其中 `content` 保存正文，`embedding_content` 会额外带上章节路径，适合送入 Embedding 模型。

## 测试

运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.pytest-tmp-run
```

查看 HR 手册的解析、清洗和分块预览：

```powershell
.\.venv\Scripts\python.exe -m pytest .\tests\test_document_chunking.py::test_hr_handbook_parse_clean_and_chunk_preview -q -s --basetemp=.pytest-tmp-preview
```

测试数据位于 [`tests/fixtures/documents`](tests/fixtures/documents)。

## 数据库初始化

数据库表结构和示例评估数据位于 [`zq_rag_app/schemas`](zq_rag_app/schemas)。正式运行前请按部署环境执行对应的 Schema 初始化或迁移流程。

已有数据库需要先执行：

```powershell
psql -d ragkb -f .\zq_rag_app\schemas\migrations\001_index_pipeline.sql
```

## 异步向量索引

索引流水线包含 SHA256 内容寻址、进程内 LRU、Redis float32 二进制缓存、
分布式锁、批量 Embedding、网络指数退避、PostgreSQL Upsert、任务心跳和租约。

启动 API：

```powershell
uv run uvicorn zq_rag_app.main:app --host 0.0.0.0 --port 8000
```

每个部署节点启动一个或多个 ARQ Worker：

```powershell
uv run arq zq_rag_app.workers.index_worker.WorkerSettings
```

对已存在于 `kb_document` 且文件已上传 MinIO 的文档创建任务：

```http
POST /documents/{doc_id}/index
Content-Type: application/json

{"task_type":"INDEX"}
```

查询任务或文档最新索引状态：

```http
GET /documents/index-tasks/{task_id}
GET /documents/{doc_id}/index-status
```

可通过环境变量调整 `EMBEDDING_BATCH_SIZE`、`EMBEDDING_CONCURRENCY`、
`EMBEDDING_CACHE_TTL`、`EMBEDDING_LOCAL_CACHE_SIZE`、
`INDEX_UPSERT_BATCH_SIZE` 和 `INDEX_TASK_MAX_RETRY`。修改缓存键或二进制格式时，
递增 `EMBEDDING_CACHE_VERSION` 即可让旧缓存自然失效。
