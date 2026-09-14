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
psql -d ragkb -f .\zq_rag_app\schemas\migrations\002_document_versions.sql
psql -d ragkb -f .\zq_rag_app\schemas\migrations\003_document_history.sql
```

## 异步向量索引

索引流水线包含 SHA256 内容寻址、进程内 LRU、Redis float32 二进制缓存、
分布式锁、批量 Embedding、网络指数退避、PostgreSQL Upsert、任务心跳和租约。

启动 API：

```powershell
uv run python -m zq_rag_app.main
```

Windows 下建议使用上述入口，它会在 Uvicorn 启动前切换到 psycopg 异步连接所需的 Selector 事件循环。

每个部署节点启动一个或多个 ARQ Worker：

```powershell
uv run python -m arq zq_rag_app.workers.index_worker.WorkerSettings
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
递增 `EMBEDDING_CACHE_VERSION` 即可让旧缓存自然失效。当前使用的 DashScope
`text-embedding-v3` 单批最多 10 条，`EMBEDDING_BATCH_SIZE` 不要配置为更大的值。
Embedding 缓存连接异常时会主动清理旧连接并降级直调模型；ARQ Worker 的 Redis
连接启用了健康检查与有限重试，以覆盖任务完成状态写回时的瞬时断线。

## 前端召回实验台

启动 API 后访问 `http://localhost:8000/`，可使用文档异步处理与余弦向量召回测试页面。页面默认连接真实的上传、文档列表和召回接口；使用 `http://localhost:8000/?demo=1` 可查看无需基础设施的演示数据。

前后端接口约定、异步阶段和余弦查询 SQL 见 [`docs/frontend-integration.md`](docs/frontend-integration.md)。

## 检索离线评估

评估语料默认读取 `qa-docs/manifest.csv`，Golden Dataset 默认读取
`qa-docs/qa_ground_truth.jsonl`。先验证目标文档及当前版本 Chunk：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_dense_eval.py --kb-id 4 --stats-only
```

再以 manifest 中的固定文档集合运行 Dense 余弦检索基线：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_dense_eval.py --kb-id 4 --top-k 20 --min-score 0
```

若 PostgreSQL 已安装 `pg_search` 并创建 `idx_chunk_bm25` 索引，可用同一套
盲测输入和 Gold 标签运行 ParadeDB BM25 基线：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_bm25_eval.py --kb-id 4 --top-k 20
```

运行 Dense + BM25 分数归一化加权并叠加 RRF 的混合检索：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_hybrid_rrf_eval.py --kb-id 4 --top-k 20 --candidate-k 50 --dense-weight 0.5 --bm25-weight 0.5 --rrf-weight 0.5 --rrf-k 60
```

混合检索先对两路候选分数分别做 Min-Max 归一化，再按 Dense/BM25 权重
合成分数；该分数与归一化加权 RRF 分数按 `rrf-weight` 组合。逐题明细会
保留两路原始分数、原始排名和融合过程分数。

在融合结果后增加 `gte-rerank-v2` 精排（融合 Top-30 精排为 Top-20）：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_hybrid_reranker_eval.py --kb-id 4 --top-k 20 --candidate-k 50 --rerank-candidate-k 30 --dense-weight 0.5 --bm25-weight 0.5 --rrf-weight 0.5 --rrf-k 60 --reranker-timeout-ms 10000
```

精排请求只包含问题与召回 Chunk 正文。评估脚本使用 `RERANKER_ENDPOINT`、
`RERANKER_MODEL` 和现有 `DASHSCOPE_API_KEY`，不会把 Gold 标签发送给模型。

脚本会批量生成 Query Embedding，并输出 `summary.json`、
`per-question.jsonl`、`blind-retrieval-input.jsonl`、
`gold-chunk-map.jsonl` 和 `report.md`。检索阶段只使用 `question_id` 和
`question`，完成全部召回后才读取 Gold 字段评分：文档级指标按
`source_files` 计算，Chunk 级指标按来源文件中的关键词/位置证据 Chunk 组
计算。报告记录 manifest 与 Golden Dataset 的 SHA256，便于后续方案使用
完全相同的数据集。`no_answer` 与 `permission` 问题不计入召回指标，需要在
回答层和 ACL 层单独评估。
