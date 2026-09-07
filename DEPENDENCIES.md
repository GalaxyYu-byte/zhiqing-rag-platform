# ZQ RAG 项目依赖说明

本文档根据 [RAG 企业知识库系统整体架构](./RAG企业知识库系统整体架构_v3.md) 和当前 [pyproject.toml](./pyproject.toml) 整理。

## 一、当前运行时依赖

| 依赖 | 作用 | 对应架构能力 |
|---|---|---|
| `fastapi` | Web API 框架，提供接口、依赖注入和请求处理 | Controller、Health API |
| `httpx` | 异步 HTTP 客户端 | 调用 LLM、Embedding、Reranker 服务 |
| `langchain` | RAG、Prompt、Runnable 和流程编排基础能力 | RAG Pipeline、Query Router |
| `langchain-openai` | LangChain 对接 OpenAI 兼容接口 | 通义千问等 LLM 服务 |
| `markdown-it-py` | Markdown 文档解析 | Markdown 文档处理 |
| `minio` | MinIO 客户端 | 原始文档和附件对象存储 |
| `openai` | 调用聊天模型和 Embedding 模型 | LLM、Embedding、Query Rewrite、HyDE |
| `pgvector` | SQLAlchemy 映射 PostgreSQL `vector` 类型 | PGVector、HNSW 向量检索 |
| `prometheus-client` | 暴露 Prometheus 指标 | 服务监控和 RAG 指标监控 |
| `psycopg[binary]` | PostgreSQL 驱动 | SQLAlchemy 异步数据库连接 |
| `pydantic-settings` | 从 `.env` 和环境变量加载配置 | 服务配置管理 |
| `pyjwt[crypto]` | JWT 创建、解析和签名验证 | JWT 认证、UserContext |
| `pypdf` | PDF 文本解析 | PDF 文档索引 |
| `python-docx` | Word `.docx` 文档解析 | Word 文档索引 |
| `python-multipart` | 文件上传和表单数据解析 | 文档上传接口 |
| `redis` | Redis 同步/异步客户端 | Embedding 缓存、查询缓存、任务进度 |
| `sqlalchemy` | ORM、数据库模型和异步数据库访问 | Repository、业务表、Chunk 表 |
| `tenacity` | 失败重试和退避 | LLM、Embedding、索引任务重试 |
| `tiktoken` | Token 数量估算 | Context Budget、历史消息裁剪 |
| `uvicorn[standard]` | ASGI 服务启动器 | FastAPI 服务运行 |

## 二、开发和测试依赖

| 依赖 | 作用 |
|---|---|
| `pytest` | 单元测试和集成测试框架 |
| `pytest-asyncio` | 测试异步函数和异步数据库逻辑 |

## 三、建议新增依赖

### 3.1 建议立即添加

```powershell
uv add neo4j openpyxl pydantic
uv add --group eval ragas
```

| 依赖 | 是否必须 | 作用 |
|---|---:|---|
| `neo4j` | 是 | 连接 Neo4j，写入实体关系并执行 GraphRAG 多跳查询 |
| `openpyxl` | 是 | 解析 `.xlsx` 和 `.xlsm` 文件；架构文档明确支持 Excel |
| `pydantic` | 建议 | 项目代码直接使用 `pydantic.computed_field`，应显式声明直接依赖 |
| `ragas` | 评测时必须 | 计算 Context Precision、Context Recall、Faithfulness、Answer Relevancy 等指标 |

如果需要支持旧版 `.xls` 文件：

```powershell
uv add xlrd
```

### 3.2 按实现方式选择

```powershell
uv add sse-starlette alembic
```

| 依赖 | 是否必须 | 作用 |
|---|---:|---|
| `sse-starlette` | 可选 | 提供更完整的 SSE `EventSourceResponse`；简单 SSE 也可以使用 FastAPI 的 `StreamingResponse` |
| `alembic` | 强烈建议 | 管理 PostgreSQL 表结构和索引迁移，替代手工维护全部数据库变更 |

如果要使用 Redis 任务队列替代当前进程内线程池，二选一：

```powershell
uv add arq
```

或：

```powershell
uv add "celery[redis]"
```

`arq` 更适合当前的 asyncio + Redis 技术栈；`celery` 生态更完整，但相对更重。

如果要在本地运行 Cross-Encoder，而不是调用远程 Reranker API：

```powershell
uv add sentence-transformers
```

该方案通常还需要安装适配环境的 `torch`。当前项目配置了远程 `reranker_endpoint`，已有 `httpx`，因此暂时不需要本地模型依赖。

## 四、不需要重复添加的能力

以下能力已经由现有依赖或 Python 标准库覆盖：

- PDF：`pypdf`
- Word：`python-docx`
- Markdown：`markdown-it-py`
- TXT：Python 标准库即可
- Embedding 和 LLM：`openai`、`langchain-openai`
- PostgreSQL + PGVector：`psycopg[binary]`、`sqlalchemy`、`pgvector`
- Redis：`redis`
- MinIO：`minio`
- JWT：`pyjwt[crypto]`
- RRF、MRR、Recall@K、HitRate@K：可以使用少量自研 Python 代码，不需要单独依赖

## 五、BM25 和 `pg_search`

`pg_search` 不是 Python 包，不能使用 `uv add pg_search` 安装。它是 PostgreSQL 服务端扩展，需要安装到 PostgreSQL 所在的服务器或数据库容器中。

当前项目使用的是：

```text
TSVECTOR + GIN + ts_rank
```

这属于 PostgreSQL 原生全文检索，并不等同于严格的 BM25。若需要严格 BM25，需要在数据库层选择以下方案之一：

1. 安装 ParadeDB 的 `pg_search` 扩展；
2. 使用 Elasticsearch/OpenSearch；
3. 小规模数据使用进程内 BM25 实现，但不建议用于生产环境。

启用扩展后，需要在目标数据库执行：

```sql
CREATE EXTENSION IF NOT EXISTS pg_search;
```

仅安装扩展不会自动把现有 GIN 索引转换成 BM25 索引，还需要修改索引定义和查询 SQL。

## 六、`uv` 项目打包配置

当前源码使用 flat layout：

```text
zq_rag_app/
├── __init__.py
└── main.py
```

但 `uv_build` 默认寻找：

```text
src/zq_rag_py/__init__.py
```

因此需要在 `pyproject.toml` 中增加：

```toml
[tool.uv.build-backend]
module-name = "zq_rag_app"
module-root = ""
```

同时删除当前指向不存在模块的 `[project.scripts]` 配置，或者改为实际存在的可调用函数。

项目启动命令：

```powershell
uv run uvicorn zq_rag_app.main:app --reload
```
