# 前端工作台接入备注

前端静态资源位于 `zq_rag_app/static`，由 FastAPI 挂载到 `/`。页面默认调用真实后端接口；使用 `/?demo=1` 可在 PostgreSQL、Redis 或 MinIO 未启动时查看内置演示数据。

## 当前召回方案

第一阶段仅使用余弦相似度召回，不接入关键词召回与 Reranker：

```text
用户问题
  → 使用 text-embedding-v3 生成 1024 维 Query 向量
  → 按知识库 ID 过滤 kb_doc_chunk
  → pgvector HNSW + cosine distance 检索
  → 最低相似度过滤
  → 按 score 降序返回 Top K
```

PostgreSQL 查询可采用：

```sql
SELECT
    id AS chunk_id,
    doc_id,
    content,
    page_num,
    section_title,
    1 - (embedding <=> :query_embedding) AS score
FROM kb_doc_chunk
WHERE kb_id = ANY(:kb_ids)
  AND 1 - (embedding <=> :query_embedding) >= :min_score
ORDER BY embedding <=> :query_embedding
LIMIT :top_k;
```

备注：必须保证 Query 与文档 Chunk 使用相同的 Embedding 模型、维度和文本预处理方式。余弦分数适合在同一模型、同一语料范围内比较，不建议当成跨模型的绝对质量分。

## 已实现接口

### 0. 知识库创建与列表

```http
GET /knowledge-bases
POST /knowledge-bases
```

创建请求使用 JSON，字段包括必填的 `name`，以及可选的 `description`、`is_public`。归属部门和创建人默认从当前登录态取得。前端上传范围和召回范围均从列表接口动态加载。

### 1. 上传文档

```http
POST /documents/upload
Content-Type: multipart/form-data
```

字段：

- `files`: 一个或多个文件；
- `kb_id`: 目标知识库 ID；

服务端会先校验整批文件，再以 UUID 隔离路径流式写入 MinIO，创建 `kb_document` 与 `kb_index_task` 后返回 `202 Accepted`。解析、清洗、分块、Embedding 与入库不会阻塞上传请求。

`uploaded_by` 由服务端从当前登录态取得，前端不得传入。认证系统接入前，服务端会将每个请求绑定到固定管理员账号。

### 当前用户与权限

```http
GET /auth/me
GET /auth/knowledge-bases/{kb_id}/permission?permission=READ
GET /auth/knowledge-bases/{kb_id}/permission?permission=WRITE
```

权限接口返回 `allowed`。管理员直接放行；普通用户按创建者、公开只读范围、用户授权和部门授权判断。

分块参数当前沿用服务端 `RAG_CHUNK_SIZE` 和 `RAG_CHUNK_OVERLAP` 配置，不接受单次上传覆盖，避免同一任务在重试时读取到不同参数。

### 2. 文档列表

```http
GET /documents?kb_id=1&status=PROCESSING&keyword=员工
```

返回文档状态、当前索引任务阶段、进度、分块数和 MinIO 对象路径，用于页面列表与进度条。

### 2.1 更新文档

```http
POST /documents/{doc_id}/versions
Content-Type: multipart/form-data
```

字段：

- `file`: 新版文件；
- `expected_version`: 用户提交时看到的正式版本号，用于避免并发覆盖。

服务端计算整文件 SHA256；内容与正式版本完全一致时返回 `409`。新版文件使用独立 MinIO 路径，索引任务绑定候选版本，Redis 按实际 Embedding 文本的 SHA256 复用已有向量。候选版本全部完成后才原子切换正式版本，因此更新期间旧版本仍可召回。

### 2.2 历史版本与恢复

```http
GET /documents/{doc_id}/versions
```

返回所有文件版本、产生方式、处理状态、来源版本、失败原因，以及 `is_current` 和 `can_restore`。接口需要知识库 `READ` 权限。

```http
POST /documents/{doc_id}/versions/{version}/restore
Content-Type: application/json

{
  "expected_current_version": 2
}
```

恢复接口需要 `WRITE` 权限。恢复历史 V1 时不会把正式版本号倒退，而是创建新的候选版本，例如当前 V2 会创建 V3，并记录 `operation_type=RESTORE`、`source_version=1`。候选版本复用 V1 的只读 MinIO 文件，重新分块后通过内容 Hash 查询本地缓存和 Redis；只有未命中分块才调用 Embedding 模型。

恢复期间 V2 仍参与召回。V3 全部完成后才原子切换；如果 V3 失败，V2 保持不变，历史接口会展示 V3 的失败原因。

### 3. 余弦召回

```http
POST /retrieval/search
Content-Type: application/json

{
  "query": "员工每年有多少天带薪年假？",
  "kb_ids": [1],
  "metric": "cosine",
  "top_k": 5,
  "min_score": 0.5
}
```

实际响应：

```json
{
  "query": "员工每年有多少天带薪年假？",
  "metric": "cosine",
  "embedding_model": "text-embedding-v3",
  "dimensions": 1024,
  "latency_ms": 42,
  "result_count": 1,
  "results": [
    {
      "rank": 1,
      "chunk_id": 2841,
      "doc_id": 1042,
      "document": "员工手册_2026版.pdf",
      "chunk_index": 12,
      "section": "第四章 · 休假管理",
      "page": 18,
      "content": "……",
      "token_count": 126,
      "score": 0.9284
    }
  ]
}
```

## 异步任务状态

页面使用以下阶段显示任务进度：

| 阶段 | 建议进度 | 含义 |
| --- | ---: | --- |
| `PENDING` | 0% | 等待 Worker |
| `PARSING` | 5% | 下载 MinIO 文件并解析 |
| `CLEANING` | 15% | 清洗无效内容与重复内容 |
| `CHUNKING` | 25% | 结构感知分块 |
| `EMBEDDING` | 30%–85% | 批量生成向量 |
| `PERSISTING` | 88%–98% | 写入 pgvector |
| `DONE` | 100% | 索引可参与召回 |

真实模式可每 1–2 秒调用已有接口 `GET /documents/index-tasks/{task_id}`，在 `DONE`、`FAILED` 或 `CANCELED` 时停止轮询。

## 演示模式

默认地址 `/` 使用真实接口。访问 `/?demo=1` 会切换为演示模式，演示模式不会上传真实文件，也不会写入数据库。
