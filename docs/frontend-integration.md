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

### 0. 知识库列表

```http
GET /knowledge-bases
```

前端上传范围和召回范围均从该接口动态加载，不再使用硬编码名称。当前认证模块尚未实现，因此接口暂时返回全部未删除知识库；接入 JWT 后必须增加创建者、公开范围和 `kb_permission` 权限过滤。

### 1. 上传文档

```http
POST /documents/upload
Content-Type: multipart/form-data
```

字段：

- `files`: 一个或多个文件；
- `kb_id`: 目标知识库 ID；
- `uploaded_by`: 测试用户 ID，默认 1。

服务端会先校验整批文件，再以 UUID 隔离路径流式写入 MinIO，创建 `kb_document` 与 `kb_index_task` 后返回 `202 Accepted`。解析、清洗、分块、Embedding 与入库不会阻塞上传请求。

当前 `uploaded_by` 是测试阶段的表单字段，默认值为 `1`。接入 JWT 后，应删除客户端传值并从服务端登录态中取得用户 ID。

分块参数当前沿用服务端 `RAG_CHUNK_SIZE` 和 `RAG_CHUNK_OVERLAP` 配置，不接受单次上传覆盖，避免同一任务在重试时读取到不同参数。

### 2. 文档列表

```http
GET /documents?kb_id=1&status=PROCESSING&keyword=员工
```

返回文档状态、当前索引任务阶段、进度、分块数和 MinIO 对象路径，用于页面列表与进度条。

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
